import os
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils.box_utils import log_sum_exp, match
from utils.config import Config

MEANS = (104, 117, 123)

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, negatives_for_hard=100.0):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.negatives_for_hard = negatives_for_hard
        self.variance = Config['variance']

    def forward(self, predictions, targets):
        #--------------------------------------------------#
        #   取出预测结果的三个值：回归信息，置信度，先验框
        #--------------------------------------------------#
        loc_data, conf_data, priors = predictions
        #--------------------------------------------------#
        #   计算出batch_size和先验框的数量
        #--------------------------------------------------#
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        #--------------------------------------------------#
        #   创建一个tensor进行处理
        #--------------------------------------------------#
        loc_t = torch.zeros(num, num_priors, 4).type(torch.FloatTensor)
        conf_t = torch.zeros(num, num_priors).long()

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            priors = priors.cuda()

        for idx in range(num):
            # 获得真实框与标签
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]

            if(len(truths)==0):
                continue

            # 获得先验框
            defaults = priors
            #--------------------------------------------------#
            #   利用真实框和先验框进行匹配。
            #   如果真实框和先验框的重合度较高，则认为匹配上了。
            #   该先验框用于负责检测出该真实框。
            #--------------------------------------------------#
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        # 所有conf_t>0的地方，代表内部包含物体
        pos = conf_t > 0
        
        #--------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   num_pos  (num, )
        #--------------------------------------------------#
        num_pos = pos.sum(dim=1, keepdim=True)
        
        #--------------------------------------------------#
        #   取出所有的正样本，并计算loss
        #   pos_idx (num, num_priors, 4)
        #--------------------------------------------------#
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        #--------------------------------------------------#
        #   batch_conf  (num * num_priors, num_classes)
        #   loss_c      (num, num_priors)
        #--------------------------------------------------#
        batch_conf = conf_data.view(-1, self.num_classes)
        # 这个地方是在寻找难分类的先验框
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(num, -1)

        # 难分类的先验框不把正样本考虑进去，只考虑难分类的负样本
        loss_c[pos] = 0 
        #--------------------------------------------------#
        #   loss_idx    (num, num_priors)
        #   idx_rank    (num, num_priors)
        #--------------------------------------------------#
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        #--------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   num_pos     (num, )
        #   neg         (num, num_priors)
        #--------------------------------------------------#
        num_pos = pos.long().sum(1, keepdim=True)
        # 限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max = pos.size(1) - 1)
        num_neg[num_neg.eq(0)] =  self.negatives_for_hard
        neg = idx_rank < num_neg.expand_as(idx_rank)

        #--------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   pos_idx   (num, num_priors, num_classes)
        #   neg_idx   (num, num_priors, num_classes)
        #--------------------------------------------------#
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # 选取出用于训练的正样本与负样本，计算loss
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = torch.max(num_pos.data.sum(), torch.ones_like(num_pos.data.sum()))
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)