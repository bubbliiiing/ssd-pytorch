from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config import  Config
from utils.box_utils import match, log_sum_exp
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import cv2

MEANS = (104, 117, 123)


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
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
        self.variance = Config['variance']

    def forward(self, predictions, targets):
        # 回归信息，置信度，先验框
        loc_data, conf_data, priors = predictions
        # 计算出batch_size
        num = loc_data.size(0)
        # 取出所有的先验框
        priors = priors[:loc_data.size(1), :]
        # 先验框的数量
        num_priors = (priors.size(0))
        # 创建一个tensor进行处理
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            priors = priors.cuda()

        for idx in range(num):
            # 获得框
            truths = targets[idx][:, :-1]
            # 获得标签
            labels = targets[idx][:, -1]
            # 获得先验框
            defaults = priors
            # 找到标签对应的先验框
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        # 转化成Variable
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        # 所有conf_t>0的地方，代表内部包含物体
        pos = conf_t > 0
        # 求和得到每一个图片内部有多少正样本
        num_pos = pos.sum(dim=1, keepdim=True)
        # 计算回归loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # 转化形式
        batch_conf = conf_data.view(-1, self.num_classes)
        # 你可以把softmax函数看成一种接受任何数字并转换为概率分布的非线性方法
        # 获得每个框预测到真实框的类的概率
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = loss_c.view(num, -1)

        loss_c[pos] = 0 
        # 获得每一张图新的softmax的结果
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # 计算每一张图的正样本数量
        num_pos = pos.long().sum(1, keepdim=True)
        # 限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 计算正样本的loss和负样本的loss
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Generator(object):
    def __init__(self,batch_size,
                 train_lines, image_size,num_classes,
                 ):
        
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.num_classes = num_classes-1
        
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []

    def generate(self, train=True):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            inputs = []
            targets = []
            for annotation_line in lines:  
                img,y=self.get_random_data(annotation_line,self.image_size[0:2])

                if len(y)==0:
                    continue
                boxes = np.array(y[:,:4],dtype=np.float32)
                boxes[:,0] = boxes[:,0]/self.image_size[1]
                boxes[:,1] = boxes[:,1]/self.image_size[0]
                boxes[:,2] = boxes[:,2]/self.image_size[1]
                boxes[:,3] = boxes[:,3]/self.image_size[0]

                boxes = np.maximum(np.minimum(boxes,1),0)
                if ((boxes[:,3]-boxes[:,1])<=0).any() and ((boxes[:,2]-boxes[:,0])<=0).any():
                    continue
                
                y = np.concatenate([boxes,y[:,-1:]],axis=-1)

                inputs.append(np.transpose(img-MEANS,(2,0,1)))                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets
