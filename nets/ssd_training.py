from random import shuffle
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config import  Config
from utils.box_utils import match, log_sum_exp


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
        num_classes = self.num_classes
        # 创建一个tensor进行处理
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            # 获得框
            truths = targets[idx][:, :-1].data
            # 获得标签
            labels = targets[idx][:, -1].data
            # 获得先验框
            defaults = priors.data
            # 找到标签对应的先验框
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            
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

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

class Generator(object):
    def __init__(self,batch_size,
                 train_lines, image_size,num_classes,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.color_jitter = []
        self.num_classes = num_classes
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y
    
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y
    
    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w = int(w)
        h = min(h, img_h)
        h = int(h)
        
        h = min(h,w)
        w = h
        w_rel = w / img_w
        h_rel = h / img_h
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets
    
    def generate(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            inputs = []
            targets = []
            for annotation_line in lines:  
                line = annotation_line.split()
                img_path = line[0]
                img = imread(img_path).astype('float32')
                shape = np.shape(img)
                y = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                if len(y)==0:
                    continue
                boxes = np.array(y[:,:4],dtype=np.float32)
                boxes[:,0] = boxes[:,0]/shape[1]
                boxes[:,1] = boxes[:,1]/shape[0]
                boxes[:,2] = boxes[:,2]/shape[1]
                boxes[:,3] = boxes[:,3]/shape[0]

                boxes = np.maximum(np.minimum(boxes,1),0)

                y = np.concatenate([boxes,y[:,-1:]],axis=-1)
                if self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                shuffle(self.color_jitter)
                for jitter in self.color_jitter:
                    img = jitter(img)
                if self.lighting_std:
                    img = self.lighting(img)
                if self.hflip_prob > 0:
                    img, y = self.horizontal_flip(img, y)
                if self.vflip_prob > 0:
                    img, y = self.vertical_flip(img, y)
                
                if len(y)==0:
                    continue
                inputs.append(np.transpose(img-MEANS,(2,0,1)))                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets
