from __future__ import division

from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function, Variable
from utils.box_utils import decode, nms
from utils.config import Config


class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = Config['variance']

    def forward(self, loc_data, conf_data, prior_data):
        #--------------------------------#
        #   先转换成cpu下运行
        #--------------------------------#
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()

        #--------------------------------#
        #   num的值为batch_size
        #   num_priors为先验框的数量
        #--------------------------------#
        num = loc_data.size(0) 
        num_priors = prior_data.size(0)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        #--------------------------------------#
        #   对分类预测结果进行reshape
        #   num, num_classes, num_priors
        #--------------------------------------#
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # 对每一张图片进行处理正常预测的时候只有一张图片，所以只会循环一次
        for i in range(num):
            #--------------------------------------#
            #   对先验框解码获得预测框
            #   解码后，获得的结果的shape为
            #   num_priors, 4
            #--------------------------------------#
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            #--------------------------------------#
            #   获得每一个类对应的分类结果
            #   num_priors,
            #--------------------------------------#
            for cl in range(1, self.num_classes):
                #--------------------------------------#
                #   首先利用门限进行判断
                #   然后取出满足门限的得分
                #--------------------------------------#
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                #--------------------------------------#
                #   将满足门限的预测框取出来
                #--------------------------------------#
                boxes = decoded_boxes[l_mask].view(-1, 4)
                #--------------------------------------#
                #   利用这些预测框进行非极大抑制
                #--------------------------------------#
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
        return output

class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # 获得输入图片的大小，默认为300x300
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        #----------------------------------------#
        #   对feature_maps进行循环
        #   利用SSD会获得6个有效特征层用于预测
        #   边长分别为[38, 19, 10, 5, 3, 1]
        #----------------------------------------#
        for k, f in enumerate(self.feature_maps):
            #----------------------------------------#
            #   分别对6个有效特征层建立网格
            #   [38, 19, 10, 5, 3, 1]
            #----------------------------------------#
            x,y = np.meshgrid(np.arange(f),np.arange(f))
            x = x.reshape(-1)
            y = y.reshape(-1)
            #----------------------------------------#
            #   所有先验框均为归一化的形式
            #   即在0-1之间
            #----------------------------------------#
            for i, j in zip(y,x):
                f_k = self.image_size / self.steps[k]
                #----------------------------------------#
                #   计算网格的中心
                #----------------------------------------#
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #----------------------------------------#
                #   获得小的正方形
                #----------------------------------------#
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                #----------------------------------------#
                #   获得大的正方形
                #----------------------------------------#
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                #----------------------------------------#
                #   获得两个的长方形
                #----------------------------------------#
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        #----------------------------------------#
        #   获得所有的先验框 8732,4
        #----------------------------------------#
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
