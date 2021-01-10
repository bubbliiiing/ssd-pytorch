import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config import Config

from nets.ssd_layers import Detect, L2Norm, PriorBox
from nets.vgg import vgg as add_vgg


class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes, confidence, nms_iou):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = Config
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, confidence, nms_iou)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        #---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        #---------------------------#
        for k in range(23):
            x = self.vgg[k](x)
        
        #---------------------------#
        #   conv4_3的内容
        #   需要进行L2标准化
        #---------------------------#
        s = self.L2Norm(x)
        sources.append(s)

        #---------------------------#
        #   获得conv7的内容
        #   shape为19,19,1024
        #---------------------------#
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        #-------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#      
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        #-------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        #-------------------------------------------------------------#      
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #-------------------------------------------------------------#
        #   进行reshape方便堆叠
        #-------------------------------------------------------------#  
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #-------------------------------------------------------------#
        #   loc会reshape到batch_size,num_anchors,4
        #   conf会reshap到batch_size,num_anchors,self.num_classes
        #   如果用于预测的话，会添加上detect用于对先验框解码，获得预测结果
        #   不用于预测的话，直接返回网络的回归预测结果和分类预测结果用于训练
        #-------------------------------------------------------------#     
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors              
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

def add_extras(i, batch_norm=False):
    layers = []
    in_channels = i

    # Block 6
    # 19,19,1024 -> 10,10,512
    layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # Block 7
    # 10,10,512 -> 5,5,256
    layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # Block 8
    # 5,5,256 -> 3,3,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    
    # Block 9
    # 3,3,256 -> 1,1,256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return layers

mbox = [4, 6, 6, 6, 4, 4]

def get_ssd(phase, num_classes, confidence=0.5, nms_iou=0.45):
    #---------------------------------------------------#
    #   add_vgg指的是加入vgg主干特征提取网络。
    #   该网络的最后一个特征层是conv7后的结果。
    #   shape为19,19,1024。
    #
    #   为了更好的提取出特征用于预测。
    #   SSD网络会继续进行下采样。
    #   add_extras是额外下采样的部分。   
    #---------------------------------------------------#
    vgg, extra_layers = add_vgg(3), add_extras(1024)

    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    #---------------------------------------------------#
    #   在add_vgg获得的特征层里
    #   第21层和-2层可以用来进行回归预测和分类预测。
    #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
    #---------------------------------------------------#
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        mbox[k] * num_classes, kernel_size=3, padding=1)]
           
    #-------------------------------------------------------------#
    #   在add_extras获得的特征层里
    #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
    #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
    #-------------------------------------------------------------#             
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                  * num_classes, kernel_size=3, padding=1)]

    #-------------------------------------------------------------#
    #   add_vgg和add_extras，一共获得了6个有效特征层，shape分别为：
    #   (38,38,512), (19,19,1024), (10,10,512), 
    #   (5,5,256), (3,3,256), (1,1,256)
    #-------------------------------------------------------------#
    SSD_MODEL = SSD(phase, vgg, extra_layers, (loc_layers, conf_layers), num_classes, confidence, nms_iou)
    return SSD_MODEL
