import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]

'''
该代码用于获得VGG主干特征提取网络的输出。
输入变量i代表的是输入图片的通道数，通常为3。

一般来讲，输入图像为(300, 300, 3)，随着base的循环，特征层变化如下：
300,300,3 -> 300,300,64 -> 300,300,64 -> 150,150,64 -> 150,150,128 -> 150,150,128 -> 75,75,128 -> 75,75,256 -> 75,75,256 -> 75,75,256 
-> 38,38,256 -> 38,38,512 -> 38,38,512 -> 38,38,512 -> 19,19,512 ->  19,19,512 ->  19,19,512 -> 19,19,512
到base结束，我们获得了一个19,19,512的特征层

之后进行pool5、conv6、conv7。
'''
def vgg(i):
    layers = []
    in_channels = i
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers
