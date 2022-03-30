import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#import mff
#from vgg import vgg as add_vgg
from nets import mff
from nets.vgg import vgg as add_vgg


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma      = scale or None
        self.eps        = 1e-10
        self.weight     = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm    = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x       = torch.div(x,norm)
        out     = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def add_extras(in_channels, backbone_name):
    layers = []
    if backbone_name == 'vgg':
        # Block 6
        # 19,19,1024 -> 19,19,256 -> 10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # Block 7
        # 10,10,512 -> 10,10,128 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # Block 8
        # 5,5,256 -> 5,5,128 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        
        # Block 9
        # 3,3,256 -> 3,3,128 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return nn.ModuleList(layers)

class SSD300(nn.Module):
    def __init__(self, num_classes, backbone_name, pretrained = False):
        super(SSD300, self).__init__()
        self.num_classes    = num_classes
        if backbone_name    == "vgg":
            self.vgg        = add_vgg(pretrained)
            self.extras     = add_extras(1024, backbone_name)
            self.upsample   = mff.deconv(256)
            self.fusion     = [mff.Fusion1(128,512,128),mff.Fusion2(256,256,128),
                                mff.Fusion3(512,256,256),mff.Fusion4(1024,256,512)]
            self.fusion1    = mff.Fusion1(128,512,128)
            self.fusion2    = mff.Fusion2(256,256,128)
            self.fusion3    = mff.Fusion3(512,256,256)
            self.fusion4    = mff.Fusion4(1024,256,512)
            #self.L2Norm     = [L2Norm(128, 20),L2Norm(256, 20),L2Norm(512, 20),L2Norm(1024, 20)]
            self.L2Norm     = L2Norm(512,20)
            mbox            = [4, 6, 6, 6, 4, 4]
            
            loc_layers      = []
            conf_layers     = []
            backbone_source = [21, -2]
            #---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            #---------------------------------------------------#
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
            #-------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            #-------------------------------------------------------------#  
            #  k = 2, 3, 4, 5
            for k, v in enumerate(self.extras[1::2], 2):     
                loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
            loc_layers      += [nn.Conv2d(128, 4 * 4, kernel_size = 3, padding = 1),
                                nn.Conv2d(256, 4 * 4, kernel_size = 3, padding = 1),
                                nn.Conv2d(512, 6 * 4, kernel_size = 3, padding = 1),
                                nn.Conv2d(1024, 6 * 4, kernel_size = 3, padding = 1)]
            conf_layers     += [nn.Conv2d(128, 4 * num_classes, kernel_size = 3, padding = 1),
                                nn.Conv2d(256, 4 * num_classes, kernel_size = 3, padding = 1),
                                nn.Conv2d(512, 6 * num_classes, kernel_size = 3, padding = 1),
                                nn.Conv2d(1024, 6 * num_classes, kernel_size = 3, padding = 1)]
        self.loc            = nn.ModuleList(loc_layers)
        self.conf           = nn.ModuleList(conf_layers)
        self.backbone_name  = backbone_name
        
    def forward(self, x):
        #---------------------------#
        #   x是300,300,3
        #---------------------------#
        sources = list()
        loc     = list()
        conf    = list()
        fusion_1= list()        # 存储低层特征图
        fusion_2= list()        # 存储高层特征图


        #---------------------------#
        #   先经过vgg网络部分
        #   vgg网络分为四个part
        #   每个part提取一个低层特征图
        #---------------------------#

        #   Part 1   layers : 0-8
        #   获得第一个低层特征图的内容  
        #   shape为150,150,128
        #   融合
        if self.backbone_name == "vgg":
            for k in range(9):
                x = self.vgg[k](x)
        s = x
        fusion_1.append(s)

        #   Part 2   layers : 9-15
        #   获得第二个高层特征图的内容   
        #   shape为75，75,256
        #   融合
        if self.backbone_name == "vgg":
            for k in range(9,16):
                x = self.vgg[k](x)
        s = x
        fusion_1.append(s)

        #   Part 3    layers : 16-22
        #   获得第三个高层特征图的内容   
        #   shape为38,38,512
        #   融合  及  预测
        if self.backbone_name == "vgg":
            for k in range(16,23):
                x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)
        fusion_1.append(s)

        #   Part 4    layers : 23-end
        #   获得第四个高层特征图的内容   
        #   shape为19,19,1024
        #   融合  及  预测
        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)      #<class 'torch.Tensor'>  torch.Size([1, 1024, 19, 19])
        s = x
        sources.append(s)
        fusion_1.append(s)    


        #-------------------------------------------------------------#
        #   经过 extra 网络
        #   第1层、第3层、第5层、第7层的特征层  可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#              
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg":
                if k % 2 == 1:
                    sources.append(x)

        #-------------------------------------------------------------#
        #   经过  upsample  上采样网络
        #   输入层、第1层、第3层、第5层的特征层，为待融合的高层特征册
        #   shape分别为(1,1,256), (3,3,256), (5,5,256), (10,10,512)
        #-------------------------------------------------------------#     
        fusion_2.append(x)        
        for k,f in enumerate(self.upsample):      
            x = F.relu(f(x),inplace=True)
            if k%2 == 1:
                fusion_2.append(x)


        #-------------------------------------------------------------#
        #   获得 4个 融合特征层
        #   shape分别为(150,150,128), (75,75,256), (38,38,512), (19,19,1024)
        #-------------------------------------------------------------#  
        '''   
        for n,fusion in enumerate(self.fusion):
            feature = fusion(fusion_1[n],fusion_2[3-n])
            sources.append(feature)'''
        feature1 = self.fusion1(fusion_1[0],fusion_2[3])
        sources.append(feature1)
        feature2 = self.fusion2(fusion_1[1],fusion_2[2])
        sources.append(feature2)
        feature3 = self.fusion3(fusion_1[2],fusion_2[1])
        sources.append(feature3)
        feature4 = self.fusion4(fusion_1[3],fusion_2[0])
        sources.append(feature4)

        #-------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        #-------------------------------------------------------------#     
        for (x, l, c) in zip(sources, self.loc, self.conf):
            
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #-------------------------------------------------------------#
        #   进行reshape方便堆叠
        #-------------------------------------------------------------#  
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #-------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        #-------------------------------------------------------------#     
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output

if __name__=="__main__":
    print("ok")