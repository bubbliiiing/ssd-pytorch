import torch.nn as nn
from torch.nn.functional import pad

# output=(input−1)×stride+kernel−2×padding+outputpadding

#--------------------------#
#Fusion Block 1
# 150,150,128  -> 150,150,128
# 10, 10, 512  -> 150,150,128
# (150,150,128 + 150,150,128) -> 150,150,128
#in_channels1低层通道数     in_channels2高层通道数      out_channels相加的通道数    最终为in_channels1  
#in_channels1 =128    in_channels2=512    out_channels=128
#--------------------------#
class Fusion1(nn.Module):
    def __init__(self, in_channels1, in_channels2,out_channels):
        super(Fusion1, self).__init__()
        # 10,10,512 -> 150,150,128
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(in_channels2, 128, kernel_size=5, stride=5),
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256,affine=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3)]
        )
        # (150,150,128 + 150,150,128) -> 150,150,128
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1, x2):    #x1 150,150,128     x2 10, 10, 512
        for up in self.up:
            x2 = up(x2)
        x1 = x1
        x  = self.relu(x1+x2)
        return x





#--------------------------#
#Fusion Block 2
# 75,75,256  -> 75,75,128
# 5, 5, 256  -> 75,75,128
# (75,75,128 + 75,75,128) -> 75,75,256
#in_channels1低层通道数     in_channels2高层通道数      out_channels相加的通道数    最终为in_channels1  
#in_channels1 =256    in_channels2=256    out_channels=128
#--------------------------#
class Fusion2(nn.Module):
    def __init__(self, in_channels1, in_channels2,out_channels):
        super(Fusion2, self).__init__()
        # 5, 5, 256  -> 75,75,128
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(in_channels2, 64, kernel_size=5, stride=5),
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128,affine=True),
            nn.ConvTranspose2d(128,64, kernel_size=3, stride=3),
            nn.Conv2d(64,128, kernel_size=3, stride=1,padding=1),]
        )
        # 75,75,256  -> 75,75,128
        self.conv = nn.Conv2d(in_channels1,out_channels,kernel_size=3,padding=1)
        # (75,75,128 + 75,75,128) -> 75,75,256
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels, in_channels1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x1, x2):    #x1 75,75,256     x2 5, 5, 256
        for up in self.up:
            x2 = up(x2)
        x1 = self.conv(x1)
        x  = self.conv_relu(x1+x2)
        return x




#--------------------------#
#Fusion Block 3
# 38,38,512 -> 38,38,256
# 3, 3, 256 -> 38,38,256
# (38,38,256,38,38,256) -> 38,38,512
#in_channels1低层通道数     in_channels2高层通道数      out_channels相加的通道数    最终为in_channels1  
#in_channels1 =512    in_channels2=256    out_channels=256
#--------------------------#
class Fusion3(nn.Module):
    def __init__(self, in_channels1, in_channels2,out_channels):
        super(Fusion3, self).__init__()
        # 3, 3, 256 -> 38,38,256
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(in_channels2, 128, kernel_size=3, stride=3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256,affine=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2,output_padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1)]
        )
        # 38,38,512 -> 38,38,256
        self.conv = nn.Conv2d(in_channels1,out_channels,kernel_size=3,padding=1)
        # (38,38,256,38,38,256) -> 38,38,512
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels, in_channels1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x1, x2):    #x1 38,38,512     x2 3,3,256
        for up in self.up:
            x2 = up(x2)
        x1 = self.conv(x1)
        x  = self.conv_relu(x1+x2)
        return x




#--------------------------#
#Fusion Block 4
# 19,19,1024 -> 19,19,512
# 1, 1, 256  -> 19,19,512
# (19,19,512,19,19,512) -> 19,19,1024
#in_channels1低层通道数     in_channels2高层通道数      out_channels相加的通道数    最终为in_channels1  
#in_channels1 =1024    in_channels2=256    out_channels=512
#--------------------------#
class Fusion4(nn.Module):
    def __init__(self, in_channels1, in_channels2,out_channels):
        super(Fusion4, self).__init__()
        # 1,1,256 -> 19,19,512
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(in_channels2, 128, kernel_size=3, stride=3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256,affine=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2,output_padding=1),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1,padding=1)]
        )
        # 19,19,1024 -> 19,19,512
        self.conv = nn.Conv2d(in_channels1,out_channels,kernel_size=3,padding=1)
        # 19,19,512 -> 19,19,1024
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels, in_channels1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x1, x2):    #x1 19,19,1024     x2 1,1,256
        for up in self.up:
            x2 = up(x2)
        x1 = self.conv(x1)
        x  = self.conv_relu(x1+x2)
        return x


