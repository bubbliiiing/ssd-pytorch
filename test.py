#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
import torch
from torchsummary import summary

from nets import ssd

if __name__ == "__main__":
    model = ssd.get_ssd("train", 21)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
