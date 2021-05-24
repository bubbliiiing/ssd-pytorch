#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
import torch
from torchsummary import summary

from nets import ssd

if __name__ == "__main__":
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ssd.get_ssd("train", 21, backbone_name='mobilenet').to(device)
    summary(model, input_size=(3, 300, 300))
