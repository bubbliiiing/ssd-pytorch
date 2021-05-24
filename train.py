import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.ssd import get_ssd
from nets.ssd_training import LossHistory, MultiBoxLoss, weights_init
from utils.config import Config
from utils.dataloader import SSDDataset, ssd_dataset_collate

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------#
#   这里看到的train.py和视频上不太一样
#   我重构了一下train.py，添加了验证集
#   这样训练的时候可以有个参考。
#   训练前注意在config.py里面修改num_classes
#   训练世代、学习率、批处理大小等参数在本代码靠下的if True:内进行修改。
#-------------------------------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,criterion,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    loc_loss        = 0
    conf_loss       = 0
    loc_loss_val    = 0
    conf_loss_val   = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   前向传播
            #----------------------#
            out = net(images)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   计算损失
            #----------------------#
            loss_l, loss_c  = criterion(out, targets)
            loss            = loss_l + loss_c
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            loc_loss    += loss_l.item()
            conf_loss   += loss_c.item()

            pbar.set_postfix(**{'loc_loss'  : loc_loss / (iteration + 1), 
                                'conf_loss' : conf_loss / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

                out = net(images)
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)

                loc_loss_val    += loss_l.item()
                conf_loss_val   += loss_c.item()

                pbar.set_postfix(**{'loc_loss'  : loc_loss_val / (iteration + 1), 
                                    'conf_loss' : conf_loss_val / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)

    total_loss  = loc_loss + conf_loss
    val_loss    = loc_loss_val + conf_loss_val

    loss_history.append_loss(total_loss/(epoch_size+1), val_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))

    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda = True
    #--------------------------------------------#
    #   与视频中不同、新添加了主干网络的选择
    #   分别实现了基于mobilenetv2和vgg的ssd
    #   可通过修改backbone变量进行主干网络的选择
    #   vgg或者mobilenet
    #---------------------------------------------#
    backbone = "vgg"

    model = get_ssd("train", Config["num_classes"], backbone)
    weights_init(model)
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    model_path = "model_data/ssd_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    annotation_path = '2007_train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, Cuda)
    loss_history = LossHistory("logs/")

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 5e-4
        Batch_size      = 32
        Init_Epoch      = 0
        Freeze_Epoch    = 50

        optimizer       = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        train_dataset   = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
        val_dataset     = SSDDataset(lines[num_train:], (Config["min_dim"], Config["min_dim"]), False)

        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)

        if backbone == "vgg":
            for param in model.vgg.parameters():
                param.requires_grad = False
        else:
            for param in model.mobilenet.parameters():
                param.requires_grad = False

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Init_Epoch,Freeze_Epoch):
            val_loss = fit_one_epoch(net,criterion,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)

    if True:
        lr              = 1e-4
        Batch_size      = 16
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 100

        optimizer       = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        train_dataset   = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
        val_dataset     = SSDDataset(lines[num_train:], (Config["min_dim"], Config["min_dim"]), False)
        
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)

        if backbone == "vgg":
            for param in model.vgg.parameters():
                param.requires_grad = True
        else:
            for param in model.mobilenet.parameters():
                param.requires_grad = True

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
            
        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            val_loss = fit_one_epoch(net,criterion,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)
