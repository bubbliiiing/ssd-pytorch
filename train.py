from nets.ssd import get_ssd
from nets.ssd_training import Generator,MultiBoxLoss
from torch.utils.data import DataLoader
from utils.dataloader import ssd_dataset_collate, SSDDataset
from utils.config import Config
from torchsummary import summary
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
def adjust_learning_rate(optimizer, lr, gamma, step):
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    # ------------------------------------#
    #   先冻结一部分权重训练
    #   后解冻全部权重训练
    #   先大学习率
    #   后小学习率
    # ------------------------------------#
    lr = 5e-4
    freeze_lr = 1e-4
    Cuda = True

    Start_iter = 0
    Freeze_epoch = 25
    Epoch = 50

    Batch_size = 4
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    model = get_ssd("train",Config["num_classes"])
    
    #-------------------------------------------#
    #   权值文件的下载请看README
    #-------------------------------------------#
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("model_data/ssd_weights.pth", map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    annotation_path = '2007_train.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_train = len(lines)

    if Use_Data_Loader:
        train_dataset = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]))
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)
    else:
        gen = Generator(Batch_size, lines,
                        (Config["min_dim"], Config["min_dim"]), Config["num_classes"]).generate()

    criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, Cuda)
    epoch_size = num_train // Batch_size


    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.vgg.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(net.parameters(), lr=lr)
        for epoch in range(Start_iter,Freeze_epoch):
            if epoch%2==0:
                adjust_learning_rate(optimizer,lr,0.9,epoch)
            loc_loss = 0
            conf_loss = 0
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    if Cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                # 前向传播
                out = net(images)
                # 清零梯度
                optimizer.zero_grad()
                # 计算loss
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                # 反向传播
                loss.backward()
                optimizer.step()
                # 加上
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()

                print('\nEpoch:'+ str(epoch+1) + '/' + str(Freeze_epoch))
                print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Loc_Loss: %.4f || Conf_Loss: %.4f ||' % (loc_loss/(iteration+1),conf_loss/(iteration+1)), end=' ')
                
                
            print('Saving state, iter:', str(epoch+1))
            torch.save(model.state_dict(), 'logs/Epoch%d-Loc%.4f-Conf%.4f.pth'%((epoch+1),loc_loss/(iteration+1),conf_loss/(iteration+1)))

    if True:
        # ------------------------------------#
        #   全部解冻训练
        # ------------------------------------#
        for param in model.vgg.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(net.parameters(), lr=freeze_lr)
        for epoch in range(Freeze_epoch,Epoch):
            if epoch%2==0:
                adjust_learning_rate(optimizer,freeze_lr,0.9,epoch)
            loc_loss = 0
            conf_loss = 0
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    if Cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                # 前向传播
                out = net(images)
                # 清零梯度
                optimizer.zero_grad()
                # 计算loss
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                # 反向传播
                loss.backward()
                optimizer.step()
                # 加上
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()

                print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
                print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Loc_Loss: %.4f || Conf_Loss: %.4f ||' % (loc_loss/(iteration+1),conf_loss/(iteration+1)), end=' ')

            print('Saving state, iter:', str(epoch+1))
            torch.save(model.state_dict(), 'logs/Epoch%d-Loc%.4f-Conf%.4f.pth'%((epoch+1),loc_loss/(iteration+1),conf_loss/(iteration+1)))
