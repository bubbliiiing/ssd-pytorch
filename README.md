# SSD：Single-Shot MultiBox Detector目标检测模型在Pytorch当中的实现
---

### 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [训练步骤 How2train](#训练步骤)
4. [参考资料 Reference](#Reference)

### 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

### 文件下载
训练所需的ssd_weights.pth可以在百度云下载。  
链接: https://pan.baidu.com/s/1ltXCkuSxKRJUsLi0IoBg2A  
提取码: uqnw 
### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2ssd.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。  
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、就会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置。  
7、在训练前需要修改model_data里面的voc_classes.txt文件，需要将classes改成你自己的classes。  
8、修改train.py里面的NUM_CLASSES与需要训练的种类的个数相同。运行train.py即可开始训练。

### Reference
https://github.com/pierluigiferrari/ssd_keras  
https://github.com/kuhung/SSD_keras  
