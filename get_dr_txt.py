from ssd import SSD
from PIL import Image
from utils.box_utils import letterbox_image,ssd_correct_boxes
from torch.autograd import Variable
import torch
import numpy as np
import os
MEANS = (104, 117, 123)
class mAP_SSD(SSD):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.05
        f = open("./input/detection-results/"+image_id+".txt","w") 
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))
        photo = np.array(crop_img,dtype = np.float64)
        # 图片预处理，归一化
        with torch.no_grad():
            photo = Variable(torch.from_numpy(np.expand_dims(np.transpose(crop_img-MEANS,(2,0,1)),0)).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()
            preds = self.net(photo)
        top_conf = []
        top_label = []
        top_bboxes = []
        for i in range(preds.size(1)):
            j = 0
            while preds[0, i, j, 0] >= self.confidence:
                score = preds[0, i, j, 0]
                label_name = self.class_names[i-1]
                pt = (preds[0, i, j, 1:]).detach().numpy()
                coords = [pt[0], pt[1], pt[2], pt[3]]
                top_conf.append(score)
                top_label.append(label_name)
                top_bboxes.append(coords)
                j = j + 1
        # 将预测结果进行解码
        if len(top_conf)<=0:
            return image
        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)


        for i, c in enumerate(top_label):
            predicted_class = c
            score = str(float(top_conf[i]))

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

ssd = mAP_SSD()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")


for image_id in image_ids:
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    image.save("./input/images-optional/"+image_id+".jpg")
    ssd.detect_image(image_id,image)
    print(image_id," done!")
    

print("Conversion completed!")
