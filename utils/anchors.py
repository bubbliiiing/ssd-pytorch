import numpy as np


class AnchorBox():
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True):
        self.input_shape = input_shape

        self.min_size = min_size
        self.max_size = max_size

        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0 / ar)

    def call(self, layer_shape, mask=None):
        # --------------------------------- #
        #   获取输入进来的特征层的宽和高
        #   比如38x38
        # --------------------------------- #
        layer_height    = layer_shape[0]
        layer_width     = layer_shape[1]
        # --------------------------------- #
        #   获取输入进来的图片的宽和高
        #   比如300x300
        # --------------------------------- #
        img_height  = self.input_shape[0]
        img_width   = self.input_shape[1]

        box_widths  = []
        box_heights = []
        # --------------------------------- #
        #   self.aspect_ratios一般有两个值
        #   [1, 1, 2, 1/2]
        #   [1, 1, 2, 1/2, 3, 1/3]
        # --------------------------------- #
        for ar in self.aspect_ratios:
            # 首先添加一个较小的正方形
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            # 然后添加一个较大的正方形
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 然后添加长方形
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        # --------------------------------- #
        #   获得所有先验框的宽高1/2
        # --------------------------------- #
        box_widths  = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # --------------------------------- #
        #   每一个特征层对应的步长
        # --------------------------------- #
        step_x = img_width / layer_width
        step_y = img_height / layer_height

        # --------------------------------- #
        #   生成网格中心
        # --------------------------------- #
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        num_anchors_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
        # 获得先验框的左上角和右下角
        anchor_boxes[:, ::4]    -= box_widths
        anchor_boxes[:, 1::4]   -= box_heights
        anchor_boxes[:, 2::4]   += box_widths
        anchor_boxes[:, 3::4]   += box_heights

        # --------------------------------- #
        #   将先验框变成小数的形式
        #   归一化
        # --------------------------------- #
        anchor_boxes[:, ::2]    /= img_width
        anchor_boxes[:, 1::2]   /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)

        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
        return anchor_boxes

#---------------------------------------------------#
#   用于计算共享特征层的大小
#---------------------------------------------------#
def get_vgg_output_length(height, width):
    filter_sizes    = [3, 3, 3, 3, 3, 3, 3, 3]
    padding         = [1, 1, 1, 1, 1, 1, 0, 0]
    stride          = [2, 2, 2, 2, 2, 2, 1, 1]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]
    
def get_mobilenet_output_length(height, width):
    filter_sizes    = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    padding         = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    stride          = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]

def get_anchors(input_shape = [300,300], anchors_size = [30, 60, 111, 162, 213, 264, 315], backbone = 'vgg'):
    if backbone == 'vgg' or backbone == 'resnet50':
        feature_heights, feature_widths = get_vgg_output_length(input_shape[0], input_shape[1])
        aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    else:
        feature_heights, feature_widths = get_mobilenet_output_length(input_shape[0], input_shape[1])
        aspect_ratios = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
        
    anchors = []
    for i in range(len(feature_heights)):
        anchor_boxes = AnchorBox(input_shape, anchors_size[i], max_size = anchors_size[i+1], 
                    aspect_ratios = aspect_ratios[i]).call([feature_heights[i], feature_widths[i]])
        anchors.append(anchor_boxes)

    anchors = np.concatenate(anchors, axis=0)
    return anchors

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    class AnchorBox_for_Vision():
        def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True):
            # 获得输入图片的大小，300x300
            self.input_shape = input_shape

            # 先验框的短边
            self.min_size = min_size
            # 先验框的长边
            self.max_size = max_size

            # [1, 2] => [1, 1, 2, 1/2]
            # [1, 2, 3] => [1, 1, 2, 1/2, 3, 1/3]
            self.aspect_ratios = []
            for ar in aspect_ratios:
                self.aspect_ratios.append(ar)
                self.aspect_ratios.append(1.0 / ar)

        def call(self, layer_shape, mask=None):
            # --------------------------------- #
            #   获取输入进来的特征层的宽和高
            #   比如3x3
            # --------------------------------- #
            layer_height    = layer_shape[0]
            layer_width     = layer_shape[1]
            # --------------------------------- #
            #   获取输入进来的图片的宽和高
            #   比如300x300
            # --------------------------------- #
            img_height  = self.input_shape[0]
            img_width   = self.input_shape[1]
            
            box_widths  = []
            box_heights = []
            # --------------------------------- #
            #   self.aspect_ratios一般有两个值
            #   [1, 1, 2, 1/2]
            #   [1, 1, 2, 1/2, 3, 1/3]
            # --------------------------------- #
            for ar in self.aspect_ratios:
                # 首先添加一个较小的正方形
                if ar == 1 and len(box_widths) == 0:
                    box_widths.append(self.min_size)
                    box_heights.append(self.min_size)
                # 然后添加一个较大的正方形
                elif ar == 1 and len(box_widths) > 0:
                    box_widths.append(np.sqrt(self.min_size * self.max_size))
                    box_heights.append(np.sqrt(self.min_size * self.max_size))
                # 然后添加长方形
                elif ar != 1:
                    box_widths.append(self.min_size * np.sqrt(ar))
                    box_heights.append(self.min_size / np.sqrt(ar))

            print("box_widths:", box_widths)
            print("box_heights:", box_heights)
            
            # --------------------------------- #
            #   获得所有先验框的宽高1/2
            # --------------------------------- #
            box_widths  = 0.5 * np.array(box_widths)
            box_heights = 0.5 * np.array(box_heights)

            # --------------------------------- #
            #   每一个特征层对应的步长
            #   3x3的步长为100
            # --------------------------------- #
            step_x = img_width / layer_width
            step_y = img_height / layer_height

            # --------------------------------- #
            #   生成网格中心
            # --------------------------------- #
            linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
            liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)
            # 构建网格
            centers_x, centers_y = np.meshgrid(linx, liny)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)

            if layer_height == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.ylim(-50,350)
                plt.xlim(-50,350)
                plt.scatter(centers_x,centers_y)

            # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
            num_anchors_ = len(self.aspect_ratios)
            anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
            anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
            
            # 获得先验框的左上角和右下角
            anchor_boxes[:, ::4]    -= box_widths
            anchor_boxes[:, 1::4]   -= box_heights
            anchor_boxes[:, 2::4]   += box_widths
            anchor_boxes[:, 3::4]   += box_heights

            print(np.shape(anchor_boxes))
            if layer_height == 3:
                rect1 = plt.Rectangle([anchor_boxes[4, 0],anchor_boxes[4, 1]],box_widths[0]*2,box_heights[0]*2,color="r",fill=False)
                rect2 = plt.Rectangle([anchor_boxes[4, 4],anchor_boxes[4, 5]],box_widths[1]*2,box_heights[1]*2,color="r",fill=False)
                rect3 = plt.Rectangle([anchor_boxes[4, 8],anchor_boxes[4, 9]],box_widths[2]*2,box_heights[2]*2,color="r",fill=False)
                rect4 = plt.Rectangle([anchor_boxes[4, 12],anchor_boxes[4, 13]],box_widths[3]*2,box_heights[3]*2,color="r",fill=False)
                
                ax.add_patch(rect1)
                ax.add_patch(rect2)
                ax.add_patch(rect3)
                ax.add_patch(rect4)

                plt.show()
            # --------------------------------- #
            #   将先验框变成小数的形式
            #   归一化
            # --------------------------------- #
            anchor_boxes[:, ::2]    /= img_width
            anchor_boxes[:, 1::2]   /= img_height
            anchor_boxes = anchor_boxes.reshape(-1, 4)

            anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
            return anchor_boxes

    # 输入图片大小为300, 300
    input_shape     = [300, 300] 
    # 指定先验框的大小，即宽高
    anchors_size    = [30, 60, 111, 162, 213, 264, 315]
    # feature_heights   [38, 19, 10, 5, 3, 1]
    # feature_widths    [38, 19, 10, 5, 3, 1]
    feature_heights, feature_widths = get_vgg_output_length(input_shape[0], input_shape[1])
    # 对先验框的数量进行一个指定 4，6
    aspect_ratios                   = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]

    anchors = []
    for i in range(len(feature_heights)):
        anchors.append(AnchorBox_for_Vision(input_shape, anchors_size[i], max_size = anchors_size[i+1], 
                    aspect_ratios = aspect_ratios[i]).call([feature_heights[i], feature_widths[i]]))

    anchors = np.concatenate(anchors, axis=0)
    print(np.shape(anchors))
