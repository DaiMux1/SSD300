# %%
from utils.augumentation import Compose, ConvertFromInts,\
    ToAbsoluteCoords,PhotometricDistort, Expand, RandomSampleCrop,\
    RandomMirror, ToPercentCoords, Resize, SubtractMeans
from extract_inform_annotation import Anno_xml 
from make_datapath import make_datapath_list
from lib import *

class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            # Truyền vào Compose 1 list các xử lý
            
            "train": Compose([ConvertFromInts(), # convert image from int to float 32
                              ToAbsoluteCoords(), # back annotation to normal type, ở file trc chia đi để về [0, 1] giờ nhân lại
                              PhotometricDistort(), #change color by random
                              Expand(color_mean),
                              RandomSampleCrop(), # random cắt ảnh
                              RandomMirror(), #xoay ảnh nguọc lại trái sang phải
                              ToPercentCoords(), # chuẩn hóa annotation data về dạng [0, 1]
                              Resize(input_size),
                              SubtractMeans(color_mean) # trừ đi trung bình của BGR
                              ]), 
            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
        
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == '__main__':
    clasess = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                'car', "cat", "chair", "cow", "diningtable", "dog",
                "horse","motorbike","person","pottedplant", "sheep", 'sofa', "train","tvmonitor"]
    
    
    rootpath = './data/VOCdevkit/VOC2012'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(rootpath)
    
    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path) # height, width, channel(bgr)
    # print(img[1])
    height, width, channels = img.shape
    trans_anno = Anno_xml(clasess)
    anno_info_list = trans_anno(train_annotation_list[0], width, height)
    
    #plot original image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()    
    
    # prepare data transform
    color_mean = (104, 117, 123)
    input_size = 300    
    
    transform = DataTransform(input_size, color_mean)
    
    
    # transform img
    phase = "train"
    # train_annotation_list = np.array(train_annotation_list)
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show() 
    #transform val img
    phase = "val"
    img_transformed, boxes, labels = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show() 
# %%

