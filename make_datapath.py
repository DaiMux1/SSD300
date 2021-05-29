from lib import * 

def make_datapath_list(rootpath):
    image_path_template = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annotation_path_template = osp.join(rootpath, "Annotations", "%s.xml")

    train_id_names = osp.join(rootpath, "ImageSets/Main/train.txt")
    val_id_names = osp.join(rootpath, "ImageSets/Main/val.txt")

    train_img_list = []
    train_annotation_list = []

    val_img_list = []
    val_annotation_list = []

    for line in open(train_id_names):
        file_id = line.strip() #xoa ki tu xuong dong, space
        img_path = (image_path_template % file_id) 
        anno_path = (annotation_path_template % file_id)

        train_img_list.append(img_path)
        train_annotation_list.append(anno_path)
        # print(line)
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_template % file_id)

        val_img_list.append(img_path)
        val_annotation_list.append(anno_path)

    return train_img_list, train_annotation_list, val_img_list, val_annotation_list

if __name__ == "__main__":
    rootpath = './data/VOCdevkit/VOC2012'
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(rootpath)

    # print((train_img_list))
    print(val_annotation_list[1])


