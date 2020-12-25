import tensorflow as tf
import numpy as np
import os
import cv2
import shutil

from diabetic_package.model_training.estimator.yolov3_detection.dataset import (
    generate_yolo_dataset)
from diabetic_package.model_training.estimator.yolov3_detection.utils import (
    img_transform,bbox_op,config)
from diabetic_package.log.log import bz_log

def parse_simple_data(cur_img_file,cur_label_file,dst_img_size):
    img=cv2.imread(cur_img_file)
    data=np.load(cur_label_file)
    cls,bboxes = data[:, -1],data[:,:4]
    # (xmin,ymin,xmax,ymax) -> (ymin,xmin,ymax,xmax)
    bboxes[:,[0,1,2,3]]=bboxes[:,[1,0,3,2]]
    bboxes = bbox_op.corner_2_center(bboxes)
    return img_transform.padding_img_and_bbox(img, bboxes, dst_img_size),cls

def make_label_func(out_path,train_folder,eval_folder,processed_train_folder,processed_eval_folder,class_num):
    # 进行图片,label的处理,真正喂入神经网络的数据路径
    bz_log.info("生成训练对应的标签格式")
    generate_label_func = generate_yolo_dataset.GenerateYoloLabel()

    for folder in [train_folder,eval_folder]:
        for subfolder in os.listdir(folder):
            if os.path.isdir(folder+os.sep+subfolder):
                print(subfolder)
                #balanced 增强后的数据路径
                sub_img_folder=folder+os.sep+subfolder+os.sep+'augmentation_img'
                sub_label_folder=folder+os.sep+subfolder+os.sep+'augmentation_label'
                #创建一个新的文件夹，用来存储计算后的标签
                if 'train' in folder:
                    img_calced_folder=processed_train_folder+os.sep+subfolder+'/img/'
                    label_calced_folder=processed_train_folder+os.sep+subfolder+'/label/'
                else:
                    img_calced_folder = processed_eval_folder + os.sep + subfolder + '/img/'
                    label_calced_folder = processed_eval_folder + os.sep + subfolder + '/label/'
                if os.path.exists(label_calced_folder):
                    shutil.rmtree(label_calced_folder)
                os.makedirs(label_calced_folder)
                if os.path.exists(img_calced_folder):
                    shutil.rmtree(img_calced_folder)
                os.makedirs(img_calced_folder)

                for k,img_p in enumerate(os.listdir(sub_img_folder)):
                    #对 一对img,label进行计算
                    img_name=img_p.split('.')[0]
                    cur_img_file=sub_img_folder+os.sep+img_p
                    cur_label_file=sub_label_folder+os.sep+img_name+'.npy'

                    bz_log.info('开始处理%d,张图%s', k, img_p)
                    print('开始处理' + str(k)+ '张图,',img_p )
                    bz_log.info("解析数据")

                    #bboxes:[center_y, center_x, height, width]
                    (img,bboxes_),cls_= parse_simple_data(cur_img_file,cur_label_file,config.img_shape[: 2])

                    # bboxes:[center_y, center_x, height, width]->[xmin,ymin,xmax,ymax]
                    bboxes=np.zeros_like(bboxes_)
                    for i,box in enumerate(bboxes_):
                        bboxes[i][0] = box[1] - box[3] / 2.
                        bboxes[i][1] = box[0] - box[2] / 2.
                        bboxes[i][2] = box[1] + box[3] / 2.
                        bboxes[i][3] = box[0] + box[2] / 2.

                    if img.shape != config.img_shape:
                        bz_log.error('输出大小不对%d,%d,%d',img.shape[0], img.shape[1], img.shape[2] )
                        raise ValueError('输出大小不对')

                    #制作3种尺寸的标签
                    print(img_p)
                    label = generate_label_func(
                        np.concatenate((bboxes,np.expand_dims(cls_,1)),axis=-1),
                        num_classes=class_num
                    )
                    cv2.imwrite(img_calced_folder + img_p, img)
                    np.save(label_calced_folder+img_name+'.npy',label)