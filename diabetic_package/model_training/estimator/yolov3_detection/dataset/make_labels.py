import tensorflow as tf
import numpy as np
import os
import cv2

from diabetic_package.model_training.estimator.yolov3_detection.dataset import (
    generate_yolo_dataset,tfexample_converter)
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

def get_record_list(folder):
    '''获取增强文件夹中子文件夹的所有img,label路径'''
    img_list, label_list= [], []
    for subfolder in os.listdir(folder):
        if os.path.isdir(folder + os.sep + subfolder):
            # balanced 增强后的数据路径
            sub_img_folder = folder + os.sep + subfolder + os.sep + 'augmentation_img'
            sub_label_folder = folder + os.sep + subfolder + os.sep + 'augmentation_label'
            for img_p in os.listdir(sub_img_folder):
                # 对 一对img,label进行计算
                img_name = img_p.split('.')[0]
                cur_img_file = sub_img_folder + os.sep + img_p
                cur_label_file = sub_label_folder + os.sep + img_name + '.npy'
                # 添加路径
                img_list.append(cur_img_file)
                label_list.append(cur_label_file)

    #进行shuffle
    img_list,label_list=np.array(img_list),np.array(label_list)
    shuffle_indices = np.arange(len(img_list))
    np.random.shuffle(shuffle_indices)
    img_list = img_list[shuffle_indices]
    label_list = label_list[shuffle_indices]

    return img_list, label_list

def pack_func(img_list,label_list,generate_label_func,class_num,saved_folder,flag,file_in_one_tfrecord=50):
    '''把train,eval分别封装进各自的文件夹下的固定数目的子tfrecords'''
    num_count = 0
    for cur_img_file,cur_label_file in zip(img_list,label_list):
        if num_count%file_in_one_tfrecord==0:
            #重定向writer
            sub_record_num=num_count//file_in_one_tfrecord
            writer=tf.python_io.TFRecordWriter(saved_folder + os.sep +\
                                               flag+str(sub_record_num)+'.tfrecords')

        # bboxes:[center_y, center_x, height, width]
        (img, bboxes_), cls_ = parse_simple_data(cur_img_file, cur_label_file, config.img_shape[: 2])
        # bboxes:[center_y, center_x, height, width]->[xmin,ymin,xmax,ymax]
        bboxes = np.zeros_like(bboxes_)
        for i, box in enumerate(bboxes_):
            bboxes[i][0] = box[1] - box[3] / 2.
            bboxes[i][1] = box[0] - box[2] / 2.
            bboxes[i][2] = box[1] + box[3] / 2.
            bboxes[i][3] = box[0] + box[2] / 2.
        if img.shape != config.img_shape:
            bz_log.error('输出大小不对%d,%d,%d', img.shape[0], img.shape[1], img.shape[2])
            raise ValueError('输出大小不对')
        # 制作3种尺寸的标签,共6个矩阵需要封装
        # (52,52,3,25),(26,26,3,25),(13,13,3,25),(150,4),(150,4),(150,4)
        label = generate_label_func(
            np.concatenate((bboxes, np.expand_dims(cls_, 1)), axis=-1),
            num_classes=class_num)

        # 在这里写入tfrecords
        simple_example_proto = tfexample_converter.serialize_yolo_example(img, label)
        writer.write(simple_example_proto)

        num_count += 1
        if num_count % file_in_one_tfrecord == 0:
            writer.close()

def make_label_func(train_folder,eval_folder,processed_train_folder,processed_eval_folder,class_num):
    # 进行图片,label的处理,真正喂入神经网络的数据路径
    bz_log.info("生成标签...")
    generate_label_func = generate_yolo_dataset.GenerateYoloLabel()
    file_in_one_tfrecord=50 #用来规定单个tfrecord中的文件数

    #添加训练,验证的所有图片,标签路径
    for folder in [train_folder, eval_folder]:
        if 'train' in folder:
            train_img_list,train_label_list=get_record_list(folder)
        elif 'val' in folder:
            eval_img_list, eval_label_list = get_record_list(folder)

    if not os.path.exists(processed_train_folder):
        os.makedirs(processed_train_folder)
    if not os.path.exists(processed_eval_folder):
        os.makedirs(processed_eval_folder)

    #把balanced_train所有文件封装到processed_train_folder下的多个tfrecords
    pack_func(train_img_list, train_label_list, generate_label_func, class_num,processed_train_folder,
              flag='train_',file_in_one_tfrecord=file_in_one_tfrecord)
    # 把balanced_eval所有文件封装到processed_eval_folder下的多个tfrecords
    pack_func(eval_img_list, eval_label_list, generate_label_func, class_num, saved_folder=processed_eval_folder,
              flag='eval_',file_in_one_tfrecord=file_in_one_tfrecord)

