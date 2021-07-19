# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：convert_label.py
#   摘   要：rgb_label和class_label的转换
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-7-30
# -----------------------------
import numpy as np

VOC_CLASSES_NAME = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# 20 类 + 背景
VOC_COLOR_TUPLE = ([0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128])





def convert_class_label_to_rgb_label(class_labels, class_num=3,
                                     label_colors=((0, 0, 0), (255, 0, 0),
                                                   (255, 165, 0), (255, 255, 0),
                                                   (0, 255, 0), (0, 127, 255),
                                                   (0, 0, 255), (139, 0, 255))):
    """
    将class的label转为彩色的label
    :param class_labels:数值型标签，输入为4维的numpy ndarray数据,[img_num,height,width,
    channel]
    :param class_num:类别数目,如果前景的类别数为ｎ，此处需加上背景，设置为n+1
    :param label_colors:颜色列表，默认是黑色代表背景，赤、橙、黄、绿、青、蓝、紫代表1到7类，
    当类别数大于8，需要更新该参数
    :return:彩色的标签图像
    """
    if not isinstance(class_labels, np.ndarray) or len(class_labels.shape) != 4:
        raise ValueError('输入必须为4维的ndarray数组！')
    if np.max(class_labels) >= class_num:
        raise ValueError('class_labels中最大类别数大于等于class_num！')

    if class_num > len(label_colors):
        raise ValueError('class_num大于label_colors个数！')

    num_images, height, width = class_labels.shape[:-1]
    outputs = np.zeros((num_images, height, width, 3), dtype=np.uint8)

    for img_index in range(num_images):
        single_class_label = class_labels[img_index]
        label_unique = np.unique(single_class_label)
        for single_class in label_unique:
            class_label_indices = np.where(single_class_label == single_class)
            outputs[img_index, class_label_indices[0], class_label_indices[1], :] = label_colors[single_class]
    return outputs



def convert_rgb_label_to_class_label(rgb_labels, class_num, label_colors=((0, 0, 0), (255, 0, 0),
                                                   (255, 165, 0), (255, 255, 0),
                                                   (0, 255, 0), (0, 127, 255),
                                                   (0, 0, 255), (139, 0, 255))):
    """
    将rbg的label转换为灰度label
    :param rgb_labels:彩色的标签图
    :param class_num:类别数目
    :param label_colors:颜色列表默认是黑色代表背景，赤、橙、黄、绿、青、蓝、紫代表1到7类
    :return: 表示类别的灰度图像
    """
    # TODO:参数检查
    if not isinstance(rgb_labels, np.ndarray) or len(rgb_labels.shape) != 4:
        raise ValueError('输入必须为4维的ndarray数组！')

    if class_num > len(label_colors):
        raise ValueError('class_num大于label_colors个数！')

    num_images, height, width = rgb_labels.shape[:-1]
    outputs = np.zeros((num_images, height, width, 1), dtype=np.uint8)

    rgb_to_label_array = np.zeros(256 ** 3)
    for i, rgb_label in enumerate(label_colors):
        rgb_to_label_array[(rgb_label[0] * 256 + rgb_label[1]) * 256 + rgb_label[2]] = i

    for img_index in range(num_images):
        single_rgb_label_int8 = rgb_labels[img_index]
        single_rgb_label_int32 = single_rgb_label_int8.astype('int32')
        label_array = np.array(
            rgb_to_label_array[
                (single_rgb_label_int32[:, :, 0] * 256 +
                 single_rgb_label_int32[:, :, 1]) * 256 + single_rgb_label_int32[:, :, 2]
                ])
        label_img = np.array(label_array).reshape(height, width, 1)  # 单通道结果
        outputs[img_index, :, :, :] = label_img
    return outputs
