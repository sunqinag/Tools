#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ----------------------------
# !  Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：python_image_processing
#   摘   要：python版图像处理
#   当前版本:2019121918
#   作   者：崔宗会
#   完成日期：2019-12-19
# -----------------------------
import numpy as np
import cv2
import os

from diabetic_package.file_operator.bz_path import get_file_name


def imread(file_path, color_space='RGB'):
    """
    读取file_path的图像，返回R数据
    :param file_path:路径必须是全路径
    :param color_space :目前只支持 gray、rbg、bgr、hsv、lab、yuv 6种颜色空间
    :return:数据
    """
    if not os.path.exists(file_path):
        raise ValueError(file_path + '对应的文件不存在！')

    color_space_lower = color_space.lower()
    if color_space_lower not in ('gray', 'rgb', 'bgr', 'hsv', 'lab',
                                 'yuv'):
        raise ("目前只支持 gray、rbg、bgr、hsv、lab、yuv 6种颜色空间！")

    if color_space_lower == 'gray':
        image_data = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 0)
        return image_data
    else:
        image_bgr = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)

        image_rgb=cv2.cvtColor(image_bgr.copy(),cv2.COLOR_BGR2RGB)
        if color_space_lower == 'rgb':
            return image_rgb
        elif color_space_lower == 'bgr':
            return image_bgr
        elif color_space_lower == 'hsv':
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        elif color_space_lower == 'lab':
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        else:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)


def imwrite(file_path, image_data):
    """

    :param file_path:存储的路径
    :param image_data:图像数据
    :return:
    """
    if not isinstance(image_data,list) and not isinstance(image_data,np.ndarray):
        raise ValueError("image_data的类型为list或者ndarray类型!")

    if len(image_data) == 0:
        raise ValueError("image_data是空列表，请检查!")

    if isinstance(image_data, list):
        image_data = np.array(image_data)

    if ((len(image_data.shape) != 3 and len(image_data.shape) != 2)) or \
            (len(image_data.shape) == 3 and image_data.shape[2] != 3):
        raise ValueError('image_data shape只能为heightXwidth 或者heightXwidthX3')
    if len(image_data.shape)==3 and image_data.shape[2]==3:
        image_data=cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB)

    folder_path = os.path.split(file_path)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    _, ext = get_file_name(file_path, return_ext=True)
    cv2.imencode('.' + ext, image_data)[1].tofile(file_path)


