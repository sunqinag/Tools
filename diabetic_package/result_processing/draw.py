# -*- coding: utf-8 -*-
# ----------------------------
# !  Copyright(C) 2019
#   All right reserved.
#   文件名称：draw.py
#   摘   要：进行轮廓绘制
#   当前版本:2019112516
#   作   者：陈瑞侠
#   完成日期：2019-11-25
# -----------------------------
import cv2
import numpy as np
def draw_contours(img, binary_img, color=(255,0,0), line_thickness=2):
    '''
    :param img 输入图像是RGB类型
    :param binary_img:输入和img对应的二值图
    :param color:绘制轮廓对应的颜色，颜色对应顺序也是RGB
    :param line_thickness:绘制线条的宽度
    :return:返回轮廓绘制后的RGB图像
    '''
    if not isinstance(img, np.ndarray) or  not isinstance(binary_img, np.ndarray):
        raise ValueError('输入图像类型必须是ndarray类型！')
    if len(np.unique(binary_img))  > 2:
        raise ValueError('binary_img输入图像必须是二值图！')
    if img.shape[0] != binary_img.shape[0] or img.shape[1] != binary_img.shape[1]:
        raise ValueError('输入图像尺寸不一致!')
    if str(img.dtype) != 'uint8' or str(binary_img.dtype) != 'uint8':
        raise ValueError('输入图像必须是ndarray的uint8类型')
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2. COLOR_RGB2BGR)
    else:
        raise ValueError("输入必须是灰度图或者RGB三通道彩色图!")
    if len(color) != 3:
        raise ValueError('必须传入rgb对应的三通道颜色值！')
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    result_img = img.copy()
    cv2.drawContours(result_img, contours, contourIdx=-1, color=color,
                     thickness=line_thickness, lineType=cv2.LINE_AA)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img


