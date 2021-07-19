# -----------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要： 该模块主要实现一些第三方库中没有的或者用起来不方便的函数
#   当前版本: 0.0
#   作   者：于川汇 陈瑞侠
#   完成日期: 2018-2-2
# -----------------------------------------------------------------
import cv2
import numpy as np


def fill_hole(binary_img):
    '''
        作用：
            利用opencv的函数对二直图像中的孔洞进行填充
        参数:
            binary_img: 输入的二直图像
        返回值：
              返回的二直图像最大值为1,最小值为0
    '''
    max_val = np.max(binary_img)
    if max_val == 0:  # 如果binary_img的最大值为0，说明没有前景，直接返回原图
        return binary_img
    binary_img = np.uint8(binary_img/max_val)
    (_, contours, _) = cv2.findContours(binary_img.copy(),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(binary_img, contours, contourIdx=-1, color=1,
                            thickness=cv2.FILLED)
