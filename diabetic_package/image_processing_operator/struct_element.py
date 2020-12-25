# ------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要: 根据给定的角度和直线的长度生成线性结构元素
#   当前版本:0.0
#   作   者：于川汇 陈瑞侠
#   完成日期：2018-1-26
# ------------------------------------------------------------
import numpy as np
import math


def get_line_kernel(l, deg):
    '''
    作用
        生成线型结构元素
    参数
        l：生成直线的长度
        deg：直线相对于x轴正方向的角度
    方法
        遍历短边上每一列的点，找到这一列上与中心形成向量的角度与
        angle偏差最小d点置为1。
    原理
        使用短边计算角度于angle偏移最小的点，可以保证生成d直线不会断
    '''
    size_y, size_x = _calc_kernel_size(l, deg)
    kernel = np.zeros((size_y, size_x), dtype=int)
    ori = (math.floor(size_y/2), math.floor(size_x/2))
    kernel[ori] = 1
    if size_x < size_y:
        deg = _convert_angle_0_180(deg)
        for y in range(ori[0]):
            min_deg_diff = 180
            min_deg_x = 0
            for x in range(size_x):
                # 由于numpy array的索引是以左上角为原点，所以求取角度时，y方向要取反
                deg_diff = abs(
                    math.degrees(
                        math.atan2(
                            ori[0] - y,
                            x - ori[1])) - deg)
                if deg_diff < min_deg_diff:
                    min_deg_diff = deg_diff
                    min_deg_x = x
            kernel[y, min_deg_x] = 1
            kernel[ori[0] - (y - ori[0]), ori[1] - (min_deg_x - ori[1])] = 1
    else:
        deg = _convert_angle_n90_90(deg)
        for x in range(ori[1] + 1, size_x):
            min_deg_diff = 180
            min_deg_y = 0
            for y in range(size_y):
                deg_diff = abs(
                    math.degrees(
                        math.atan2(
                            ori[0] - y,
                            x - ori[1])) - deg)
                if deg_diff < min_deg_diff:
                    min_deg_diff = deg_diff
                    min_deg_y = y
            kernel[min_deg_y, x] = 1
            kernel[ori[0] - (min_deg_y - ori[0]),
                   ori[1] - (x - ori[1])] = 1  # 将中心对称的点置为1
    return kernel.astype(np.uint8)


def _convert_angle_n90_90(angle):
    angle = _convert_angle_0_180(angle)
    if angle >= 90:
        angle -= 180
    return angle


def _convert_angle_0_180(angle):
    angle %= 360
    if angle >= 180:
        angle -= 180
    return angle


def _calc_kernel_size(l, deg):
    '''
    根据长度l和角度deg生成一个宽是距离l*sin(deg)最近奇数，
    高是距离l*cos(deg)最近的奇数
    '''
    rad = math.radians(deg)
    size_x = _get_closest_odd(l * abs(math.cos(rad)))
    size_y = _get_closest_odd(l * abs(math.sin(rad)))
    return (size_y, size_x)


def _get_closest_odd(num):
    '''
        根据num生成最近的奇数
    '''
    num_int = math.floor(num)
    if num_int % 2 == 0:
        return num_int + 1
    else:
        return num_int
