# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：data augmentation
#   摘   要：对图像数据进行增强，包括随机旋转，随机裁剪，随机噪声，平移变换，图像反转,默认输入RGB图像
#   当前版本:2019090311
#   作   者：陈瑞侠,崔宗会
#   完成日期：2019-09-03
# ---------------------------------
'''
功能：
    对分类分割和检测任务进行数据增强，：
要求：
    对分类任务，输入的label类型为int或float
    对分割任务，请输入一个二维的ndarray对象，且label的高宽与image高宽相等
    对检测任务，请输入一个N*5的ndarray对象，其中N为图像中目标个数，5则是bbox的左上右下角坐标（顺序先列后行）和一个类别
'''

import cv2
import numpy as np
from skimage import util


def rotate_image_and_label(image, label=None, rotate_angle=30, center=None):
    """
    对image和label进行旋转
    :param image:输入的图像
    :param label:输入的标签,分类任务输入一个为int或者float的值，
                分割任务输入一张二值图(与原图宽高相等)，
                检测任务输入一个N*5的ndarray其中N为一张图中的目标个数，5则为bbbox左上右下角坐标（顺序先列后行）和目标类别
    :param rotate_angle:旋转的角度
    :param center:旋转中心的坐标，None默认为图像中心
    :return:旋转后的图像(标签)
    """
    height, width = image.shape[:2]
    if center is None:
        center = (height / 2, width / 2)
    rotation_matrix2d = cv2.getRotationMatrix2D(center, rotate_angle, scale=1)
    rotated_img = cv2.warpAffine(image, rotation_matrix2d, (width, height))
    if label is None:
        return rotated_img
    elif isinstance(label, int) or isinstance(label, float):
        return rotated_img, label
    elif len(label.shape) == 2 and label.shape == image.shape[:2]:
        label = cv2.warpAffine(label, rotation_matrix2d, (width, height),
                               flags=cv2.INTER_NEAREST)
        return rotated_img, label
    elif len(label.shape) == 2 and label.shape[1] == 5:
        label = label.astype(np.float64)
        label = __rotate_label(label=label, rotate_angle=rotate_angle, center=center, width=width, height=height)
        return rotated_img, label
    raise ValueError('分割任务label的高宽要与image的高宽一致！',
                     '目标检测任务label的shape要是一个N*5的ndarray对象')


def random_rotate_image_and_label(image, label=None, center=None, min_angle=-30,
                                  max_angle=30):
    """
    对image和label进行随机旋转
    :param image:输入的图像
    :param label:输入的标签
    :param min_angle:旋转角度的最小值
    :param max_angle:旋转角度的最大值
    :param center:旋转中心的坐标，None默认为图像中心
    :return:旋转后的图像(标签)
    """
    rotate_angle = np.random.randint(min_angle, max_angle)
    return rotate_image_and_label(image=image, label=label, center=center, rotate_angle=rotate_angle)


def shear_image_and_label(image, label=None, scale1=-1.0, scale2=1.0):
    """
    对image和label进行剪切变形
    :param image:输入的图像
    :param label:输入的标签
    :param scale1:剪切变形第一个尺度参数
    :param scale2:剪切变形第二个尺度参数
    :return:剪切后的图像(标签)
    """
    height, width = image.shape[:2]
    rotation_matrix2d = np.ones((2, 3))
    rotation_matrix2d[0, 1] = scale1
    rotation_matrix2d[1, 0] = scale2
    deltx = -rotation_matrix2d[0, 1] * height / 2
    delty = - rotation_matrix2d[1, 0] * width / 2

    rotation_matrix2d[0, 2] = deltx
    rotation_matrix2d[1, 2] = delty

    transformed_img = cv2.warpAffine(image, rotation_matrix2d, (width, height))
    if label is None:
        return transformed_img
    elif isinstance(label, int) or isinstance(label, float):
        return transformed_img, label
    elif len(label.shape) == 2 and label.shape == image.shape[:2]:
        label = cv2.warpAffine(label, rotation_matrix2d, (width, height),flags=cv2.INTER_LINEAR)
        return transformed_img, label
    elif len(label.shape) == 2 and label.shape[1] == 5:
        label = label.astype(np.float64)
        bboxes = label[:, :4]
        binary_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i, bbox in enumerate(bboxes):
            img = binary_image.copy()
            img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
            bbox_img = cv2.warpAffine(img, rotation_matrix2d, (img.shape[1], img.shape[0]))
            x, y, w, h = cv2.boundingRect(bbox_img)
            bboxes[i, :] = x, y, x + w, y + h
        label[:, :4] = __clip_box(bboxes, [0, 0, width, height])
        return transformed_img, label
    raise ValueError('分割任务label的高宽要与image的高宽一致！',
                     '目标检测任务label的shape要是一个N*5的ndarray对象')


def random_shear_image_and_label(image, label=None, min_scale=-0.5, max_scale=0.5):
    """
    对image和label进行随机剪切变形
    :param image:输入的图像
    :param label:输入的标签
    :param min_scale:剪切变形尺度参数最小值
    :param max_scale:剪切变形尺度参数最大值
    :return:剪切后的图像(标签)
    """
    scale1 = np.random.uniform(min_scale, max_scale)
    scale2 = np.random.uniform(min_scale, max_scale)
    return shear_image_and_label(image=image, label=label, scale1=scale1, scale2=scale2)

# def resize_image_and_label(image, label=None, width_and_height=(500, 500), interpolation=cv2.INTER_LINEAR):  #resize可以通过计算尺度使用scale来算
#     """
#     对image和label进行缩放变形
#     :param image:输入的图像
#     :param label:输入的标签
#     :param width_and_height:resize的宽高,输入一个包含两个元素的tuple
#     :param interpolation:图像插值方式
#     :return:缩放后的图像(标签)
#     """
#     height, width = image.shape[:2]
#     scale_x = width_and_height[0] / width
#     scale_y = width_and_height[1] / height
#
#     resized_img = cv2.resize(
#         image, width_and_height, interpolation=interpolation)
#     if label is None:
#         return resized_img
#     elif isinstance(label, int) or isinstance(label, float):
#         return resized_img, label
#     elif len(label.shape) == 2 and label.shape == image.shape[:2]:
#         label = cv2.resize(
#             label, width_and_height,
#             interpolation=cv2.INTER_NEAREST)
#         return resized_img, label
#     elif len(label.shape) == 2 and label.shape[1] == 5:
#         label = label.astype(np.float64)
#         label[:, :4] *= [scale_x, scale_y, scale_x, scale_y]
#         return resized_img, label
#     raise ValueError('分割任务label的高宽要与image的高宽一致！'
#                      '目标检测任务label的shape要是一个N*5的ndarray对象')

def resize_image_and_label(image, label=None, width_and_height=(500, 500), interpolation=cv2.INTER_LINEAR):  #resize可以通过计算尺度使用scale来算
    """
    对image和label进行缩放变形
    :param image:输入的图像
    :param label:输入的标签
    :param width_and_height:resize的宽高,输入一个包含两个元素的tuple
    :param interpolation:图像插值方式
    :return:缩放后的图像(标签)
    """
    height, width = image.shape[:2]
    scale_x = width_and_height[0] / width
    scale_y = width_and_height[1] / height
    return scale_image_and_label(image,label,scale_x,scale_y,interpolation=interpolation)

def scale_image_and_label(image, label=None, scale_x=0.5, scale_y=0.5, interpolation=cv2.INTER_LINEAR):
    """
    对image和label进行缩放变形
    :param image:输入的图像
    :param label:输入的标签
    :param scale:缩放的系数
    :param interpolation:图像插值方式
    :return:缩放后的图像(标签)
    """
    scaled_img = cv2.resize(
        image, None, fx=scale_x, fy=scale_y,
        interpolation=interpolation)

    if label is None:
        return scaled_img
    elif isinstance(label, int) or isinstance(label, float):
        return scaled_img, label
    elif len(label.shape) == 2 and label.shape == image.shape[:2]:
        label = cv2.resize(
            label, None, fx=scale_x, fy=scale_y,
            interpolation=cv2.INTER_NEAREST)
        return scaled_img, label
    elif len(label.shape) == 2 and label.shape[1] == 5:
        label = label.astype(np.float64)
        label[:, :4] *= [scale_x, scale_y, scale_x, scale_y]
        return scaled_img, label
    raise ValueError('分割任务label的高宽要与image的高宽一致！'
                     '目标检测任务label的shape要是一个N*5的ndarray对象')


def random_scale_image_and_label(image, label=None, min_scale=0.5, max_scale=2.0,
                                 interpolation=cv2.INTER_LINEAR):
    """
    对image和label进行随机缩放变形
    :param image:输入的图像
    :param label:输入的标签
    :param min_scale:缩放的系数最小值
    :param max_scale:缩放的系数最大值
    :param interpolation:图像差值方式
    :return:缩放后的图像(标签)
    """
    scale_x = np.random.uniform(min_scale, max_scale)
    scale_y = np.random.uniform(min_scale, max_scale)
    return scale_image_and_label(image=image, label=label, scale_x=scale_x, scale_y=scale_y,
                                 interpolation=interpolation)


def flip_image_and_label(image, label=None, flip_num=0):
    """
    对image和label进行翻转
    :param image:输入的图像
    :param label:输入的标签
    :param flip_num:值为－１水平和竖直方向均翻转，０是竖直方向翻转，１是水平方向翻转，其它值不翻转
    :return:翻转后的图像(标签)
    """
    if flip_num in [-1, 0, 1, 2]:
        fliped_image = cv2.flip(image, flip_num)
        if label is None:
            return fliped_image
        elif isinstance(label, int) or isinstance(label, float):
            return image, label
        elif len(label.shape) == 2 and label.shape == image.shape[:2]:
            label = cv2.flip(label, flip_num)
            return fliped_image, label
        elif len(label.shape) == 2 and label.shape[1] == 5:
            label = label.astype(np.float64)
            if flip_num == 1:
                label[:, [2, 0]] = image.shape[1] - label[:, [0, 2]]
                return fliped_image, label
            elif flip_num == 0:
                label[:, [3, 1]] = image.shape[0] - label[:, [1, 3]]
                return fliped_image, label
            elif flip_num == -1:
                label[:, [3, 1]] = image.shape[0] - label[:, [1, 3]]
                label[:, [2, 0]] = image.shape[1] - label[:, [0, 2]]
                return fliped_image, label
            else:
                return image, label
        raise ValueError('分割任务label的高宽要与image的高宽一致！'
                         '目标检测任务label的shape要是一个N*5的ndarray对象')


def random_flip_image_and_label(image, label=None):
    """
    对image和label进行随机翻转
    :param image:输入的图像
    :param label:输入的标签
    :return:翻转后的图像(标签)
    """
    flip_num = np.random.randint(-1, 3)
    return flip_image_and_label(image=image, label=label, flip_num=flip_num)


def crop_image_and_label(image, label=None, crop_height_width=(350, 350), center=None):
    """
    对image和label进行裁剪
    :param image:输入的图像
    :param label:输入的标签
    :param crop_height_width:裁剪的高宽,输入一个包含两个元素的tuple或者list
    :param center:剪切的中心,None默认为图像中心
    :return:剪切后的图像(标签)
    """
    image_height_width = image.shape[:2]
    if len(crop_height_width) != 2:
        raise ValueError('crop_height_width参数是一个包含两个元素的tuple或者list!')
    half_crop_height_width = np.floor(np.array(crop_height_width) / 2.0)

    if center is None:
        center = np.floor(np.array(image_height_width) / 2)

    roi_min_row = center[0] - half_crop_height_width[0]
    roi_max_row = center[0] + half_crop_height_width[0]
    roi_min_col = center[1] - half_crop_height_width[1]
    roi_max_col = center[1] + half_crop_height_width[1]

    roi_min_col = np.int16(np.max([roi_min_col, 0]))
    roi_min_row = np.int16(np.max([roi_min_row, 0]))
    roi_max_col = np.int16(np.min([roi_max_col, image_height_width[1]]))
    roi_max_row = np.int16(np.min([roi_max_row, image_height_width[0]]))

    crop_img = image[roi_min_row: roi_max_row, roi_min_col: roi_max_col]

    if label is None:
        return crop_img
    elif isinstance(label, int) or isinstance(label, float):
        return crop_img, label
    elif len(label.shape) == 2 and image.shape[:2] == label.shape:
        label = label[roi_min_row: roi_max_row, roi_min_col: roi_max_col]
        return crop_img, label
    elif len(label.shape) == 2 and label.shape[1] == 5:
        label = label.astype(np.float64)
        label[:, :4] -= [roi_min_col, roi_min_row, roi_min_col, roi_min_row]
        label[:, :4] = __clip_box(label[:, :4], [0, 0, roi_max_col - roi_min_col, roi_max_row - roi_min_row])
        return crop_img, label
    raise ValueError('分割任务label的高宽要与image的高宽一致！'
                     '目标检测任务label的shape要是一个N*5的ndarray对象')


def random_crop_image_and_label(image, label=None, crop_height_width=(200, 200)):
    """
        对image和label进行随机翻转
        :param image:输入的图像
        :param label:输入的标签
        :param crop_height_width:裁剪的高宽,输入一个包含两个元素的tuple或者list
        :return:翻转后的图像(标签)
    """
    image_height_width = image.shape[:2]
    if len(crop_height_width) != 2:
        raise ValueError('crop_height_width参数是一个包含两个元素的tuple或者list!')
    crop_height = np.min([image_height_width[0], crop_height_width[0]])
    crop_width = np.min([image_height_width[1], crop_height_width[1]])
    crop_height_width_update = (crop_height, crop_width)
    center_delta = np.floor((np.array(image_height_width) - np.array(crop_height_width_update)) / 2)

    center_row = np.random.randint(center_delta[0]) + image_height_width[0] / 2
    center_column = np.random.randint(center_delta[1]) + image_height_width[1] / 2
    return crop_image_and_label(image=image, label=label, crop_height_width=crop_height_width,
                                center=(center_row, center_column))


def random_scale_and_crop_image_and_label(image, label=None, min_scale=0.5, max_scale=2.0,
                                          crop_height_scale=0.8,
                                          crop_width_scale=0.8, interpolation=cv2.INTER_LINEAR):
    """
    对image和label进行随机缩放变形和crop
    :param image:输入的图像
    :param label:输入的标签
    :param min_scale:缩放的系数最小值
    :param max_scale:缩放的系数最大值
    :param crop_height_scale:剪切高度系数
    :param crop_width_scale:剪切宽度系数
    :param interpolation:图像差值方式
    :return:缩放后的图像(标签)
    """
    scale_x = np.random.uniform(min_scale, max_scale)
    scale_y = np.random.uniform(min_scale, max_scale)
    scale_height = np.round(scale_y * np.array(image.shape[0]))
    scale_width = np.round(scale_x * np.array(image.shape[1]))
    crop_height = np.round(scale_height * crop_height_scale)
    crop_width = np.round(scale_width * crop_width_scale)
    if label is None:
        scale_image = scale_image_and_label(image=image, label=label, scale_x=scale_x, scale_y=scale_y,
                                            interpolation=interpolation)
        return random_crop_image_and_label(scale_image, label=None,
                                           crop_height_width=(crop_height, crop_width))
    else:
        scale_image, scale_label = scale_image_and_label(image=image, label=label, scale_x=scale_x, scale_y=scale_y,
                                                         interpolation=interpolation)
        return random_crop_image_and_label(scale_image, label=scale_label,
                                           crop_height_width=(crop_height, crop_width))


def translation_image_and_label(image, label=None, dist_row_scale=0.1, dist_column_scale=0.1):
    """
        对image和label进行平移
        :param image:输入的图像
        :param label:输入的标签
        :param dist_row_scale:行方向平移距离比例系数
        :param dist_column_scale:列方向平移距离比例系数
        :return:平移后的图像(标签)
    """
    if dist_row_scale >= 1 or dist_column_scale >= 1:
        raise ValueError("平移距离超过图像的高宽！")
    height, width = image.shape[:2]
    Matrix = np.float32([[1, 0, np.int32(width * dist_column_scale)], [0, 1, np.int32(height * dist_row_scale)]])
    shifted_img = cv2.warpAffine(
        image, Matrix, (image.shape[1], image.shape[0]))
    if label is None:
        return shifted_img
    elif isinstance(label, int) or isinstance(label, float):
        return shifted_img, label
    elif len(label.shape) == 2 and label.shape == image.shape[:2]:
        label = cv2.warpAffine(label, Matrix, (label.shape[1], label.shape[0]))
        return shifted_img, label
    elif len(label.shape) == 2 and label.shape[1] == 5:
        label = label.astype(np.float64)
        translate_x = np.int32(dist_column_scale * width)
        translate_y = np.int32(dist_row_scale * height)

        label[:, :4] += [translate_x, translate_y, translate_x, translate_y]
        label[:, :4] = __clip_box(label[:, :4], [0, 0, width, height])
        return shifted_img, label
    raise ValueError('分割任务label的高宽要与image的高宽一致！'
                     '目标检测任务label的shape要是一个N*5的ndarray对象')


def random_translation_image_and_label(image, label=None, max_dist=0.5):
    """
        对image和label进行随机平移
        :param image:输入的图像
        :param label:输入的标签
        :param dist_row:行方向平移距离
        :param dist_column:列方向平移距离
        :return:平移后的图像(标签)
    """
    dist_row = np.random.uniform(0, max_dist)
    dist_column = np.random.uniform(0, max_dist)

    return translation_image_and_label(image=image, label=label, dist_row_scale=dist_row,
                                       dist_column_scale=dist_column)


def add_noise(image, mode_list, mean, var):
    '''函数作用:给图片加入噪声
        参数说明：
                mode_list:传入的是要添加的噪声mode的列表
                mode可以选择'gaussian','poisson',
                'salt','pepper','speckle','s&p','localvar'其中任意一种噪声
        mean:噪声随机分布的均值
        var:噪声随机分布的方差
    '''
    result = []
    noise_img = image
    for mode in mode_list:
        if mode == 'gaussian' or mode == 'speckle':  # 高斯噪声
            noise_img = util.random_noise(image, mode=mode, mean=mean, var=var)
        elif mode == 'salt' or mode == 'pepper' or mode == 'salt & pepper':  # 椒盐噪声
            noise_img = util.random_noise(image, mode=mode, amount=var)
        elif mode == 's&p':  # 椒盐高斯混合
            noise_img = util.random_noise(image, mode=mode, amount=mean, salt_vs_pepper=var)
        elif mode == 'localvar':  # 局部亮度改变
            local_vars = np.zeros_like(image) + var
            noise_img = util.random_noise(image, mode=mode, local_vars=local_vars)
        elif mode == 'speckle':  # 斑点噪声
            noise_img = util.random_noise(image, mode=mode, mean=mean, var=var)
        elif mode == 'poisson':  # 泊松噪声
            noise_img = util.random_noise(image, mode=mode)
        result.append(noise_img)
    return  result



def stauration_noise(image, hsv_list=((0.6, 0.6), (0.6, 0.7), (0.4, 0.8))):
    '''函数作用:给图片加入饱和度光照噪声
        :param image:输入的图像
        :param　hsv_list:饱和度参数
        :return:增加饱和度光照噪声的图像
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # 增加饱和度光照的噪声
    hsv[:, :, 0] = hsv[:, :, 0] * (hsv_list[0]
                                   [0] + np.random.random() * hsv_list[0][1])
    hsv[:, :, 1] = hsv[:, :, 1] * (hsv_list[1]
                                   [0] + np.random.random() * hsv_list[1][1])
    hsv[:, :, 2] = hsv[:, :, 2] * (hsv_list[2]
                                   [0] + np.random.random() * hsv_list[2][1])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img


def __rotate_label(label, rotate_angle, center, width, height):
    '''
    旋转label
    :param label: 标注框，为一个N*5的二维数组分别为bbox左上右下点坐标（先列后行）和类别
    :param rotate_angle: 旋转角度
    :param center: 旋转中心点,None默认为图像中心
    :param width: 图像宽度
    :param height: 图像高度
    :return: 经过旋转之后的label
    '''
    theta = rotate_angle * np.pi / 180
    # 找出四个坐标点
    corners = np.array(
        [label[:, 0], label[:, 2], label[:, 2], label[:, 0], label[:, 1], label[:, 1], label[:, 3],
         label[:, 3]]).T
    # 构建仿射矩阵
    rotation_matrix2d = np.array([[np.cos(theta), 0, 0, 0, -np.sin(theta), 0, 0, 0],
                                  [0, np.cos(theta), 0, 0, 0, - np.sin(theta), 0, 0],
                                  [0, 0, np.cos(theta), 0, 0, 0, -np.sin(theta), 0],
                                  [0, 0, 0, np.cos(theta), 0, 0, 0, -np.sin(theta)],
                                  [np.sin(theta), 0, 0, 0, np.cos(theta), 0, 0, 0],
                                  [0, np.sin(theta), 0, 0, 0, np.cos(theta), 0, 0],
                                  [0, 0, np.sin(theta), 0, 0, 0, np.cos(theta), 0],
                                  [0, 0, 0, np.sin(theta), 0, 0, 0, np.cos(theta)]])

    # 转移中心点
    corners[:, :4] -= center[0]
    corners[:, 4:] -= center[1]

    # 计算偏移位置
    corners = np.dot(corners, rotation_matrix2d)

    # 中心回到原点
    corners[:, :4] += center[0]
    corners[:, 4:] += center[1]

    # 计算外接矩形
    bboxes = []
    for corner in corners:
        xmin = np.min(corner[:4])
        xmax = np.max(corner[:4])
        ymin = np.min(corner[4:])
        ymax = np.max(corner[4:])
        bbox = [xmin, ymin, xmax, ymax]
        bboxes.append(bbox)
    bboxes = np.array(bboxes)
    label[:, :4] = __clip_box(bboxes, [0, 0, width, height])
    return label


def __clip_box(bbox, clip_box):
    '''
    边框剪裁
    :param bbox:图像中的目标框左上角和右下角坐标(先列后行)是一个二维的ndarray对象
    :param clip_box: 图像的左上角和右下角坐标（先列后行）是一个一维的array对象
    :return:
    '''
    bbox[:, 0][bbox[:, 0] < clip_box[0]] = clip_box[0]
    bbox[:, 1][bbox[:, 1] < clip_box[1]] = clip_box[1]
    bbox[:, 2][bbox[:, 2] > clip_box[2]] = clip_box[2]
    bbox[:, 3][bbox[:, 3] > clip_box[3]] = clip_box[3]

    bbox[:, 0][bbox[:, 0] > clip_box[2]] = clip_box[2]
    bbox[:, 1][bbox[:, 1] > clip_box[3]] = clip_box[3]
    bbox[:, 2][bbox[:, 2] < clip_box[0]] = clip_box[0]
    bbox[:, 3][bbox[:, 3] < clip_box[1]] = clip_box[1]
    return bbox

if __name__ == '__main__':
    # cls
    image = '/home/xtcsun/PycharmProjects/package_update/bk/cls/class_split/train/0/img/img7_flip_029130_flip_072904_flip_458511.jpg'
    image = cv2.imread(image)
    label = 1
    #单纯图片
    # resize_image = resize_image_and_label(image, None, (227, 227))
    #分类
    # resize_image,resize_label = resize_image_and_label(image,label,(227,227))

    #分割
    image_seg = '/home/xtcsun/PycharmProjects/package_update/bk/seg/split_seg/val/0/img/PHYQT18BD76M1_4_MIC_4_W2_20181201235726_scale_crop_802022_flip_819556.jpg'
    label_seg = '/home/xtcsun/PycharmProjects/package_update/bk/seg/split_seg/val/0/label/PHYQT18BD76M1_4_MIC_4_W2_20181201235726_scale_crop_802022_flip_819556.png'
    image_seg = cv2.imread(image_seg)
    label_seg = cv2.imread(label_seg,0)*255
    resize_image, resize_label = resize_image_and_label(image_seg, label_seg, (227, 227))

    def show_seg(image,label):
        cv2.imshow('image',image)
        cv2.imshow('label',label)
        cv2.waitKey(0)
    show_seg(image_seg,label_seg)
    show_seg(resize_image, resize_label)


    # #目标检测
    # image_det = '/home/xtcsun/PycharmProjects/package_update/bk/det/detection_split/train/0/img/000017.jpg'
    # image_det = cv2.imread(image_det)
    # label_det = [[62,185,199,279,0],[78,90,336,403,5]]
    # label_det = np.array(label_det)
    # resize_image, resize_label = resize_image_and_label(image_det, label_det, (227, 227))
    # print(resize_image.shape)
    # print(resize_label.shape)
    #
    #
    # def show_detect(resize_image,resize_label):
    #     for i in range(len(resize_label)):
    #         p1 = (int(resize_label[i][0]), int(resize_label[i][1]))
    #         p2 = (int(resize_label[i][2]), int(resize_label[i][3]))
    #         print(p1,p2)
    #         cv2.rectangle(resize_image, p1, p2, (255, 0, 0))
    #         p3 = (max(p1[0], 15), max(p1[1], 15))
    #         title = "%s:%d" % ('label::', resize_label[i][-1])
    #         cv2.putText(resize_image, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    #     cv2.imshow("view", resize_image)
    #     cv2.waitKey(0)
    # show_detect(image_det,label_det)
    # show_detect(resize_image, resize_label)
