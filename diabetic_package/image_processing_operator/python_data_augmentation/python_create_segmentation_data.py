# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：python_create_segmentation_data.py
#   摘   要：利用缺陷生成图像, 包括对缺陷进行随机旋转，随机裁剪，随机噪声，随机缩放，反转,
#           然后将其随机放置到目标图像的任意位置,默认输入RGB图像
#   当前版本:2.0
#   作   者：王茜
#   完成日期：2018-08-29
# ---------------------------------

import numpy as np
import cv2
import copy
from .. import region_operator
from . import python_base_data_augmentation


def get_defect_difference_from_background(defect_img, defect_label):
    """
    :param defect_img: 缺陷图像
    :param defect_label: 缺陷图像标签
    :return:
    """

    if defect_img.shape[:2] != defect_label.shape:
        raise ValueError('缺陷图像与缺陷标签图像高宽必须一致！')
    binary = (defect_label > 0).astype(np.int32)
    binary_inv = 1 - binary
    if len(defect_img.shape) != 3:
        defect_img = np.expand_dims(defect_img, axis=2)
    mean_intensity_list = []
    for i in range(defect_img.shape[2]):
        regions = region_operator.region_props(
            binary_image=binary_inv, intensity_img=defect_img[:, :, i])
        all_area = 0
        all_intensity = 0
        for region in regions:
            area = region.area
            mean_intensity = region.mean_intensity
            all_area += area
            all_intensity += mean_intensity * area
        if all_area == 0:
            raise ZeroDivisionError('除数all_area不能为0！')
        all_mean_intensity = all_intensity / all_area
        mean_intensity_list.append(all_mean_intensity)
    return np.squeeze(defect_img - np.array(mean_intensity_list))


def add_noise_to_defect(defect_img,
                        gauss_params=(0.000000001, 0.00000001),
                        local_params=(0.000000001, 0.000000006),
                        salt_params=(0.00001, 0.000015)):
    """
    :param defect_img: 缺陷与背景的差值
    :param gauss_params: gauss噪声params
    :param local_params: local噪声params
    :param salt_params: salt噪声params
    :return:
    """
    random_noise = np.random.randint(0, 3)
    if random_noise == 0:
        return defect_img
    elif random_noise == 1:
        defect_noise = python_base_data_augmentation.add_noise(
            defect_img.copy(), ['gaussian'],
            mean=gauss_params[0], var=gauss_params[1])
    elif random_noise == 2:
        defect_noise = python_base_data_augmentation.add_noise(
            defect_img.copy(), ['localvar'],
            mean=local_params[0], var=local_params[1])
    else:
        defect_noise = python_base_data_augmentation.add_noise(
            defect_img.copy(), ['salt'],
            mean=salt_params[0], var=salt_params[1])
    return np.uint8(defect_noise[0] * 255)


# 在目标图像上平移缺陷
def random_locate_defect_to_img(img,
                                label,
                                defect_difference,
                                defect_label):
    """
    将缺陷的坐标随机放置到目标图像上
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景的差值
    :param defect_label: 缺陷图像标签
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    height, width = np.shape(img)[:2]
    defect_height, defect_width = np.shape(defect_difference)[:2]
    img_convert = img.astype(np.float32)
    label_convert = label.astype(np.float32)
    img_convert_copy = img_convert.copy()
    new_label = np.zeros([height, width])
    # 增加while循环
    while(1):
        location_height = np.random.randint(0, height - defect_height)
        location_width = np.random.randint(0, width - defect_width)
        # 分别求取每个缺陷
        binary = (defect_label > 0).astype(np.int32)
        regions = region_operator.region_props(binary)
        for region in regions:
            coords = region.coords
            img_convert_copy[location_height + coords[:, 0],
                             location_width + coords[:, 1]] += \
                defect_difference[coords[:, 0], coords[:, 1]]
            img_convert_copy[img_convert_copy < 0] = 0
            img_convert_copy[img_convert_copy > 255] = 255
            new_label[
                location_height + coords[:, 0], location_width + coords[:,
                                                                 1]] = \
                defect_label[coords[:, 0], coords[:, 1]]
        if (np.sum(np.multiply(label_convert, new_label)) == 0):
            break
        new_label = np.zeros([height, width])
    return np.uint8(img_convert_copy), np.uint8(label_convert + new_label)


# 在目标图像上平移缺陷
def random_locate_defect_to_img_bbox(img,
                                     label,
                                     defect_difference,
                                     defect_label):
    """
    将缺陷的bbox所生成的矩形区域随机放置到目标图像上
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景的差值
    :param defect_label: 缺陷图像标签
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    height, width = np.shape(img)[:2]
    defect_height, defect_width = np.shape(defect_difference)[:2]
    img_convert = img.astype(np.float32)
    label_convert = label.astype(np.float32)
    img_convert_copy = img_convert.copy()
    new_label = np.zeros([height, width])
    # 增加while循环
    while(1):
        location_height = np.random.randint(0, height - defect_height)
        location_width = np.random.randint(0, width - defect_width)
        # 分别求取每个缺陷
        binary = (defect_label > 0).astype(np.int32)
        regions = region_operator.region_props(binary)
        for region in regions:
            bbox = np.array(region.bbox)
            img_convert_copy[location_height+bbox[0]: location_height + bbox[2],
                             location_width+bbox[1]: location_width + bbox[3]] \
                += defect_difference[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            img_convert_copy[img_convert_copy < 0] = 0
            img_convert_copy[img_convert_copy > 255] = 255
            new_label[
            location_height + bbox[0]: location_height + bbox[2],
            location_width + bbox[1]: location_width + bbox[3]] = \
                defect_label[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        if (np.sum(np.multiply(label_convert, new_label)) == 0):
            break
        new_label = np.zeros([height, width])
    img_fuzzy = fuzzy_defect_boundary_from_img(img_convert_copy,
                                               defect_difference,
                                               location_height,
                                               location_width,
                                               10,
                                               10)
    return np.uint8(img_fuzzy), np.uint8(label_convert + new_label)

# 增加差值图像后做模糊
def fuzzy_defect_boundary_from_img(img,
                                   defect_difference,
                                   row,
                                   column,
                                   extend_height=10,
                                   extend_width=10):
    """
    :param img: 目标图像
    :param defect_difference: 缺陷与背景的差值
    :param column: 缺陷图像在目标图像的起始行
    :param row: 缺陷图像在目标图像的起始列
    :param extend_height: 扩张高度
    :param extend_width: 扩张宽度
    :return:
    """
    height, width = img.shape[:2]
    defect_height, defect_width = defect_difference.shape[:2]
    # 考虑边界问题
    left_column = np.maximum(0, row - extend_height)
    right_column = np.minimum(row + defect_height + extend_height, height)
    left_row = np.maximum(0, column - extend_width)
    right_row = np.minimum(column + defect_width + extend_width, width)
    extend_defect_img = img[left_column: right_column, left_row: right_row]
    extend_defect_height, extend_defect_width = extend_defect_img.shape[:2]
    extend_defect_img_copy = copy.deepcopy(extend_defect_img)
    fuzzy_defect_img = cv2.blur(extend_defect_img_copy, (3, 3))
    # 将扩张后的模糊缺陷放回到目标图像上
    img[left_column: right_column, left_row: right_row] = fuzzy_defect_img
    # # 将原始缺陷图像放回到目标图像上
    # img[left_column + extend_height: right_column - extend_height,
    #     left_row + extend_width: right_row - extend_width] = \
    # extend_defect_img_copy[extend_height: extend_defect_height - extend_height,
    #                   extend_width: extend_defect_width - extend_width]

    # 将原始缺陷图像放回到目标图像上
    img[left_column + 2*extend_height: right_column - 2*extend_height,
    left_row + 2*extend_width: right_row - 2*extend_width] = \
        extend_defect_img_copy[2*extend_height: extend_defect_height - 2*extend_height,
        2*extend_width: extend_defect_width - 2*extend_width]
    return img

# 在目标图像上翻转缺陷
def random_flip_defect_to_img(img,
                              label,
                              defect_difference,
                              defect_label):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景的差值
    :param defect_label: 缺陷图像标签
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    flip_type = np.random.randint(-1, 3)
    if flip_type == 2:
        return random_locate_defect_to_img(img,
                                           label,
                                           defect_difference,
                                           defect_label)
    flip_defect = cv2.flip(defect_difference, flip_type)
    flip_defect_label = cv2.flip(defect_label, flip_type)
    return random_locate_defect_to_img(
        img, label, flip_defect, flip_defect_label)

# 在目标图像上缩放缺陷
def random_zoom_defect_to_img(img,
                              label,
                              defect_difference,
                              defect_label=None,
                              min_scale=0.5,
                              max_scale=2):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景的差值
    :param defect_label: 缺陷图像标签
    :param min_scale: 缩放最小值比例
    :param max_scale: 缩放最大值比例
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if min_scale < 0 or max_scale < 0:
        raise ValueError('min_scale, max_scale必须大于0！')
    if min_scale > max_scale:
        raise ValueError('min_angle必须大于max_angle!')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    level_scale, vertial_scale = np.random.uniform(min_scale, max_scale, [2])
    defect_height, defect_width = np.shape(defect_difference)[:2]
    scale_height = (np.round(defect_height * level_scale)).astype(np.int32)
    scale_width = (np.round(defect_width * vertial_scale)).astype(np.int32)
    scale_defect = cv2.resize(defect_difference,
                              (scale_width, scale_height),
                              interpolation=cv2.INTER_LINEAR)
    scale_defect_label = cv2.resize(defect_label,
                                    (scale_width, scale_height),
                                    interpolation=cv2.INTER_NEAREST)
    return random_locate_defect_to_img(
        img, label, scale_defect, scale_defect_label)


# 在目标图像旋转缺陷
def random_rotate_defect_to_img(img,
                                label,
                                defect_difference,
                                defect_label=None,
                                min_angle=0,
                                max_angle=360):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景差值
    :param defect_label: 缺陷图像标签
    :param min_angle: 旋转最小角度
    :param max_angle: 旋转最大角度
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if min_angle > max_angle:
        raise ValueError('min_angle必须大于max_angle!')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    # 随机旋转缺陷
    rotation_angle = np.random.uniform(min_angle, max_angle)
    def get_extension_img(img):
        height, width = np.shape(img)[:2]
        max_length = np.maximum(height, width)
        if len(img.shape) != 3:
            extension_img = np.zeros([max_length, max_length])
        else:
            extension_img = np.zeros([max_length, max_length, img.shape[2]])
        extension_img[np.int32(max_length / 2 - height / 2):
                      np.int32(max_length / 2 + height / 2),
                      np.int32(max_length / 2 - width / 2):
                      np.int32(max_length / 2 + width / 2)] = \
            img[: height, : width]
        return extension_img
    extension_defect = get_extension_img(defect_difference)
    extension_height, extension_width = np.shape(extension_defect)[:2]
    rotation_center = (extension_height / 2, extension_width / 2)
    rotation_matrix2d = cv2.getRotationMatrix2D(
        rotation_center, rotation_angle, scale=1)
    rotation_defect = cv2.warpAffine(extension_defect,
                                     rotation_matrix2d,
                                     (extension_width, extension_height))
    extension_defect_label = get_extension_img(defect_label)
    rotation_defect_label = cv2.warpAffine(extension_defect_label,
                                           rotation_matrix2d,
                                           (extension_width, extension_height),
                                           flags=cv2.INTER_NEAREST)
    return random_locate_defect_to_img(
        img, label, rotation_defect, rotation_defect_label)


# 在目标图像上拉伸缺陷
def random_shear_defect_to_img(img,
                               label,
                               defect_difference,
                               defect_label=None,
                               min=-1.0,
                               max=1.0):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景差值
    :param defect_label: 缺陷图像标签
    :param min1: 最小值
    :param max1: 最大值
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if min > max:
        raise ValueError('min_angle必须小于max_angle!')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    level_scale, vertical_scale, level_angle, vertical_angle = \
        np.random.uniform(min, max, [4])
    M = np.array(
        [[level_scale, level_angle, 0], [vertical_angle, vertical_scale, 0]],
        dtype=np.float32)
    shear_defect = cv2.warpAffine(defect_difference, M, (
        np.max(defect_difference.shape), np.max(defect_difference.shape)))

    shear_defect_label = cv2.warpAffine(defect_label, M, (
        np.max(defect_label.shape), np.max(defect_label.shape)),flags=cv2.INTER_LINEAR)
    return random_locate_defect_to_img(
        img, label, shear_defect, shear_defect_label)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import os
#     defect_img_dir = './缺陷小图/异物'
#     defect_label_dir = './缺陷小图/异物_label/'
#     img_dir = './正常图/2'
#     label_dir = './正常图/2_label'
#     location_result_dir = './locate_img'
#     for img_path in bz_path.get_file_path(img_dir, ret_full_path=True):
#         img_name = os.path.splitext(os.path.split(img_path)[1])[0]
#         img = cv2.imread(img_path)
#         label = cv2.imread(label_dir + '/' + img_name + '.png', 0)
#         flip_img = img.copy()
#         flip_label = label.copy()
#         for defect_img_path in bz_path.get_file_path(
#                 defect_img_dir, ret_full_path=True):
#             defect_img = cv2.imread(defect_img_path)
#             defect_img_name = \
#             os.path.splitext(os.path.split(defect_img_path)[1])[0]
#             defect_label = cv2.imread(
#                 defect_label_dir + '/' + defect_img_name + '.png', 0)
#             defect_label = defect_label
#             difference_img = get_defect_difference_from_background(
#                 defect_img, defect_label)
#             noise_img = add_noise_to_defect(defect_img)
#
#             # 在目标图像上平移缺陷
#             location_img, location_label = random_locate_defect_to_img(
#                 img, label, difference_img, defect_label)
#             location_img_bbox, location_label_bbox = random_locate_defect_to_img_bbox(
#                 img, label, difference_img, defect_label)
#             plt.figure()
#             plt.subplot(221)
#             plt.imshow(location_img)
#             plt.subplot(222)
#             plt.imshow(location_label)
#             plt.subplot(223)
#             plt.imshow(location_img_bbox)
#             plt.subplot(224)
#             plt.imshow(location_label_bbox)
#             plt.show()
