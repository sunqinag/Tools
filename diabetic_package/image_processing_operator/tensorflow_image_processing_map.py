# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：tensorflow_image_processing_map.py
#   摘   要：tensorflow 中图像处理的map函数库
#   当前版本:2019091818
#   作   者：崔宗会
#   完成日期：2019-09-18
# -----------------------------
import tensorflow as tf
from functools import partial
from ..image_processing_operator import tensorflow_image_processing
import numpy as np


def function_false_default(image, label):
    """
    条件为false默认执行函数
    :param image: image
    :param label: label
    :return:原始的数据
    """
    # image=tf.cast(image,tf.uint8)
    if label is None:
        return image
    return image, label


def cond_function(random_index, image, label, function_true, function_false=function_false_default):
    """
    条件函数
    :param random_index:随机的索引
    :param image: image
    :param label: label
    :param function_true:如果条件为true则执行该函数,参数为image和label
    :param function_false: 如果条件为false则执行该函数,参数为image和label
    :return:
    """
    image_shape = tf.shape(image)
    label_shape = tf.shape(label)
    image_type = image.dtype
    label_type = label.dtype
    if label is None:
        image = tf.cond(
            tf.equal(random_index, 0), lambda: function_true(image, label=label), lambda: function_false(image, label))
        image = tf.image.resize_images(image, [image_shape[1], image_shape[2]], method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, image_type)
        return image, None
    else:
        image, label = tf.cond(
            tf.equal(random_index, 0), lambda: function_true(image, label=label), lambda: function_false(image, label))
        image = tf.image.resize_bilinear(image, [image_shape[1], image_shape[2]])
        label = tf.image.resize_nearest_neighbor(label, size=[label_shape[1], label_shape[2]])
        image = tf.cast(image, image_type)
        label = tf.cast(label, label_type)
        return image, label


def random_crop_or_pad_image_and_label_tf_map(image, label, crop_height, crop_width, rate=0.25):
    """
    使用tensorflow函数以rate概率进行图像进行随机crop
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label:,四维[b,h,w,c]的tensor或者None
    :param crop_height: crop的高
    :param crop_width: crop的宽
    :param rate:进行该操作的概率
    :return: 返回crop后的图像
    """
    _check_parameter_rate(rate)

    def function_false(image, label):
        image_type = image.dtype
        image = tf.image.resize_bilinear(image, size=[crop_height, crop_width])
        image = tf.cast(image, image_type)
        if label is None:
            return image
        label_type = label.dtype
        label = tf.image.resize_nearest_neighbor(label, size=[crop_height, crop_width])
        label = tf.cast(label, label_type)
        return image, label
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_crop_or_pad_image_and_label_tf,
        crop_width=crop_width, crop_height=crop_height)

    result = cond_function(random_index, image, label, function_true, function_false=function_false)

    if label is None:
        return result, None
    return result


def random_flip_image_and_label_tf_map(image, label):
    """
    以２５%概率进行翻转图像和标签,包括上下翻转,左右翻转,对角线翻转,和不翻转
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param label:输入标签,四维[b,h,w,c]的tensor或者None
    :param rate:进行该操作的概率
    :return:返回翻转后的图像
    """

    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    random_index = tf.floor(tf.random.uniform([], 0.0, 5.0))
    if label is None:
        image = tf.cond(tf.equal(random_index, 1),
                        lambda: tensorflow_image_processing.flip_up_down_image_and_label_tf(image, label=None),
                        lambda: tf.cond(tf.equal(random_index, 2),
                                        lambda: tensorflow_image_processing.flip_left_right_image_and_label_tf(image, label=None),
                                        lambda: tf.cond(tf.equal(random_index, 3),
                                                        lambda: tensorflow_image_processing.transpose_image_and_label_tf(image, label=None),
                                                        lambda: (image))))
        return image, None

    image, label = tf.cond(tf.equal(random_index, 1),
                           lambda: tensorflow_image_processing.flip_up_down_image_and_label_tf(image, label),
                           lambda: tf.cond(tf.equal(random_index, 2),
                                           lambda: tensorflow_image_processing.flip_left_right_image_and_label_tf(image, label),
                                           lambda: tf.cond(tf.equal(random_index, 3),
                                                           lambda: tensorflow_image_processing.transpose_image_and_label_tf(image, label),
                                                           lambda: (image, label))))

    return image, label


def flip_up_down_image_and_label_tf_map(image, label, rate=0.25):
    """
    以rate概率进行上下翻转,
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param label:输入标签,四维[b,h,w,c]的tensor或者None
    :param rate:进行该操作的概率
    :return:返回翻转后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = tensorflow_image_processing.flip_up_down_image_and_label_tf
    return cond_function(random_index, image, label, function_true)


def flip_left_right_image_and_label_tf_map(image, label, rate=0.25):
    """
    以rate概率进行左右翻转
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param label:输入标签,四维[b,h,w,c]的tensor或者None
    :param rate:进行该操作的概率
    :return:返回翻转后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = tensorflow_image_processing.flip_left_right_image_and_label_tf
    return cond_function(random_index, image, label, function_true)


def transpose_image_and_label_tf_map(image, label, rate=0.25):
    """
    以rate概率进行转置翻转
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param label:输入标签,四维[b,h,w,c]的tensor或者None
    :param rate:进行该操作的概率
    :return:返回翻转后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = tensorflow_image_processing.transpose_image_and_label_tf
    return cond_function(random_index, image, label, function_true)


def random_rescale_image_and_label_tf_map(image, label, min_scale=0.5, max_scale=1, rate=0.25):
    """
    使用tensorflow函数以rate概率进行图像进行缩放
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label:,四维[b,h,w,c]的tensor或者None
    :param min_scale: 缩放最小系数
    :param max_scale: 缩放最大系数
    :param rate:进行该操作的概率
    :return: 返回缩放后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.random_rescale_image_and_label_tf, min_scale=min_scale,
                            max_scale=max_scale)
    return cond_function(random_index, image, label, function_true)


def rotate_image_and_label_tf_map(image, label, angle, rate=0.25):
    """
    使用tensorflow函数以rate概率进行图像旋转
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param angle: 旋转的角度
    :param rate:进行该操作的概率
    :return: 旋转后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.rotate_image_and_label_tf, angle=angle)
    return cond_function(random_index, image, label, function_true)


def random_rotate_image_and_label_tf_map(image, label, max_angle, rate=0.25):
    """
    使用tensorflow函数以rate概率进行图像随机旋转
   :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param angle: 旋转的最大角度
    :param rate:进行该操作的概率
    :return: 旋转后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.random_rotate_image_and_label_tf, max_angle=max_angle)
    return cond_function(random_index, image, label, function_true)


def translate_image_and_label_tf_map(image, label, dx, dy, rate=0.25):
    """
    使用tensorflow函数以rate概率进行图像平移
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param dx:水平方向平移,向右为正
    :param dy:垂直方向平移,向下为正
    :param rate:进行该操作的概率
    :return:返回平移后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.translate_image_and_label_tf, dx=dx, dy=dy)
    return cond_function(random_index, image, label, function_true)


def random_translate_image_and_label_tf_map(image, label, max_dx, max_dy, rate=0.25):
    """

    使用tensorflow函数以以rate概率进行图像随机平移
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param max_dx:水平方向平移最大距离,向右为正
    :param max_dy:垂直方向平移最大距离,向下为正
    :param rate:进行该操作的概率
    :return:返回平移后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.random_translate_image_and_label_tf, max_dx=max_dx, max_dy=max_dy)
    return cond_function(random_index, image, label, function_true)


def random_rescale_and_crop_image_and_label_tf_map(image, label, min_scale=0.5, max_scale=2.0,
                                                   crop_height=256, crop_width=256, rate=0.25):
    """
    使用tensorflow函数以以rate概率进行图像进行随机缩放和crop
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label:,四维[b,h,w,c]的tensor或者None
    :param min_scale: 缩放最小系数
    :param max_scale: 缩放最大系数
    :param crop_height:crop高度
    :param crop_width: crop宽
    :param rate:进行该操作的概率
    :return: 返回该操作后的图像
    """
    _check_parameter_rate(rate)

    def function_false(image, label):
        image_type = image.dtype
        image = tf.image.resize_bilinear(image, size=[crop_height, crop_width])
        image = tf.cast(image, image_type)
        if label is None:
            return image
        label_type = label.dtype
        label = tf.image.resize_nearest_neighbor(label, size=[crop_height, crop_width])
        tf.cast(label, label_type)
        return image, label
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.random_rescale_and_crop_image_and_label_tf,
                            min_scale=min_scale, max_scale=max_scale, crop_height=crop_height, crop_width=crop_width)
    return cond_function(random_index, image, label, function_true, function_false)


def adjust_brightness_image_tf_map(image, delta, rate=0.25):
    """
    使用tensorflow函数以rate概率调整图像亮度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param delta:增加的值
    :param rate:进行该操作的概率
    :return:调整后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.adjust_brightness_image_tf, delta=delta)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def adjust_random_brightness_image_tf_map(image, max_delta, rate=0.25):
    """
    使用tensorflow函数以rate概率随机调整图像亮度
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param max_delta: 调整的最大值
    :param seed: 种子点
    :param rate:进行该操作的概率
    :return: 调整后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.random_brightness_image_tf, max_delta=max_delta)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def adjust_contrast_image_tf_map(image, contrast_factor, rate=0.25):
    """
    使用tensorflow函数以rate概率调整图像对比度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param contrast_factor:调整的倍率
    :param rate:进行该操作的概率
    :return:调整后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.adjust_contrast_image_tf, contrast_factor=contrast_factor)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def adjust_random_contrast_image_tf_map(image, lower, upper, seed=None, rate=0.25):
    """
      使用tensorflow函数以rate概率随机调整图像对比度
      :param image: 输入图像,四维[b,h,w,c]的tensor
      :param lower: 调整的最小值
      :param upper: 调整的最大值
      :param seed: 种子点
      :param rate:进行该操作的概率
      :return: 调整后的图像
      """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.random_contrast_image_tf, lower=lower, upper=upper, seed=seed)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def adjust_hue_image_tf_map(image, delta, rate=0.25):
    """
    使用tensorflow函数以rate概率调整图像色相
    :param image:输入图像,四维[b,h,w,c]的tensor,c必须为3
    :param delta:调整颜色通道的增加量
    :param rate:进行该操作的概率
    :return:调整后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.adjust_hue_image_tf, delta=delta)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def adjust_random_hue_image_tf_map(image, max_delta, rate=0.25):
    """
    使用tensorflow函数以rate概率随机调整图像色相
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param max_delta: 调整颜色通道的最大随机值
    :param seed: 种子点
    :param rate:进行该操作的概率
    :return: 调整后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.random_hue_image_tf, max_delta=max_delta)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def adjust_saturation_image_tf_map(image, saturation_factor, rate=0.25):
    """
    使用tensorflow函数以rate概率调整图像饱和度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param saturation_factor:调整的倍率
    :param rate:进行该操作的概率
    :return:调整后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.adjust_saturation_image_tf, saturation_factor=saturation_factor)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def adjust_random_saturation_image_tf_map(image, lower, upper, seed=None, rate=0.25):
    """
    使用tensorflow函数以rate概率随机调整图像饱和度
    :param lower: 调整的最小值
    :param upper: 调整的最大值
    :param seed: 种子点
    :param rate:进行该操作的概率
    :return: 调整后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_saturation_image_tf, lower=lower, upper=upper, seed=seed)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def per_image_standardization_tf_map(image, rate=0.25):
    """
    以rate概率进行图像标准化
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param rate:进行该操作的概率
    :return:标准化后的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = tensorflow_image_processing.per_image_standardization_tf

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def add_gaussian_noise_tf_map(image, mean, std, rate=0.25):
    """
    以rate概率增加高斯噪声
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param mean: 噪声均值
    :param std: 噪声标准差
    :param rate:进行该操作的概率
    :return: 添加噪声的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.add_gaussian_noise, mean=mean, std=std)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def add_random_noise_tf_map(image, minval, maxval, rate=0.25):
    """
    以rate概率增加随机噪声
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param minval:噪声的最小值
    :param maxval:噪声的最大值
    :param rate:进行该操作的概率
    :return:添加噪声的图像
    """
    _check_parameter_rate(rate)
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.add_random_noise, minval=minval, maxval=maxval)

    return tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)


def add_salt_and_pepper_noise_pyfunc_tf_map(image, scale, value, rate=0.25):
    """
    以rate概率图像添加椒盐噪声
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param scale: 噪声的比例系数
    :param value: 噪声的值
    :param rate:进行该操作的概率
    :return: 添加噪声的图像
    """
    _check_parameter_rate(rate)
    image_shape = tf.shape(image)
    image_type = image.dtype
    random_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.add_salt_and_pepper_noise_pyfunc, scale=scale, value=value)
    image = tf.cast(image, image_type)
    image = tf.cond(
        tf.equal(random_index, 1), lambda: function_true(image), lambda: image)
    image = tf.reshape(image, shape=[image_shape[0], image_shape[1], image_shape[2], image_shape[3]])

    return image


def _check_parameter_rate(rate):
    if not isinstance(rate, float) or rate <= 0 or rate > 1:
        raise ValueError('rate 必须是大于零小于１的float类型正数！')
