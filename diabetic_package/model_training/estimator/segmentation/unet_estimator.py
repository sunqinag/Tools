# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：base_segmentation_estimator.py
#   摘   要：图像分割的子类，定义了network
#   当前版本:2019092917
#   作   者：王茜，崔宗会
#   完成日期：2018-09-26
# ---------------------------------

import tensorflow as tf
from collections import OrderedDict
import json

from . import base_segmentation_estimator
from diabetic_package.log.log import bz_log

class UnetEstimator(base_segmentation_estimator.BaseSegmentationEstimator):
    def __init__(self,
                 # 像素类别数
                 channel=3,
                 class_num=3,
                 regularizer_scale=(0.0000025, 0.000025),
                 optimizer_fn=tf.train.AdamOptimizer,
                 background_and_foreground_loss_weight=(0.45, 0.55),
                 class_loss_weight=(1, 1, 1),
                 max_img_outputs=6,
                 learning_rate=1e-3,
                 tensor_to_log={'probablities': 'softmax'},
                 assessment_list=['accuracy', 'iou', 'recall', 'precision', 'auc'],
                 # 卷积特征提取层
                 feature_conv_filter_nums=(),
                 feature_conv_kernel_sizes=(),
                 feature_conv_strides=(),
                 # unet卷积层
                 unet_conv_filter_nums=(16, 32, 64, 128, 256),
                 unet_conv_kernel_sizes=(3, 3, 3, 3, 3),
                 unet_conv_strides=(1, 1, 1, 1, 1),
                 conv_padding='same',
                 conv_num_before_pool=2,
                 conv_activate_fn=tf.nn.relu,
                 is_bn_before_activation=True,
                 pool_sizes=(2, 2, 2, 2, 2),
                 pool_strides=(2, 2, 2, 2, 2),
                 pooling_fn=tf.layers.max_pooling2d,
                 deconv_filter_nums=(128, 64, 32, 16, 8),
                 deconv_kernel_sizes=(3, 3, 3, 3, 3),
                 deconv_strides=(2, 2, 2, 2, 2),
                 deconv_padding='same',
                 deconv_activate_fn=tf.nn.sigmoid,
                 model_dir='./model_dir',
                 transfer_checkpoint_path='',
                 use_background_and_foreground_loss=True
                 ):
        """
        :param class_num: 分类个数
        :param feature_conv_filter_nums: 特征提取层卷积核个数
        :param feature_conv_kernel_sizes: 特征提取层卷积核大小
        :param feature_conv_strides: 特征提取层卷积步长
        :param unet_conv_filter_nums: unet卷积层卷积核个数
        :param unet_conv_kernel_sizes: unet卷积层卷积核大小
        :param unet_conv_strides: unet卷积层卷积步长
        :param conv_padding: 卷积层padding方式
        :param conv_num_before_pool: 池化前卷积层个数
        :param conv_activate_fn: 卷积层激活函数
        :param is_bn_before_activation: bn层是否在激活函数前面
        :param pool_sizes: 池化大小
        :param pool_strides: 池化步长
        :param pool_padding: 池化padding方式
        :param pooling_fn: 池化函数
        :param deconv_filter_nums: 反卷积核个数
        :param deconv_kernel_sizes: 反卷积核大小
        :param deconv_strides: 反卷积步长
        :param deconv_padding: 反卷积padding方式
        :param deconv_activate_fn: 反卷积激活函数
        :param l_scale: 正则大小
        :param model_dir: 模型路径
        :param assessment_list: 评价模型的list
        :param use_get_background_and_foreground_loss:是否使用前后背景的loss
        """

        super().__init__(class_num=class_num,
                         model_dir=model_dir,
                         regularizer_scale=regularizer_scale,
                         optimizer_fn=optimizer_fn,
                         background_and_foreground_loss_weight=background_and_foreground_loss_weight,
                         class_loss_weight=class_loss_weight,
                         max_img_outputs=max_img_outputs,
                         learning_rate=learning_rate,
                         tensor_to_log=tensor_to_log,
                         assessment_list=assessment_list)
        # 卷积结构参数
        # 特征提取层参数
        self.channel = channel
        self.feature_conv_filter_nums = feature_conv_filter_nums
        self.feature_conv_kernel_sizes = feature_conv_kernel_sizes
        self.feature_conv_strides = feature_conv_strides
        # unet卷积层参数
        self.unet_conv_filter_nums = unet_conv_filter_nums
        self.unet_conv_kernel_sizes = unet_conv_kernel_sizes
        self.unet_conv_strides = unet_conv_strides

        self.conv_padding = conv_padding
        self.conv_num_before_pool = conv_num_before_pool
        self.conv_activate_fn = conv_activate_fn
        self.is_bn_before_activation = is_bn_before_activation
        # 池化结构参数
        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.pooling_fn = pooling_fn
        # 反卷积结构参数
        self.deconv_filter_nums = deconv_filter_nums
        self.deconv_kernel_sizes = deconv_kernel_sizes
        self.deconv_strides = deconv_strides
        self.deconv_padding = deconv_padding
        self.deconv_activate_fn = deconv_activate_fn
        self.layer_info_dict = OrderedDict()
        self.transfer_checkpoint_path = transfer_checkpoint_path
        self.use_background_and_foreground_loss = use_background_and_foreground_loss
        self.__check_param()

    def network(self, inputs, is_training):
        """
        :param inputs: 输入数据
        :param is_training: 是否训练
        :return:
        """
        pool_tensor_list = []
        unet_conv_tensor_list = []
        # 开始建图
        heights = []
        widths = []
        heights.append(tf.shape(inputs)[1])
        widths.append(tf.shape(inputs)[2])
        result = tf.identity(inputs, name='input_img_layer')
        result = tf.cast(result, dtype=tf.float32)
        self.layer_info_dict[result.name] = result.get_shape().as_list()
        # 卷积层
        # 卷积特征提取层
        for i in range(len(self.feature_conv_filter_nums)):
            result = self.__conv_layer_with_batch_normalization(
                result,
                filter_num=self.feature_conv_filter_nums[i],
                conv_filter_size=self.feature_conv_kernel_sizes[i],
                conv_filter_stride=self.feature_conv_strides[i],
                conv_padding=self.conv_padding,
                activate_fn=self.conv_activate_fn,
                is_training=is_training,
                regularizer=self._regularizer)
            result = tf.identity(
                result, name='feature_conv_layer_' + str(i + 1))
            self.layer_info_dict[result.name] = result.get_shape().as_list()
        # unet卷积层
        for i in range(len(self.unet_conv_filter_nums)):
            # 2层卷积
            for j in range(self.conv_num_before_pool):
                result = self.__conv_layer_with_batch_normalization(
                    result,
                    filter_num=self.unet_conv_filter_nums[i],
                    conv_filter_size=self.unet_conv_kernel_sizes[i],
                    conv_filter_stride=self.unet_conv_strides[i],
                    conv_padding=self.conv_padding,
                    activate_fn=self.conv_activate_fn,
                    is_training=is_training,
                    regularizer=self._regularizer)
                unet_conv_tensor_list.append(result)
                result = tf.identity(
                    result,
                    name='unet_conv_layer_' + str(i + 1) + '_' + str(j + 1))
                self.layer_info_dict[result.name] = result.get_shape().as_list()
            # 池化层
            result = self.pooling_fn(result,
                                     pool_size=self.pool_sizes[i],
                                     strides=self.pool_strides[i],
                                     padding='same')
            pool_tensor_list.append(result)
            heights.append(tf.shape(result)[1])
            widths.append(tf.shape(result)[2])
            result = tf.identity(result, name='pool_layer_' + str(i + 1))
            self.layer_info_dict[result.name] = result.get_shape().as_list()
        # # 反卷积层
        concat_unet_layer = None
        for i in range(len(self.deconv_filter_nums)):
            result = self.__deconv_layer(
                result,
                deconv_filter_num=self.deconv_filter_nums[i],
                filter_size=self.deconv_kernel_sizes[i],
                stride=self.deconv_strides[i],
                deconv_padding=self.deconv_padding,
                activate_fn=self.deconv_activate_fn,
                regularizer=self._regularizer)
            result = tf.image.resize_bilinear(
                result, size=(heights[-i - 2], widths[-i - 2]))
            if i < (len(self.deconv_filter_nums) - 1):
                # 跨层连接
                concat_unet_layer_list = []
                for j in range(self.conv_num_before_pool):
                    concat_unet_layer_list.append(
                        unet_conv_tensor_list[
                            -self.conv_num_before_pool * i - j - 1])
                concat_unet_layer = tf.concat(concat_unet_layer_list, axis=3)
                concat_layer = tf.concat(
                    [concat_unet_layer,
                     pool_tensor_list[-i - 2],
                     result],
                    axis=3)
                result = tf.layers.conv2d(concat_layer,
                                          filters=self.deconv_filter_nums[i],
                                          kernel_size=1,
                                          strides=(1, 1),
                                          activation=self.conv_activate_fn)

            result = tf.identity(result, name='deconv_layer_' + str(i + 1))
            self.layer_info_dict[result.name] = result.get_shape().as_list()
        result = tf.identity(result, name='transfer_layer')
        self.layer_info_dict[result.name] = result.get_shape().as_list()
        result = tf.layers.conv2d(result,
                                  filters=self.class_num,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  activation=None)
        result = tf.identity(result, name='logits')
        self.layer_info_dict[result.name] = result.get_shape().as_list()
        self.__write_tensors_list(is_print=False)
        if self.transfer_checkpoint_path and not super().latest_checkpoint():
            exclude = ['global_step','conv2d_15']
            variables_to_restore = tf.contrib.framework. \
                get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(
                self.transfer_checkpoint_path,
                {v.name.split(':')[0]: v for v in
                 variables_to_restore})
        return result

    def get_loss(self, logits, labels):
        if self.use_background_and_foreground_loss:
            return self._get_background_and_foreground_loss(logits, labels)
        # return self._get_background_and_foreground_class_weight_loss(logits, labels)
        return self._get_class_weighted_loss(logits, labels)

    def _serving_input_receiver_fn(self):
        img = tf.placeholder(dtype=tf.float32,
                             shape=[None,
                                    None,
                                    None,
                                    self.channel],
                             name='img')

        label = tf.placeholder(dtype=tf.float32,
                               shape=[None,
                                      None,
                                      None,
                                      1],
                               name='label')
        features = {'img': img, 'label': label}
        return tf.estimator.export.ServingInputReceiver(features, features)

    def __check_param(self):
        if not (len(self.feature_conv_filter_nums) ==
                len(self.feature_conv_kernel_sizes) ==
                len(self.feature_conv_strides)):
            bz_log.error('输入的卷积特征提取层结构参数长度不匹配！')
            raise ValueError('输入的卷积特征提取层结构参数长度不匹配！')
        if not (len(self.unet_conv_kernel_sizes) ==
                len(self.unet_conv_filter_nums) ==
                len(self.unet_conv_strides) ==
                len(self.pool_strides) ==
                len(self.pool_sizes)):
            bz_log.error('输入的unet卷积层结构参数长度不匹配！')
            raise ValueError('输入的unet卷积层结构参数长度不匹配！')
        if not (len(self.deconv_kernel_sizes) ==
                len(self.deconv_filter_nums) ==
                len(self.deconv_strides)):
            bz_log.error('输入的反卷积结构参数长度不匹配！')
            raise ValueError('输入的反卷积结构参数长度不匹配！')

    def __conv_layer_with_batch_normalization(self,
                                              inputs,
                                              filter_num,
                                              conv_filter_size,
                                              conv_filter_stride,
                                              conv_padding,
                                              activate_fn=tf.nn.relu,
                                              is_training=True,
                                              regularizer=None):
        """
        :param inputs: 输入数据
        :param filter_num: 卷积核个数
        :param conv_filter_size: 卷积核大小
        :param conv_filter_stride: 卷积步长
        :param conv_padding: padding方式
        :param activate_fn: 激活函数
        :param is_training: 是否训练
        :param regularizer: 正则化方式
        :return:
        """

        h_conv = tf.layers.conv2d(
            inputs,
            filters=filter_num,
            kernel_size=conv_filter_size,
            strides=(conv_filter_stride, conv_filter_stride),
            padding=conv_padding,
            activation=None,
            use_bias=True,
            kernel_regularizer=regularizer)
        if self.is_bn_before_activation:
            return activate_fn(
                tf.layers.batch_normalization(h_conv, training=is_training))
        return tf.layers.batch_normalization(
            activate_fn(h_conv), training=is_training)

    def __deconv_layer(self,
                       inputs,
                       deconv_filter_num,
                       filter_size,
                       stride,
                       deconv_padding,
                       activate_fn,
                       regularizer):
        """
        :param inputs: 输入数据
        :param deconv_filter_num: 反卷积核个数
        :param filter_size: 反卷积核大小
        :param stride: 反卷积步长
        :param deconv_padding: padding方式
        :param activate_function: 激活函数
        :param regularizer: 正则化方式
        :return:
        """
        h_deconv = tf.layers.conv2d_transpose(
            inputs,
            filters=deconv_filter_num,
            kernel_size=(filter_size, filter_size),
            strides=stride,
            padding=deconv_padding,
            kernel_regularizer=regularizer)
        return activate_fn(h_deconv)

    def __write_tensors_list(self, is_print):
        if is_print:
            print(self.layer_info_dict)
        with open('tensor.json', 'w') as f:
            json.dump(self.layer_info_dict, f, indent=1)
