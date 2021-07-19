# ---------------------------------
#   !Copyright(C) 2019,北京博众
#   All right reserved.
#   文件名称：Alexnet_estimator.py
#   摘   要：Alex网络实现
#   当前版本:2019090920
#   作   者：戴卓伦
#   完成日期：2019-07-12
# ---------------------------------

import tensorflow as tf
import json
import os
from .import base_classify_estimator
from collections import OrderedDict

tf.logging.set_verbosity(tf.logging.INFO)


class AlexnetEstimator(base_classify_estimator.BaseClassifyEstimator):
    def __init__(self,
                 model_dir='./',
                 img_shape=(227, 227, 3),
                 alexnet_layers_used_num=7,
                 split=True,
                 custom_conv_kernel_sizes=(),
                 custom_conv_strides=(),
                 custom_conv_filters=(),
                 custom_dense_filters=[],
                 custom_dropout_indices=(),
                 custom_activation_fns=(tf.nn.relu, tf.nn.relu),
                 class_num=6,
                 transfer_checkpoint_path=None,
                 regularizer_scale=(1e-7, 1e-6),
                 label_weight=1.0,
                 optimizer_fn=tf.train.AdamOptimizer,
                 learning_rate=1e-3,
                 tensors_to_log={'probabilities': 'softmax:0'},
                 **kwargs
                 ):
        """
            :param model_dir:  model_dir前缀
            :param alexnet_layers_used_num:  要保留的alexnet层数
            :param split:  是否对alexnet2，4，5层做split
            :param custom_conv_kernel_sizes:  list形式，每个元素代表每层卷积核大小
            :param custom_conv_strides:  list形式，每个元素代表每层卷积stride大小
            :param custom_conv_filters:  list形式，每个元素代表每层卷积输出层个数
            :param custom_dense_filters： list形式，每个元素代表每层全连接层输出层个数
            :param custom_dropout_indices:  list形式，dropout层索引
            :param custom_activation_fns:  激活函数，双元素tumple格式，分别表示卷积层和全连接层的激活函数。
            :param class_num： 分类类别数
            :param transfer_checkpoint_path  预训练模型checkpoint路径
            :param regularizer_scale:  l1,l2正则项系数,tumple格式
            :param label_weight: 不同类别样本加权时为list形式，不加权时设为一个数
            :param optimizer_fn： 优化函数。
            :param learning_rate： 学习率
            :param tensors_to_log： train时输出的log，字典形式
            :param **kwargs： 可变参数，可传递custom_network pooling层等信息
        """
        super(AlexnetEstimator, self).__init__(
            label_weight=label_weight,
            class_num=class_num,
            model_dir=model_dir,
            regularizer_scale=regularizer_scale,
            optimizer_fn=optimizer_fn,
            learning_rate=learning_rate,
            tensors_to_log=tensors_to_log)
        custom_dense_filters.append(class_num)
        self.transfer_checkpoint_path = transfer_checkpoint_path
        self.alexnet_layers_used_num = alexnet_layers_used_num
        self.custom_conv_kernel_sizes = custom_conv_kernel_sizes
        self.custom_conv_kernel_sizes = custom_conv_strides
        self.custom_conv_filters = custom_conv_filters
        self.custom_dense_filters = custom_dense_filters
        self.custom_dropout_indices = custom_dropout_indices
        self.custom_activation_fns = custom_activation_fns
        self.split = split
        self.class_num = class_num
        self.par_dict = kwargs
        self.img_shape = img_shape
        self.layer_info_dict = OrderedDict()
        self.__check_param()

    def network(self, img_batch, is_training):
        """

        :param img_batch: network 输入层
        :param is_training: 是否在输入模式下调用
        :return:
        """
        if img_batch.shape[-3:-1] != self.img_shape[:-1]:
            raise ValueError('输入network的图像尺寸有误')
        input_layer = tf.reshape(
            img_batch, [-1, self.img_shape[0], self.img_shape[1], self.img_shape[2]])
        input_layer=tf.cast(input_layer,tf.float32)
        alexnet_outputs = self.__alexnet(input_layer, is_training)
        logits = self.__custom_network(alexnet_outputs, is_training)
        self.__write_tensors()
        return logits

    def serving_input_receiver_fn(self):
        img = tf.placeholder(dtype=tf.float32,
                             shape=[None,
                                    self.img_shape[0],
                                    self.img_shape[1],
                                    self.img_shape[2]],
                             name='img')
        label = tf.placeholder(dtype=tf.float32,
                               shape=[None],
                               name='label')

        features = {'img': img, 'label': label}
        return tf.estimator.export.ServingInputReceiver(features, features)

    def __check_param(self):
        if len(self.img_shape) != 3:
            raise ValueError('输入参数img_shape的维度必须是3')
        if (self.img_shape[0] < 0
                or self.img_shape[1] < 0 or self.img_shape[2] < 0):
            raise ValueError('输入参数img_shape的三个维度必须大于0')
        if not (len(self.custom_conv_kernel_sizes) ==
                len(self.custom_conv_filters) ==
                len(self.custom_conv_kernel_sizes)):
            raise ValueError('输入的卷基层结构参数长度不匹配!')
        if len(self.regularizer_scale) != 2 or self.regularizer_scale[0] < 0 or self.regularizer_scale[1] < 0:
            raise ValueError('输入的正则项系数列表长度必须为2且两个值都要大于等于0')
        if not isinstance(self.label_weight, list) or len(self.label_weight) != self.class_num:
            raise ValueError('输入的label_weight参数应为长度等于分类数目的列表')
        if self.transfer_checkpoint_path is not None and self.split is None:
            raise ValueError('做迁移学习时要对Alexnet相应层进行拆分')
        if len(self.custom_conv_filters) != 0 and self.alexnet_layers_used_num > 5:
            raise ValueError('costum网络中有卷积层时alexnet层只能取卷积层，即alexnet_layers_used_num<=5')
        if self.alexnet_layers_used_num > 7 or self.alexnet_layers_used_num < 1:
            raise ValueError('alexnet_layers_used_num得取值范围为1~7')

    def __conv_layers(self, scope_name, input_layer, conv_kernel_sizes, conv_strides, conv_filters, activation_fn,
                      is_training, split_layers_indices=(), pool_indices=(), pool_size=(), pool_strides=()):
        """

        :param scope_name:  str 网络命名
        :param input_layer:  输入层
        :param conv_kernel_sizes:  list形式，每个元素代表每层卷积核大小
        :param conv_strides:  list形式，每个元素代表每层卷积stride大小
        :param conv_filters:  list形式，每个元素代表每层卷积输出层个数
        :param is_training:  是否在训练模式下调用
        :param split_layers_indices:  alexnet split索引
        :param pool_indices:  池化层suoyin
        :param pool_size:  list形式，每个元素代表pool_indices相应层pooling size大小
        :param pool_strides:  list形式，每个元素代表pool_indices相应层pooling stride大小
        :return:
        """
        last_layer = input_layer
        conv_layers_num = len(conv_kernel_sizes)
        for conv_layer_index in range(1, conv_layers_num+1):
            with tf.variable_scope(scope_name + '/conv' + str(conv_layer_index)):
                if conv_layer_index in split_layers_indices:
                    conv_groups = tf.split(axis=3, value=last_layer, num_or_size_splits=2)
                    with tf.variable_scope('up'):
                        conv_up = tf.layers.batch_normalization(tf.layers.conv2d(
                            inputs=conv_groups[0],
                            filters=int(conv_filters[conv_layer_index-1] / 2),
                            kernel_size=conv_kernel_sizes[conv_layer_index-1],
                            strides=conv_strides[conv_layer_index-1],
                            padding='SAME',
                            activation=activation_fn,
                            kernel_regularizer=self._regularizer()
                           ), training=is_training)

                    with tf.variable_scope('down'):
                        conv_down = tf.layers.batch_normalization(tf.layers.conv2d(
                            inputs=conv_groups[1],
                            filters=int(conv_filters[conv_layer_index-1] / 2),
                            kernel_size=conv_kernel_sizes[conv_layer_index-1],
                            strides=conv_strides[conv_layer_index-1],
                            padding='SAME',
                            activation=activation_fn,
                            kernel_regularizer=self._regularizer()),
                            training=is_training)
                    last_layer = tf.concat([conv_up, conv_down], axis=-1)
                else:
                    last_layer = tf.layers.batch_normalization(tf.layers.conv2d(
                        inputs=last_layer,
                        filters=conv_filters[conv_layer_index-1],
                        kernel_size=conv_kernel_sizes[conv_layer_index-1],
                        strides=conv_strides[conv_layer_index-1],
                        padding='SAME',
                        activation=activation_fn,
                        kernel_regularizer=self._regularizer()),  training=is_training)
                self.layer_info_dict[scope_name + '/conv' + str(conv_layer_index)] = last_layer.get_shape().as_list()
            if conv_layer_index in pool_indices:
                with tf.variable_scope(scope_name + '/pool' + str(conv_layer_index)):
                    pool_indices_index = pool_indices.index(conv_layer_index)
                    last_layer = tf.layers.max_pooling2d(
                        inputs=last_layer,
                        pool_size=pool_size[pool_indices_index],
                        strides=pool_strides[pool_indices_index],
                        padding='VALID')
                    self.layer_info_dict[
                        scope_name + '/conv' + str(conv_layer_index)] = last_layer.get_shape().as_list()
        return last_layer

    def __dense_layers(self, scope_name, input_layer, dense_filters, dropout_indices, activation_fn,
                       is_training):
        """

        :param scope_name:  str 网络命名
        :param input_layer:  输入层
        :param dense_filters:  list形式，每个元素代表每层全连接层输出个数
        :param dropout_indices:  dropout层索引
        :param is_training:  是否在训练模式下调用
        :return:
        """
        last_layer = tf.layers.flatten(input_layer)
        dense_layers_num = len(dense_filters)
        for dense_layer_index in range(1, dense_layers_num+1):
            with tf.variable_scope(scope_name + '/dense' + str(dense_layer_index)):
                if dense_layer_index == dense_layers_num and dense_filters[dense_layer_index-1] == self.class_num:
                    activation_fn = None
                last_layer = tf.layers.batch_normalization(tf.layers.dense(
                    inputs=last_layer,
                    units=dense_filters[dense_layer_index-1],
                    activation=activation_fn,
                    kernel_regularizer=self._regularizer()),
                    training=is_training
                )
                if dense_layer_index in dropout_indices:
                    last_layer = tf.layers.dropout(
                        inputs=last_layer,
                        rate=0.5,
                        training=is_training)
                self.layer_info_dict[scope_name + '/dense' + str(dense_layer_index)] = last_layer.get_shape().as_list()

        return last_layer

    def __alexnet(self, img_batch, is_training):
        """

        :param img_batch: alexnet输入
        :param is_training: 是否在输入模式下调用
        :return:
        """

        conv_kernel_sizes = [11, 5, 3, 3, 3]
        conv_strides = [4, 1, 1, 1, 1]
        conv_filters = [96, 256, 384, 384, 256]
        dense_filters = [4096, 4096, 47]
        pool_indices = [1, 2, 5]
        pool_size = [3, 3, 3]
        pool_strides = [2, 2, 2]
        alexnet_conv_layers_num = 5
        alexnet_activation_fn = tf.nn.relu
        dropout_indices = [1]
        if self.split:
            split_layers_indices = [2, 4, 5]
        else:
            split_layers_indices = []
        if self.alexnet_layers_used_num > alexnet_conv_layers_num:
            dense_filters = dense_filters[0: self.alexnet_layers_used_num - 5]
            conv_output = self.__conv_layers('alexnet', img_batch, conv_kernel_sizes, conv_strides, conv_filters,
                                             alexnet_activation_fn, is_training, split_layers_indices, pool_indices,
                                             pool_size, pool_strides)
            alexnet_output = self.__dense_layers('alexnet', conv_output, dense_filters, dropout_indices,
                                                 alexnet_activation_fn, is_training)
        else:
            conv_filters = conv_filters[0: self.alexnet_layers_used_num]
            conv_strides = conv_strides[0: self.alexnet_layers_used_num]
            conv_kernel_sizes = conv_kernel_sizes[0: self.alexnet_layers_used_num]
            alexnet_output = self.__conv_layers('alexnet', img_batch, conv_kernel_sizes, conv_strides, conv_filters,
                                                alexnet_activation_fn, is_training, split_layers_indices, pool_indices,
                                                pool_size, pool_strides)
        # if self.transfer_checkpoint_path and not self.latest_checkpoint():
        #     print('迁移学习初始化…')
        #     variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['global_step'])
        #     tf.train.init_from_checkpoint(self.transfer_checkpoint_path,
        #                                   {v.name.split(':')[0]: v for v in variables_to_restore})
        return alexnet_output

    def __custom_network(self, input_layer, is_training):
        """

        :param input_layer: custom network 输入层
        :param is_training: 是否在train模式下调用
        :return:
        """
        last_layer = input_layer
        if len(self.custom_conv_filters) > 0:
            last_layer = self.__conv_layers('custom_network', last_layer, self.custom_conv_kernel_sizes,
                                            self.custom_conv_kernel_sizes, self.custom_conv_filters,
                                            self.custom_activation_fns[0], is_training)
        last_layer = self.__dense_layers('custom_network', last_layer, self.custom_dense_filters,
                                         self.custom_dropout_indices, self.custom_activation_fns[1], is_training)
        return last_layer

    def __write_tensors(self):
        if self.latest_checkpoint():
            model_dir = os.path.dirname(self.model_dir)
            with open(model_dir + '/tensor.json', 'w') as f:
                json.dump(self.layer_info_dict, f, indent=1)




