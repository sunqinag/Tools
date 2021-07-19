import numpy as np
import tensorflow as tf

from . import feature_extractor
from . import layers


class DarkNet(feature_extractor.IFeatureExtractor):
    def __init__(self, **feature_extractor_params):
        '''
        first_conv_filters:
            darknet中第一个卷积层的filter个数
        dark_base_filter_list:
            每个dark base layer输出的channel个数
        residual_num_per_base_layer:
            每个dark base layer中残差结构的重复个数
        activation:
            激活函数
        '''
        super().__init__()
        self.activation = feature_extractor_params['activation']

    def __call__(self, imgs, training):
        '''
        imgs:
            输入网络的img batch
        training:
            是否是在训练

        返回值:
            最后一个feature map
        '''
        #在这里进行归一化
        input_data=tf.divide(imgs,tf.constant(255.),name='img_norm_op')

        with tf.variable_scope('darknet'):
            input_data = layers.convolutional(input_data, filters_shape=(3, 3, 3, 32), trainable=training,name='conv0')
            input_data = layers.convolutional(input_data, filters_shape=(3, 3, 32, 64),
                                              trainable=training, name='conv1', downsample=True)
            for i in range(1):
                input_data = layers.residual_block(input_data, 64, 32, 64, trainable=training,
                                                   name='residual%d' % (i + 0))

            input_data = layers.convolutional(input_data, filters_shape=(3, 3, 64, 128),
                                              trainable=training, name='conv4', downsample=True)

            for i in range(2):
                input_data = layers.residual_block(input_data, 128, 64, 128, trainable=training,
                                                   name='residual%d' % (i + 1))

            input_data = layers.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                              trainable=training, name='conv9', downsample=True)

            for i in range(8):
                input_data = layers.residual_block(input_data, 256, 128, 256, trainable=training,
                                                   name='residual%d' % (i + 3))

            route_1 = input_data
            input_data = layers.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                              trainable=training, name='conv26', downsample=True)

            for i in range(8):
                input_data = layers.residual_block(input_data, 512, 256, 512, trainable=training,
                                                   name='residual%d' % (i + 11))

            route_2 = input_data
            input_data = layers.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                              trainable=training, name='conv43', downsample=True)

            for i in range(4):
                input_data = layers.residual_block(input_data, 1024, 512, 1024, trainable=training,
                                                   name='residual%d' % (i + 19))

            self.feature_maps.append(route_1)
            self.feature_maps.append(route_2)
            self.feature_maps.append(input_data)
            return input_data
