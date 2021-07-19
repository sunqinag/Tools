import tensorflow as tf

from ..model import layers

class YoloNetwork():
    def __init__(self, feature_extractor):
        '''
        feature_extractor:
            用于提取特征的IFeatureExtractor实例
        '''
        self.feature_extractor = feature_extractor

    def __call__(self,
                 imgs,
                 grids,
                 prior_num_per_cell,
                 combined_feature_map_inds,
                 class_num,
                 activation,
                 training):
        '''
        imgs:
            输入网络的img batch
        grids:
            分块数目
        prior_num_per_cell:
            每个grid cell prior的数量
        combined_feature_map_inds:
            feature map extractor返回的所有尺度的feature map中用于预测的
            feature map的索引，combined feature map都会被resize到
            combined_feature_map_inds指定的最后一维feature map的大小
        class_num:
            objectness的分类个数
        activation:
            激活函数
        output_layer_type:
            输出层的处理方式，包括:
                global_average_pooling, global_average_pooling_conn, conv
        training:
            是否是在训练

        返回值:
            返回[batch_size, grid_y, grid_x, prior_num_per_cell, 5 + class_num]
        '''
        self.grids = grids
        self.prior_num_per_cell = prior_num_per_cell
        self.combined_feature_map_inds = combined_feature_map_inds
        self.class_num = class_num
        self.activation = activation
        self.training = training
        self.upsample_method="resize"

        #特征提取层
        _ = self.feature_extractor(imgs, training)
        #带FPN卷积的检测块
        #route_1:[?,52,52,256],route_2:[?,26,26,512],input_data:[?,13,13,1024]
        route_1, route_2, input_data=self.feature_extractor.feature_maps[self.combined_feature_map_inds[-1]], \
            self.feature_extractor.feature_maps[self.combined_feature_map_inds[-2]],\
            self.feature_extractor.feature_maps[self.combined_feature_map_inds[-3]]

        input_data = layers.convolutional(input_data, (1, 1, 1024, 512), self.training, 'conv52')
        input_data = layers.convolutional(input_data, (3, 3, 512, 1024), self.training, 'conv53')
        input_data = layers.convolutional(input_data, (1, 1, 1024, 512), self.training, 'conv54')
        input_data = layers.convolutional(input_data, (3, 3, 512, 1024), self.training, 'conv55')
        input_data = layers.convolutional(input_data, (1, 1, 1024, 512), self.training, 'conv56')

        conv_lobj_branch = layers.convolutional(input_data, (3, 3, 512, 1024), self.training, name='conv_lobj_branch')
        conv_lbbox = layers.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (5+self.class_num)),
                                          trainable=self.training, name='conv_lbbox', activate=False, bn=False)

        input_data = layers.convolutional(input_data, (1, 1, 512, 256), self.training, 'conv57')
        input_data = layers.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = layers.convolutional(input_data, (1, 1, 768, 256), self.training, 'conv58')
        input_data = layers.convolutional(input_data, (3, 3, 256, 512), self.training, 'conv59')
        input_data = layers.convolutional(input_data, (1, 1, 512, 256), self.training, 'conv60')
        input_data = layers.convolutional(input_data, (3, 3, 256, 512), self.training, 'conv61')
        input_data = layers.convolutional(input_data, (1, 1, 512, 256), self.training, 'conv62')

        conv_mobj_branch = layers.convolutional(input_data, (3, 3, 256, 512), self.training, name='conv_mobj_branch')
        conv_mbbox = layers.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (5+self.class_num)),
                                          trainable=self.training, name='conv_mbbox', activate=False, bn=False)

        input_data = layers.convolutional(input_data, (1, 1, 256, 128), self.training, 'conv63')
        input_data = layers.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = layers.convolutional(input_data, (1, 1, 384, 128), self.training, 'conv64')
        input_data = layers.convolutional(input_data, (3, 3, 128, 256), self.training, 'conv65')
        input_data = layers.convolutional(input_data, (1, 1, 256, 128), self.training, 'conv66')
        input_data = layers.convolutional(input_data, (3, 3, 128, 256), self.training, 'conv67')
        input_data = layers.convolutional(input_data, (1, 1, 256, 128), self.training, 'conv68')

        conv_sobj_branch = layers.convolutional(input_data, (3, 3, 128, 256), self.training, name='conv_sobj_branch')
        conv_sbbox = layers.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (5+self.class_num)),
                                          trainable=self.training, name='conv_sbbox', activate=False, bn=False)
        return conv_lbbox, conv_mbbox, conv_sbbox