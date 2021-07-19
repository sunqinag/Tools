# -----------------------------------------------------------------
#   !Copyright(C) 2020, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要：
#   当前版本: 0.0
#   作   者：陈瑞侠
#   完成日期: 2020
# -----------------------------------------------------------------
import tensorflow as tf
from  diabetic_package.model_training.estimator.LW_classification import classify_estimator
tf.logging.set_verbosity(tf.logging.INFO)


class CNNEstimator(classify_estimator.ClassifyEstimator):
    def __init__(self,
                 model_dir,
                 best_checkpoint_dir,
                 img_shape,
                 class_num):
        '''
            model_dir: 存储checkpoint的路径
            best_checkpoint_dir: 存储结果最好的checkpoint的路径
            img_shape: 输入图像的尺寸(height, width, channel)
            conv_kernel_sizes: 每个卷积层的大小，第一维维数就是
                                          卷基层的个数
            conv_shapes: 一个二维数组，第一维的元素代表每个卷基层的shape
            conv_strides: 一个二维数组，第一维的元素代表每层卷积的卷积步长
            conv_padding: 'same'或'valid'
            pool_sizes: 每个卷积层后pooling层kernel的大小
            pool_strides: 每个pool的步长
            pool_padding: 'same'或'valid'
            dense_dropout_rate: 每个全连接层后添加的dropout概率
            dense_units:每个全连接层输出的节点数
            dense_activation:每个全连接层后的激活函数
            l1_scale: l1正则项系数
            l2_scale: l2正则项系数
            label_smoothing:使用soft label时,统计的标签存在不准确的概率
        '''
        super(CNNEstimator, self).__init__(
            model_dir,
            best_checkpoint_dir,
            img_shape,
            class_num)
        self.img_height = img_shape[0]
        self.img_width = img_shape[1]
        self.channel_num = img_shape[2]
        self.img_shape = img_shape
        self.class_num = class_num
        self.__check_param()

    def __check_param(self):
        if len(self.img_shape) != 3:
            raise ValueError('输入参数img_shape的维度必须是3')
        if (self.img_shape[0] < 0
                or self.img_shape[1] < 0 or self.img_shape[2] < 0):
            raise ValueError('输入参数img_shape的三个维度必须大于0')


    def __conv_layers(self, input_layer, is_training):
        x = tf.cast(input_layer, dtype=tf.float32)
        return self.cnn_nin(x)

    def cnn_nin(self, x):
        x = tf.identity(x, name='img')
        x = tf.layers.conv2d(inputs=x,
                             filters=32,
                             kernel_size=[3, 3],
                             strides=[2,2],
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv1---", x.shape)
        x = tf.layers.conv2d(inputs=x,
                             filters=32,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv1-1---", x.shape)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=[2, 2],
            padding="SAME")
        print("pool1---", x.shape)
        x = tf.layers.conv2d(inputs=x,
                             filters=64,
                             kernel_size=[3, 3],
                             strides=[2, 2],
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv2---", x.shape)
        x = tf.layers.conv2d(inputs=x,
                             filters=64,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv2-1---", x.shape)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=[2, 2],
            padding="SAME")
        x1 = tf.layers.conv2d(inputs=x,
                             filters=128,
                              kernel_size=[3, 3],
                              strides=[2, 2],
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv3---", x1.shape)
        x2 = tf.layers.conv2d(inputs=x1,
                             filters=128,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv3-1---", x2.shape)
        # x3 = tf.concat([x1, x2],-1)
        pool = tf.layers.max_pooling2d(
            inputs=x2,
            pool_size=[2, 2],
            strides=[2, 2],
            padding="SAME")
        x4 = tf.layers.conv2d(inputs=pool,
                             filters=256,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv4---", x4.shape)
        x5 = tf.layers.conv2d(inputs=x4,
                             filters=256,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv4-1---", x5.shape)
        # x6 = tf.concat([x4, x5], -1)
        pool2 = tf.layers.max_pooling2d(
            inputs=x5,
            pool_size=[2, 2],
            strides=[2, 2],
            padding="SAME")
        x7 = tf.layers.conv2d(inputs=pool2,
                             filters=self.class_num,
                             kernel_size=[3, 3],
                             strides=[2, 2],
                             padding="SAME",
                             # activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        print("conv5---", x7.shape)
        x8 = tf.layers.dropout(x7, 0.5)
        logits = tf.reduce_mean(x8, [1, 2])
        print("第logits层参数")
        print(logits.shape)
        return logits

    def nin_cocat(self, x):
        x = tf.layers.conv2d(inputs=x,
                             filters=32,
                             kernel_size=[3, 3],
                             strides=[2,2],
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        x = tf.layers.conv2d(inputs=x,
                             filters=32,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=[2, 2],
            padding="SAME")
        x = tf.layers.conv2d(inputs=x,
                             filters=64,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        x = tf.layers.conv2d(inputs=x,
                             filters=64,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)

        x = tf.layers.max_pooling2d(
            inputs=x,
            pool_size=[2, 2],
            strides=[2, 2],
            padding="SAME")
        x1 = tf.layers.conv2d(inputs=x,
                             filters=128,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        x2 = tf.layers.conv2d(inputs=x1,
                             filters=128,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        x3 = tf.concat([x1, x2],-1)
        pool = tf.layers.max_pooling2d(
            inputs=x3,
            pool_size=[2, 2],
            strides=[2, 2],
            padding="SAME")
        x4 = tf.layers.conv2d(inputs=pool,
                             filters=256,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        x5 = tf.layers.conv2d(inputs=x4,
                             filters=256,
                             kernel_size=1,
                             padding="SAME",
                             activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        x6 = tf.concat([x4, x5], -1)
        pool2 = tf.layers.max_pooling2d(
            inputs=x6,
            pool_size=[2, 2],
            strides=[2, 2],
            padding="SAME")
        x7 = tf.layers.conv2d(inputs=pool2,
                             filters=self.class_num,
                             kernel_size=[3, 3],
                             strides=[2, 2],
                             padding="SAME",
                             # activation=tf.nn.leaky_relu,
                             kernel_regularizer=self.regularizer)
        x8 = tf.layers.dropout(x7, 0.5)
        logits = tf.reduce_mean(x8, [1, 2])
        print("第logits层参数")
        print(logits.shape)
        return logits



    def network(self, img_batch, is_training):
        input_layer = tf.reshape(
            img_batch, [-1, self.img_height, self.img_width, self.channel_num])
        logits = self.__conv_layers(input_layer, is_training)

        return logits
