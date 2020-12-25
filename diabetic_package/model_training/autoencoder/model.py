import tensorflow as tf
from tensorflow import layers
import numpy as np

# 正态分布初始化
normal_initiallizer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)


def Model_concat(inputs):
    inputs = tf.identity(inputs, name='img')
    inputs = tf.layers.conv2d(inputs,64,(3,3),padding='same',activation=tf.nn.relu,)

    inputs = tf.layers.max_pooling2d(inputs,(2,2),(2,2))
    layer1 = inputs
    print('layer1：',layer1.shape)

    inputs = tf.layers.conv2d(inputs, 32, (3,3), padding='same', activation=tf.nn.relu,)

    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2))
    layer2=inputs
    print('layer2：', layer2.shape)

    inputs = tf.layers.conv2d(inputs, 32, (3,3), padding='same', activation=tf.nn.relu,)

    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2))
    layer3 = inputs
    print('layer3：', layer3.shape)

    inputs = tf.layers.conv2d(inputs, 16, (3,3), padding='same', activation=tf.nn.relu,)

    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2))
    layer4 = inputs
    print('layer4：', layer4.shape)

    _,w,h,c = inputs.shape
    inputs = tf.layers.dense(inputs,w*h*c,activation=tf.nn.relu)
    # # inputs = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
    # inputs = tf.reshape(inputs,[-1,w*h*c])
    # inputs = tf.reshape(inputs, [-1, w , h , c])
    # inputs = tf.concat([inputs,layer4],axis=-1)
    inputs = tf.layers.conv2d_transpose(inputs=inputs, kernel_size=(3,3), strides=(2,2), filters=16,
                                        padding='same', activation=tf.nn.relu,)
    decode1 = inputs
    print('decode1：', decode1.shape)
    inputs = tf.concat([decode1, layer3], axis=-1)
    inputs = tf.layers.conv2d_transpose(inputs=inputs, kernel_size=(3,3), strides=(2,2), filters=32,
                                        padding='same', activation=tf.nn.relu,)

    decode2 = inputs
    inputs = tf.concat([decode2, layer2], axis=-1)
    print('decode2：', decode2.shape)

    inputs = tf.layers.conv2d_transpose(inputs=inputs, kernel_size=(3,3), strides=(2,2), filters=32,
                                        padding='same', activation=tf.nn.relu,)

    decode3 = inputs
    # inputs = tf.concat([decode3, layer1], axis=-1)
    print('decode3：', decode3.shape)
    inputs = tf.layers.conv2d_transpose(inputs=inputs, kernel_size=(3,3), strides=(2,2), filters=64,
                                        padding='same', activation=tf.nn.relu,)

    decode4 = inputs
    print('decode4：', decode4.shape)
    logits = tf.layers.conv2d(inputs,1,3,padding='same',name='output')
    # logits = tf.layers.conv2d(inputs,256,(3,3),padding='same')
    return logits

def Model_no_concat(inputs):
    inputs = tf.identity(inputs,name='img')
    inputs = tf.layers.conv2d(inputs,64,(3,3),padding='same',activation=tf.nn.relu,)

    inputs = tf.layers.max_pooling2d(inputs,(2,2),(2,2))
    layer1 = inputs
    print('layer1：',layer1.shape)

    inputs = tf.layers.conv2d(inputs, 32, (3,3), padding='same', activation=tf.nn.relu,)

    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2))
    layer2=inputs
    print('layer2：', layer2.shape)

    inputs = tf.layers.conv2d(inputs, 32, (3,3), padding='same', activation=tf.nn.relu,)

    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2))
    layer3 = inputs
    print('layer3：', layer3.shape)

    inputs = tf.layers.conv2d(inputs, 16, (3,3), padding='same', activation=tf.nn.relu,)

    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2))
    layer4 = inputs
    print('layer4：', layer4.shape)

    # _,w,h,c = inputs.shape
    # inputs = tf.layers.dense(inputs,w*h*c,activation=tf.nn.relu)

    inputs = tf.concat([inputs,layer4],axis=-1)
    inputs = tf.layers.conv2d_transpose(inputs=inputs, kernel_size=(3,3), strides=(2,2), filters=16,
                                        padding='same', activation=tf.nn.relu)
    decode1 = inputs
    print('decode1：', decode1.shape)
    # inputs = tf.concat([decode1, layer3], axis=-1)
    inputs = tf.layers.conv2d_transpose(inputs=inputs, kernel_size=(3,3), strides=(2,2), filters=32,
                                        padding='same', activation=tf.nn.relu)

    decode2 = inputs
    print('decode2：', decode2.shape)
    # inputs = tf.concat([decode2, layer2], axis=-1)
    inputs = tf.layers.conv2d_transpose(inputs=inputs, kernel_size=(3,3), strides=(2,2), filters=32,
                                        padding='same', activation=tf.nn.relu)

    decode3 = inputs
    print('decode3：', decode3.shape)
    # inputs = tf.concat([decode3, layer1], axis=-1)
    inputs = tf.layers.conv2d_transpose(inputs=inputs, kernel_size=(3,3), strides=(2,2), filters=64,
                                        padding='same', activation=tf.nn.relu)

    decode4 = inputs
    print('decode4：', decode4.shape)
    logits = tf.layers.conv2d(inputs,1,3,padding='same',name='output')
    # logits = tf.layers.conv2d(inputs,256,(3,3),padding='same')
    return logits

def Model_dense(inputs):
    inputs = tf.identity(inputs,name='img')
    # 第一个卷积层和池化层
    conv1 = tf.layers.conv2d(inputs,64,[3, 3],(1, 1),'same',activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1,[2, 2],strides=(2, 2),padding='same')
    print("第一个卷积层和池化层后参数：\n")
    print(conv1.shape)
    print(pool1.shape)

    # 第二个卷基层池化层
    conv2 = tf.layers.conv2d(pool1,32,[3, 3],(1, 1),'same',activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2,[2, 2],strides=(2, 2),padding='same')
    print("第二个卷积层和池化层参数：\n")
    print(conv2.shape)
    print(pool2.shape)

    # 第三个卷积池化层
    conv3 = tf.layers.conv2d(pool2,32,[5, 5],(1, 1),'same',activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3,[2, 2],strides=(2, 2),padding='same')
    print("第三个卷积层和池化层参数：\n")
    print(conv3.shape)
    print(pool3.shape)

    # 第四个卷积池化层
    conv4 = tf.layers.conv2d(pool3,16,[3, 3],(1, 1),'same',activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(conv4,[2, 2],(2, 2),padding='same')
    print("第四个卷积层和池化层参数：\n")
    print(conv4.shape)
    print(pool4.shape)

    # 全连接层
    fcn1 = tf.layers.dense(pool4,1024,activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    fcn2 = tf.layers.dense(fcn1,512,activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    fcn3 = tf.layers.dense(fcn2,1024,activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    # 第一个反卷积
    de_conv1 = tf.layers.conv2d_transpose(fcn3,32,[3, 3],2,'same',activation=tf.nn.relu)
    print("第1个反卷积层参数：\n")
    print(de_conv1.shape)

    # 第二个反卷积
    de_conv2 = tf.layers.conv2d_transpose(de_conv1,32,[3, 3],2,'same',activation=tf.nn.relu)
    print("第2个反卷积层参数：\n")
    print(de_conv2.shape)
    # 第三个反卷积
    de_conv3 = tf.layers.conv2d_transpose(de_conv2,64,[3, 3],2,'same',activation=tf.nn.relu)
    print("第3个反卷积层参数：\n")
    print(de_conv3.shape)

    de_conv4 = tf.layers.conv2d_transpose(de_conv3,1,[3, 3],2,'same',activation=tf.nn.relu, name="output")
    print("第4个反卷积层参数：\n")
    print(de_conv4.shape)

    # keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # output = tf.nn.dropout(de_conv4, keep_prob)
    # tf.add_to_collection('output', de_conv4)
    # return h_prob, keep_prob
    return de_conv4


def mvtcAE(inputs):
    inputs = tf.identity(inputs, name='img')

    # define model

    # Encode-----------------------------------------------------------
    x = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='same'
    )
    print("第一层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='same'
    )
    print("第2层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='same'
    )
    print("第3层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    print("第4层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='same'
    )
    print("第5层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    print("第6层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='same'
    )
    print("第7层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    print("第8层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    print("第9层参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=1,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    print("第10层参数")
    print(x.shape)
    # 全连接层
    x = tf.layers.dense(x, 1024, activation=tf.nn.relu,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(
                               0.003))

    x = tf.layers.dense(x, 512, activation=tf.nn.relu,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(
                               0.003))

    x = tf.layers.dense(x, 1024, activation=tf.nn.relu,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(
                               0.003))

    # Decode---------------------------------------------------------------------
    x = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    print("第1层解码参数")
    print(x.shape)
    x = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu
    )
    print("第2层解码参数")
    print(x.shape)
    x = layers.conv2d_transpose(
        inputs=x,
        filters=128,
        kernel_size=[3, 3],
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )
    print("第3层解码参数")
    print(x.shape)
    x = layers.conv2d_transpose(
        inputs=x,
        filters=64,
        kernel_size=[3, 3],
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )
    print("第4层解码参数")
    print(x.shape)
    x = layers.conv2d_transpose(
        inputs=x,
        filters=64,
        kernel_size=[3, 3],
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )
    print("第5层解码参数")
    print(x.shape)
    x = layers.conv2d_transpose(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        strides=2,
        padding='same',
        activation=tf.nn.relu
    )
    print("第6层解码参数")
    print(x.shape)
    decoded = layers.conv2d_transpose(
        inputs=x,
        filters=1,
        kernel_size=[3, 3],
        strides=2,
        padding='same',
        activation=tf.nn.relu,
        name="output"
    )
    print("第7层解码参数")
    print(decoded.shape)

    # keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # h_prob = tf.nn.dropout(decoded, keep_prob)
    # layer_out = (tf.reshape(de_conv5, [-1, 1556, 1924, 1]))
    # out_result = h_prob
    # output = tf.placeholder(tf.uint8,name='output')
    # output = tf.cast(out_result, tf.uint8)
    # tf.add_to_collection('output', output)
    return decoded





