#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ----------------------------
# !  Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：convert_to_pb
#   摘   要：将checkpoint或者save_model转为pb
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-07-25
# -----------------------------

import tensorflow as tf
from tensorflow.python.framework import graph_util


def convert_checkpoint_to_pb(input_checkpoint, output_pb_path, output_node_names):
    '''
    :param input_checkpoint:
    :param output_pb_path: PB模型保存路径
    :return:
    '''
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(
            # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_pb_path, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
    print('转换完成，输出节点：', output_node_names)


def convert_export_model_to_pb(export_model_path, output_pb_path,
                               output_node_names):
    '''
    :param input_checkpoint:
    :param output_pb_path: PB模型保存路径
    :return:
    '''
    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            export_model_path)

        output_graph_def = graph_util.convert_variables_to_constants(
            # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_pb_path, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
    print('转换完成，输出节点：', output_node_names)


def add_input_holder_to_checkpoint(input_checkpoint, network_estimator_obj,
                                   output_path, input_parameter):
    """
    :param input_checkpoint:原始的checkpoint
    :param network_estimator_obj:estimator oject
    :param output_path:输出路径
    :param input_parameter:输入的参数
    :return:
    """
    input = input_parameter
    result = network_estimator_obj.network(input, is_training=False)
    softmax = tf.nn.softmax(result, axis=-1, name='softmax')
    classess = tf.argmax(softmax, name='classes')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        saver.save(sess, output_path)


# def main():
#     checkpoint_path=r"D:\中核\convert_checkpoint_to_pb\model\model.ckpt"
#     out_path=r"D:\中核\pb_path\frozen_model.pb"
#     convert_checkpoint_to_pb(checkpoint_path,out_path,output_node_names="classes,softmax_1")
#
# main()
#
# def main():
#     checkpoint_path=r"D:\中核\export_model_dir\1563901406"
#     out_path=r"D:\中核\pb_path\frozen_model.pb"
#     convert_export_model_to_pb(checkpoint_path,out_path,output_node_names="softmax,"
#                                                             "classes"
#                                                             )
#
# main()
#
#
# def main():
#     from diabetic_package.model_training.estimator.segmentation import \
#         unet_estimator
#
#     checkpoint_path=r"D:\中核\best_checkpoint_dir\model.ckpt"
#     out_path=r"./model/model.ckpt"
#     network_estimator_obj = unet_estimator.UnetEstimator(
#         img_shape=[900, 738, 1], class_num=3)
#
#     input=tf.placeholder(tf.float16,shape=[None,900,738,1])
#
#     add_input_holder_to_checkpoint(checkpoint_path,network_estimator_obj,out_path,input_parameter=input)
#
# main()
