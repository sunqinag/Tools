# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：loss.py
#   摘   要：计算loss
#   当前版本:2019121715
#   作   者：崔宗会
#   完成日期：2019-12-17
# -----------------------------
import tensorflow as tf


def calculate_background_and_foreground_weighted_loss(
        logits, labels, background_and_foreground_loss_weight):
    """
    :param logits: 预测结果
    :param labels: 标签
    :param background_and_foreground_loss_weight: 背景前景权重
    :return:
    """
    if len(background_and_foreground_loss_weight) != 2:
        raise ValueError('前景背景权重的长度必须是2！')
    # 背景
    background_indices = tf.squeeze(
        tf.where(tf.equal(labels, 0)), 1)
    background_labels = tf.gather(labels, background_indices)
    background_logits = tf.gather(logits, background_indices)

    # 前景
    foreground_indices = tf.squeeze(
        tf.where(tf.greater(labels, 0)), 1)
    foreground_labels = tf.gather(labels, foreground_indices)
    foreground_logits = tf.gather(logits, foreground_indices)

    # 背景loss
    background_loss = tf.losses.sparse_softmax_cross_entropy(
        background_labels,
        background_logits,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    # 前景loss
    foreground_loss = tf.losses.sparse_softmax_cross_entropy(
        foreground_labels,
        foreground_logits,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    return background_and_foreground_loss_weight[0] * background_loss \
        + background_and_foreground_loss_weight[1] * foreground_loss


def calculate_class_weighted_loss(logits, labels, class_loss_weight):
    """
    :param logits: 预测结果
    :param labels: 标签
    :param class_loss_weight: 类别权重
    :return:
    """
    weights = __calculate_loss_weight(labels, class_loss_weight)
    return tf.losses.sparse_softmax_cross_entropy(
        labels,
        logits,
        weights=weights,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)


def __calculate_loss_weight(labels, weights):
    """
    :param labels: 标签
    :param weights: 每一个类别的权重
    :return:
    """
    label_shape = tf.shape(labels)
    weight = tf.zeros(label_shape)
    loss_weight = tf.zeros(label_shape)
    for i in range(len(weights)):
        classes_loss_weight = tf.where(
            tf.equal(labels, i),
            tf.multiply(tf.ones(label_shape), weights[i]),
            weight)
        loss_weight = tf.add_n([loss_weight, classes_loss_weight])
    return loss_weight
