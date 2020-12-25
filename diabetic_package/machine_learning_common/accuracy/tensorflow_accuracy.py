# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：tensorflow_accuracy.py
#   摘   要：计算精度、iou、召回率、准确率和auc(tensorflow版)
#   当前版本:2019091917
#   作   者：王茜,崔宗会
#   完成日期：2019-9-19
# -----------------------------
import tensorflow as tf


def calculate_class_accuracy(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return:平均精度和各个类别精度的字典
    """
    # if (len(labels.get_shape().as_list())) != 1 \
    #         and (len(predictions.get_shape().as_list())) != 1:
    #     raise ValueError('labels和predictions维度必须是1!')

    accuracy_dict = {}
    accuracy_sum = tf.constant(0, dtype=tf.float16)
    update_op_sum = tf.constant(0, dtype=tf.float16)

    for i in range(class_num):
        # 每一个类别的精度
        class_labels = tf.where(tf.equal(labels, i),
                                tf.ones_like(labels,dtype=tf.uint8),
                                tf.zeros_like(labels,dtype=tf.uint8))
        class_predictions = tf.where(tf.equal(predictions, i),
                                     tf.ones_like(predictions,dtype=tf.uint8),
                                     tf.zeros_like(predictions,dtype=tf.uint8))
        class_accuracy = tf.metrics.accuracy(class_labels, class_predictions)
        accuracy_dict[mode + '_class' + str(i) + '_accuracy'] = class_accuracy
        accuracy_sum += tf.cast(class_accuracy[0],tf.float16)
        update_op_sum += tf.cast(class_accuracy[1],tf.float16)
    # 计算平均精度
    accuracy_dict[mode + '_mean_accuracy'] = (
        tf.divide(accuracy_sum, class_num), tf.divide(update_op_sum, class_num))
    return accuracy_dict


def calculate_class_iou(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return: 平均iou和各个类别iou率的字典
    """
    if (len(labels.get_shape().as_list())) != 1 \
            and (len(predictions.get_shape().as_list())) != 1:
        raise ValueError('labels和predictions维度必须是1!')

    iou_dict = {}
    iou_sum = tf.constant(0, dtype=tf.float16)
    update_op_sum = tf.constant(0, dtype=tf.float16)
    for i in range(class_num):
        # 计算每个类别的iou
        class_labels = tf.where(tf.equal(labels, i),
                                tf.ones_like(labels,dtype=tf.uint8),
                                tf.zeros_like(labels,dtype=tf.uint8))
        class_predictions = tf.where(tf.equal(predictions, i),
                                     tf.ones_like(predictions,dtype=tf.uint8),
                                     tf.zeros_like(predictions,dtype=tf.uint8))

        label_indices = tf.where(tf.equal(labels, i))
        predictions_indices = tf.where(tf.equal(predictions, i))

        label_indices=tf.cast(label_indices,tf.int32)
        predictions_indices=tf.cast(predictions_indices,tf.int32)

        indices = tf.concat([label_indices, predictions_indices], axis=0)
        indices=tf.reshape(indices,[-1])
        indices = tf.unique(indices)[0]
        class_labels = tf.gather(class_labels, indices)
        class_predictions = tf.gather(class_predictions, indices)

        class_tp = tf.metrics.true_positives(class_labels, class_predictions)
        class_fn = tf.metrics.false_negatives(class_labels, class_predictions)
        class_fp = tf.metrics.false_positives(class_labels, class_predictions)

        class_iou0 = tf.cast(tf.divide(class_tp[0],
                               class_tp[0] + class_fp[0] + class_fn[0] + \
                               tf.constant(1e-5)),tf.float16)
        class_iou1 = tf.cast(tf.divide(class_tp[1],
                               class_tp[1] + class_fp[1] + class_fn[1] + \
                               tf.constant(1e-5)),tf.float16)
        class_iou = (class_iou0, class_iou1)

        iou_dict[mode + '_class' + str(i) + '_iou'] = class_iou
        iou_sum += class_iou[0]
        update_op_sum += class_iou[1]
    # 计算平均iou
    iou_dict[mode + '_mean_iou'] = (
        tf.divide(iou_sum, class_num), tf.divide(update_op_sum, class_num))
    return iou_dict


def calculate_class_recall(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return: 平均召回率和各个类别召回率的字典
    """

    # if (len(labels.get_shape().as_list())) != 1 \
    #         and (len(predictions.get_shape().as_list())) != 1:
    #     raise ValueError('labels和predictions维度必须是1!')


    recall_dict = {}
    recall_sum = tf.constant(0, dtype=tf.float16)
    update_op_sum = tf.constant(0, dtype=tf.float16)
    for i in range(class_num):
        # 计算每个类别的召回率
        class_labels = tf.where(tf.equal(labels, i),
                                tf.ones_like(labels,dtype=tf.uint8),
                                tf.zeros_like(labels,dtype=tf.uint8))
        class_predictions = tf.where(tf.equal(predictions, i),
                                     tf.ones_like(predictions,dtype=tf.uint8),
                                     tf.zeros_like(predictions,dtype=tf.uint8))

        class_indices = tf.cast(tf.where(tf.equal(labels, i)),tf.int32)
        class_labels = tf.gather(class_labels, class_indices)
        class_predictions = tf.gather(class_predictions, class_indices)
        class_recall = tf.metrics.recall(class_labels, class_predictions)
        recall_dict[mode + '_class' + str(i) + '_recall'] = class_recall
        recall_sum += tf.cast(class_recall[0],tf.float16)
        update_op_sum += tf.cast(class_recall[1],tf.float16)
    # 计算平均召回率
    recall_dict[mode + '_mean_recall'] = (
        tf.divide(recall_sum, class_num), tf.divide(update_op_sum, class_num))
    return recall_dict


def calculate_class_precision(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return: 平均召回率和各个类别召回率的字典
    """
    # if (len(labels.get_shape().as_list())) != 1 \
    #         and (len(predictions.get_shape().as_list())) != 1:
    #     raise ValueError('labels和predictions维度必须是1!')
    precision_dict = {}
    precision_sum = tf.constant(0, dtype=tf.float16)
    update_op_sum = tf.constant(0, dtype=tf.float16)
    for i in range(class_num):
        # 计算每个类别的准确率
        class_labels = tf.where(tf.equal(labels, i),
                                tf.ones_like(labels,dtype=tf.uint8),
                                tf.zeros_like(labels,dtype=tf.uint8))
        class_predictions = tf.where(tf.equal(predictions, i),
                                     tf.ones_like(predictions,dtype=tf.uint8),
                                     tf.zeros_like(predictions,dtype=tf.uint8))

        class_indices = tf.cast(tf.where(tf.equal(predictions, i)),tf.int32)
        class_labels = tf.gather(class_labels, class_indices)
        class_predictions = tf.gather(class_predictions, class_indices)

        class_precision = tf.metrics.precision(class_labels, class_predictions)
        precision_dict[mode + '_class' + str(i) + '_precision'] = \
            class_precision
        precision_sum += tf.cast(class_precision[0],tf.float16)
        update_op_sum += tf.cast(class_precision[1],tf.float16)
    # 计算平均准确率
    precision_dict[mode + '_mean_precision'] = (
        tf.divide(precision_sum, class_num),
        tf.divide(update_op_sum, class_num))
    return precision_dict


def calculate_class_auc(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return: 平均auc和各个类别auc的字典
    """
    if (len(labels.get_shape().as_list())) != 1 \
            and (len(predictions.get_shape().as_list())) != 1:
        raise ValueError('labels和predictions维度必须是1!')

    auc_dict = {}
    auc_sum = tf.constant(0, dtype=tf.float16)
    update_op_sum = tf.constant(0, dtype=tf.float16)
    for i in range(class_num):
        # 计算每个类别的auc
        class_labels = tf.where(tf.equal(labels, i),
                                tf.ones_like(labels,dtype=tf.uint8),
                                tf.zeros_like(labels,dtype=tf.uint8))
        class_predictions = tf.where(tf.equal(predictions, i),
                                     tf.ones_like(predictions,dtype=tf.uint8),
                                     tf.zeros_like(predictions,dtype=tf.uint8))
        class_auc = tf.metrics.auc(class_labels, class_predictions)
        auc_dict[mode + '_class' + str(i) + '_auc'] = class_auc
        auc_sum += tf.cast(class_auc[0],tf.float16)
        update_op_sum += tf.cast(class_auc[1],tf.float16)
    # 计算平均auc
    auc_dict[mode + '_mean_auc'] = (
        tf.divide(auc_sum, class_num),
        tf.divide(update_op_sum, class_num))
    return auc_dict


def get_assessment_result(labels,
                          predictions,
                          class_num,
                          mode,
                          assessment_list):
    '''
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :param assessment_list: 评估list
    :return: 返回评估list所对应的结果
    '''
    if not isinstance(assessment_list, list):
        raise ValueError('assessment_list的类型必须是list！')
    if not set(assessment_list).issubset(
            {'accuracy', 'iou', 'recall', 'precision', 'auc'}):
        raise ValueError('输入的assessment_list必须是accuracy, iou, recall, '
                         'precision, auc的子集！')
    assessment_dict = {}
    for assessment in assessment_list:
        if assessment == 'accuracy':
            single_assessment_dict = calculate_class_accuracy(
                labels, predictions, class_num, mode)
        elif assessment == 'iou':
            single_assessment_dict = calculate_class_iou(
                labels, predictions, class_num, mode)
        elif assessment == 'recall':
            single_assessment_dict = calculate_class_recall(
                labels, predictions, class_num, mode)
        elif assessment == 'precision':
            single_assessment_dict = calculate_class_precision(
                labels, predictions, class_num, mode)
        else:
            single_assessment_dict = calculate_class_auc(
                labels, predictions, class_num, mode)
        assessment_dict.update(single_assessment_dict)
    return assessment_dict

# if __name__ == '__main__':
#     import numpy as np
#
#     labels_list = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
#                             [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]])
#     predictions_list = np.array([[0, 1, 0, 1, 0, 3, 2, 3, 2, 3, 1, 3],
#                                  [0, 1, 1, 2, 3, 2, 1, 2, 0, 1, 2, 3]])
#
#     with tf.Session() as sess:
#
#         labels = tf.placeholder(shape=[None],dtype=tf.int8)
#         predictions = tf.placeholder(shape=[None],dtype=tf.int8)
#         iou_dict = calculate_class_iou(labels, predictions, 4, 'train')
#         accuracy_dict = calculate_class_accuracy(labels, predictions, 4,
#                                                  'train')
#         recall_dict = calculate_class_recall(labels, predictions, 4, 'train')
#         precision_dict = calculate_class_precision(labels, predictions, 4,
#                                                    'train')
#         auc_dict = calculate_class_auc(labels, predictions, 4, 'train')
#         assessment_dict = get_assessment_result(labels, predictions, 4, 'train',
#                                                 ['accuracy', 'iou'])
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.local_variables_initializer())
#         for i in range(2):
#             out = sess.run([labels, predictions,accuracy_dict,iou_dict,recall_dict,precision_dict,auc_dict],
#                            feed_dict={labels: labels_list[i],
#                                       predictions: predictions_list[i]})
#             print(out[0])
#             print(out[1])
#             print(out[2])
#             print(out[3])
#             print(out[4])
#             print(out[5])
#             print(out[6])

