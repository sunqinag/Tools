# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：python_accuracy.py
#   摘   要：摘   要：计算精度、iou、召回率、准确率和auc python版)
#   当前版本:2019121715
#   作   者：王茜,崔宗会
#   完成日期：2019-12-17
# -----------------------------
import numpy as np
from sklearn import metrics


def calculate_class_accuracy(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return:平均精度和各个类别精度的字典
    """
    __check_parameters(labels, predictions, class_num)

    accuracy_dict = {}
    accuracy_sum = 0
    for i in range(class_num):
        # 每一个类别的精度
        class_labels = np.where(np.equal(labels, i),
                                np.ones_like(labels),
                                np.zeros_like(labels))
        class_predictions = np.where(np.equal(predictions, i),
                                     np.ones_like(predictions),
                                     np.zeros_like(predictions))
        class_accuracy = metrics.accuracy_score(class_labels, class_predictions)
        accuracy_dict[mode + '_class' + str(i) + '_accuracy'] = class_accuracy
        accuracy_sum += class_accuracy
    # 计算平均精度
    accuracy_dict[mode + '_mean_accuracy'] = np.divide(accuracy_sum, class_num)
    return accuracy_dict


def calculate_class_iou(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return: 平均iou和各个类别iou的字典
    """
    __check_parameters(labels, predictions, class_num)

    iou_dict = {}
    iou_sum = 0.0
    for i in range(class_num):
        class_labels = np.where(np.equal(labels, i),
                                np.ones_like(labels),
                                np.zeros_like(labels))
        class_predictions = np.where(np.equal(predictions, i),
                                     np.ones_like(predictions),
                                     np.zeros_like(predictions))

        label_indices = np.where(np.equal(labels, i))[0]
        prediction_indices = np.where(np.equal(predictions, i))[0]
        concat_indices = np.concatenate([label_indices, prediction_indices], axis=0)
        concat_indices = np.unique(concat_indices)
        select_labels = class_labels[concat_indices]
        select_predictions = class_predictions[concat_indices]
        intersection = np.sum(np.multiply(select_labels, select_predictions))
        union = len(concat_indices)
        union = np.maximum(union, 1e-6)
        iou = intersection / union

        iou_dict[mode + '_class' + str(i) + '_iou'] = iou
        iou_sum += iou
    # 计算平均iou
    iou_dict[mode + '_mean_iou'] = iou_sum / class_num
    return iou_dict


def calculate_class_recall(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return:平均召回率和各个类别召回率的字典
    """
    __check_parameters(labels, predictions, class_num)

    recall_dict = {}
    recall_sum = 0
    for i in range(class_num):
        # 每一个类别的召回率
        class_labels = np.where(np.equal(labels, i),
                                np.ones_like(labels),
                                np.zeros_like(labels))
        class_predictions = np.where(np.equal(predictions, i),
                                     np.ones_like(predictions),
                                     np.zeros_like(predictions))

        class_indices = np.where(np.equal(labels, i))

        select_labels = class_labels[class_indices]
        select_predicts = class_predictions[class_indices]

        class_recall = metrics.recall_score(select_labels, select_predicts)

        recall_dict[mode + '_class' + str(i) + '_recall'] = class_recall
        recall_sum += class_recall
    # 计算平均召回率
    recall_dict[mode + '_mean_recall'] = np.divide(recall_sum, class_num)
    return recall_dict


def calculate_class_precision(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return:平均准确度率和各个类别准确率的字典
    """
    __check_parameters(labels, predictions, class_num)

    precision_dict = {}
    precision_sum = 0
    for i in range(class_num):
        # 每一个类别的准确率
        class_labels = np.where(np.equal(labels, i),
                                np.ones_like(labels),
                                np.zeros_like(labels))
        class_predictions = np.where(np.equal(predictions, i),
                                     np.ones_like(predictions),
                                     np.zeros_like(predictions))
        class_indices = np.where(np.equal(predictions, i))

        select_labels = class_labels[class_indices]
        select_predicts = class_predictions[class_indices]

        class_precision = metrics.precision_score(select_labels, select_predicts)
        precision_dict[mode + '_class' + str(i) + '_precision'] = \
            class_precision
        precision_sum += class_precision
    # 计算平均准确率
    precision_dict[mode + '_mean_precision'] = np.divide(precision_sum,
                                                         class_num)
    return precision_dict


def calculate_class_auc(labels, predictions, class_num, mode):
    """
    :param labels: 标签
    :param predictions: 预测类别结果
    :param class_num: 类别数
    :param mode: 模式(训练或验证)
    :return:平均auc率和各个类别auc的字典
    """
    __check_parameters(labels, predictions, class_num)

    auc_dict = {}
    auc_sum = 0
    for i in range(class_num):
        # 每一个类别的auc
        class_labels = np.where(np.equal(labels, i),
                                np.ones_like(labels),
                                np.zeros_like(labels))
        class_predictions = np.where(np.equal(predictions, i),
                                     np.ones_like(predictions),
                                     np.zeros_like(predictions))
        class_auc = 0.0
        if 1 in np.unique(class_labels):
            class_auc = metrics.roc_auc_score(class_labels, class_predictions)
        auc_dict[mode + '_class' + str(i) + '_auc'] = class_auc
        auc_sum += class_auc
    # 计算平均auc
    auc_dict[mode + '_mean_auc'] = np.divide(auc_sum, class_num)
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


def calculate_class_weighted_accuracy(*weight_tuple,**result):
    '''
    :param result: 精度结果
    :param accuracy_weight: 精度权重
    :return: 精度的加权和
    '''
    if isinstance(result, dict) is False:
        raise ValueError('result必须是字典类型！')
    value=0
    for index in np.arange(len(weight_tuple)):
        value+=weight_tuple[index]*result['evaluate_class'+str(index)+'_accuracy']
    return value


def calculate_class_weighted_recall(*weight_tuple,**result):
    '''
    :param result: 精度结果
    :param accuracy_weight: 精度权重
    :return: 精度的加权和
    '''
    if isinstance(result, dict) is False:
        raise ValueError('result必须是字典类型！')
    value=0
    for index in np.arange(len(weight_tuple)):
        value+=weight_tuple[index]*result['evaluate_class'+str(index)+'_recall']
    return value

def calculate_class_weighted_precision(*weight_tuple,**result):
    '''
    :param result: 精度结果
    :param accuracy_weight: 精度权重
    :return: 精度的加权和
    '''
    if isinstance(result, dict) is False:
        raise ValueError('result必须是字典类型！')
    value=0
    for index in np.arange(len(weight_tuple)):
        value+=weight_tuple[index]*result['evaluate_class'+str(index)+'_precision']
    return value


def calculate_class_weighted_auc(*weight_tuple,**result):
    '''
    :param result: 精度结果
    :param accuracy_weight: 精度权重
    :return: 精度的加权和
    '''
    if isinstance(result, dict) is False:
        raise ValueError('result必须是字典类型！')
    value=0
    for index in np.arange(len(weight_tuple)):
        value+=weight_tuple[index]*result['evaluate_class'+str(index)+'_auc']
    return value

def calculate_class_weighted_iou(*weight_tuple,**result):
    '''
    :param result: 精度结果
    :param accuracy_weight: 精度权重
    :return: 精度的加权和
    '''
    if isinstance(result, dict) is False:
        raise ValueError('result必须是字典类型！')
    value=0
    for index in np.arange(len(weight_tuple)):
        value+=weight_tuple[index]*result['evaluate_class'+str(index)+'_iou']
    return value




def __check_parameters(labels, predictions, class_num):
    """
    参数检查
    :param labels: labels
    :param predictions: predictions
    :param class_num: class_num
    :return:
    """
    if (len(labels.shape)) != 1 and (len(predictions.shape)) != 1:
        raise ValueError('labels和predictions维度必须是1!')

    if labels.shape != predictions.shape:
        raise ValueError('labels和predictions维度必须相等!')

    if np.max(labels) >= class_num:
        raise ValueError('labels中存在大于等于class_num的类别！')

    if np.max(predictions) >= class_num:
        raise ValueError('predictions中存在大于等于class_num的类别！')

    if class_num <= 0:
        raise ValueError('class_num 必须为大于０的正整数！')


# if __name__ == '__main__':
#     labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
#     predictions = np.array([0, 1, 0, 1, 0, 3, 2, 3, 2, 3, 1, 3])
#     labels = np.reshape(labels, [-1])
#     predictions = np.reshape(predictions, [-1])
#     accuracy_dict = calculate_class_accuracy(labels, predictions, 4, 'train')
#     print(accuracy_dict)
#     recall_dict = calculate_class_recall(labels, predictions, 4, 'train')
#     print(recall_dict)
#     precision_dict = calculate_class_precision(labels, predictions, 4, 'train')
#     print(precision_dict)
#     auc_dict = calculate_class_auc(labels, predictions, 4, 'train')
#     print(auc_dict)
#     assessment_dict = get_assessment_result(labels, predictions, 4, 'train',
#                                             ['accuracy','recall','precision','auc','iou'])
#     print(assessment_dict)
#     iou_dict = calculate_class_iou(labels, predictions, 4, 'train')
#     print(iou_dict)
