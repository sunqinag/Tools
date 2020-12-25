# -----------------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要: 根据训练好的网络，模型对图像/patches进行分类，并计算敏感性和特异性。
#           当输入的数据量过大时，直接统计敏感性、特异性可能导致内存过大程序中断，
#           可以通过sensibility_num（）计算每次输入的图像/patches的敏感性数目，
#           specificity_num（）计算每次输入图像/patches的特异性的数目，然后将每
#           次计算的数据进行叠加，最后分别除以图像/patches的总数就是整个数据敏感性
#           和特异性。
#   当前版本 : 0.0
#   作   者 ：于川汇 陈瑞侠
#   完成日期 : 2018-1-26
# -----------------------------------------------------------------------
import numpy as np
from ..model_prediction.classifier import IClassifier


def sensibility_num(classifier, input_data, labels):
    '''
        作用：
            计算图像/patches敏感性的数目
        参数:
            classifier： IClassifier子类的实例
            input_data： 输入的数据
            labels    ： 输入数据对应的标签

    '''
    if not isinstance(classifier, IClassifier):
        raise TypeError('classifier的类型应该是IClassifier')
    positive_correct_num = np.sum(
        classifier.classify(input_data) * labels
    )
    positive_num = np.sum(labels)
    return (positive_correct_num, positive_num)


def sensibility(classifier, input_data, labels):
    '''
        作用：
            计算图像/patches的敏感性
        参数:
            classifier： IClassifier子类的实例
            input_data： 输入的数据
            labels    ： 输入数据对应的标签

    '''
    (positive_correct_num, positive_num) = sensibility_num(
        classifier, input_data)
    return positive_correct_num / positive_num


def specificity_num(classifier, input_data, labels):
    '''
        作用：
            计算图像/patches特异性的数目
        参数:
            classifier： IClassifier子类的实例
            input_data： 输入的数据
            labels    ： 输入数据对应的标签

    '''
    if not isinstance(classifier, IClassifier):
        raise TypeError('classifier的类型应该是IClassifier')
    negative_correct_num = np.sum(
        (1 - classifier.classify(input_data)) * (1 - labels)
    )

    negative_num = np.sum(1 - labels)
    negative_p_num = np.sum(classifier.classify(input_data) * (1 - labels))
    return (negative_correct_num, negative_num, negative_p_num)


def specificity(classifier, input_data, labels):
    '''
        作用：
            计算图像/patches的特异性
        参数:
            classifier： IClassifier子类的实例
            input_data： 输入的数据
            labels    ： 输入数据对应的标签
    '''
    (negative_correct_num, negative_num) = specificity_num(
        classifier, input_data)
    return negative_correct_num / negative_num


def accuracy_num(classifier, input_data, labels):
    '''
           作用：
               计算图像/patches敏感性的数目
           参数:
               classifier： IClassifier子类的实例
               input_data： 输入的数据
               labels    ： 输入数据对应的标签

    '''
    if not isinstance(classifier, IClassifier):
        raise TypeError('classifier的类型应该是IClassifier')
    correct_num = np.sum(
        classifier.classify(input_data) == labels
    )
    return (correct_num, len(labels))


def accuracy(classifier, input_data, labels):
    '''
        作用：
            计算分类准确率
        参数:
            classifier： IClassifier子类的实例
            input_data： 输入的数据
            labels    ： 输入数据对应的标签

    '''
    (correct_num, total_num) = accuracy_num(
        classifier, input_data, labels)
    return correct_num / total_num
