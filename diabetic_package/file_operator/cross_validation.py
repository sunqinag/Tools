from sklearn.model_selection import KFold
import numpy as np


def create_cross_validation_data(k_fold, img_list, label_list):
    '''
    切分出交叉验证使用的数据集
    :param k_fold: 交叉验证折数
    :param img_list: 所有图片路径
    :param label_list: feature对应的label
    :return: 一个list,分别为划分好的训练集和验证集图片路径和对应label
    测试用例：# img_list = np.array([1.jpg,2.jpg,3.jpg,4.jpg,5.jpg,6.jpg,7.jpg,8.jpg,9.jpg,10.jpg])
            # label_list=np.array([1,2,0,0,1,1,2,0,2,1])
    '''
    if len(img_list) != len(label_list):
        raise ValueError('输入的img_list和label_list长度不匹配！')

    data_list = []
    KF = KFold(n_splits=k_fold, shuffle=True)
    img_list = np.array(img_list)
    label_list = np.array(label_list)

    for train_index, test_index in KF.split(img_list):
        train_img_list, val_img_list = img_list[train_index], img_list[
            test_index]
        train_label_list, val_label_list = label_list[train_index], label_list[test_index]
        data_list.append(
            [train_img_list, train_label_list, val_img_list, val_label_list])

    return data_list
