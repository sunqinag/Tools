# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：python_split_train_eval_test_data.py
#   摘   要：数据前处理模块
#   当前版本:2019121715
#   作   者：刘恩甫,崔宗会
#   完成日期：2019-12-17
# -----------------------------

import os
import shutil
import numpy as np
import cv2

from diabetic_package.file_operator import bz_path
from diabetic_package.file_operator.balance_data import BalanceData
from diabetic_package.image_processing_operator.python_data_augmentation.python_data_augmentation import DataAugmentation
from diabetic_package.image_processing_operator.python_image_processing import imread,imwrite
from diabetic_package.log.log import bz_log

def copy_and_split_train_val_data(original_data_path, out_path, min_example_num=20, ext_list=(['jpg'], ['npy']), task='classification'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for sub_folder in bz_path.get_subfolder_path(original_data_path, ret_full_path=False, is_recursion=False):
        img_path = original_data_path + sub_folder + '/img/'
        label_path = original_data_path + sub_folder + '/label/'

        print(img_path)
        img_copy_path = out_path + '/original_data_copy/' + sub_folder + '/img/'
        label_copy_path = out_path + '/original_data_copy/' + sub_folder + '/label/'

        if not os.path.exists(img_copy_path):
            os.makedirs(out_path + '/original_data_copy/' + sub_folder + '/img/')
            os.makedirs(out_path + '/original_data_copy/' + sub_folder + '/label/')

        img_file_path_list = np.sort(np.array(bz_path.get_file_path(img_path, ret_full_path=True)))
        label_file_path_list = np.sort(np.array(bz_path.get_file_path(label_path, ret_full_path=True)))

        img_list_num = len(img_file_path_list)
        if img_list_num < min_example_num:
            if task == 'classification' or task == 'segmentation':
                data_aug = DataAugmentation(img_file_path_list, label_file_path_list,
                                            augmentation_ratio=np.ceil(min_example_num / img_list_num),
                                            generate_data_folder=out_path + '/generate_data/',
                                            task=task,out_file_extension_list=ext_list)
                data_aug.augment_data()

                for file_path in bz_path.get_file_path(out_path + '/generate_data/augmentation_img/', ret_full_path=True):
                    file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                    # img = cv2.imread(file_path)
                    # cv2.imwrite(out_path + '/original_data_copy/' + sub_folder + '/img/' + file_name + '.jpg', img)
                    img=imread(file_path,'RGB')
                    imwrite(out_path + '/original_data_copy/' + sub_folder + '/img/' + file_name + '.jpg', img)

                if task.lower() == 'classification':
                    for file_path in bz_path.get_file_path(out_path + '/generate_data/augmentation_label/', ret_full_path=False):
                        shutil.copy(out_path + '/generate_data/augmentation_label/' + file_path,
                                    out_path + '/original_data_copy/' + sub_folder + '/label/' + file_path)
                elif task.lower() == 'segmentation':
                    for file_path in bz_path.get_file_path(out_path + '/generate_data/augmentation_label/', ret_full_path=True):
                        file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                        # label = cv2.imread(file_path, 0)
                        # cv2.imwrite(out_path + '/original_data_copy/' + sub_folder + '/label/' + file_name + ".png", label)
                        label=imread(file_path,'gray')
                        imwrite(out_path + '/original_data_copy/' + sub_folder + '/label/' + file_name + ".png", label)
            elif task == 'detection':
                print('样本小于min_example_num，进行预增强...')
                bz_log.info('样本小于min_example_num，进行预增强...')
                #进行txt的格式转换
                txt2npy_path = out_path + '/txt2npy/' + sub_folder + '/label/'
                if os.path.exists(txt2npy_path):
                    shutil.rmtree(txt2npy_path)
                os.makedirs(txt2npy_path)

                for file_path in bz_path.get_file_path(label_path, ret_full_path=True):
                    file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                    if ext == 'txt':  # 进行格式转换
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        data = []
                        for line in lines:
                            temp = list(map(int, line.strip().split(',')))
                            data.append([temp[1], temp[0],temp[3],temp[2], temp[4]])
                        np.save(txt2npy_path + file_name + ".npy", data)
                if len(os.listdir(txt2npy_path))!=0:
                    label_file_path_list = np.sort(np.array(bz_path.get_file_path(txt2npy_path, ret_full_path=True)))

                yolo_min_example_augmentation_data = DataAugmentation(img_list=img_file_path_list,
                                                                                        label_list=label_file_path_list,
                                                                                        channel=3,
                                                                                        augmentation_ratio=np.ceil(
                                                                                            min_example_num / img_list_num),
                                                                                        # 增强倍数
                                                                                        generate_data_folder=out_path + '/generate_data/'+sub_folder+os.sep,
                                                                                        task='detection')
                yolo_min_example_augmentation_data.augment_data()

                for file_path in bz_path.get_file_path(out_path + '/generate_data/'+sub_folder+'/augmentation_img/', ret_full_path=True):
                    file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                    img = cv2.imread(file_path)
                    cv2.imwrite(out_path + '/original_data_copy/' + sub_folder + '/img/' + file_name + '.jpg', img)

                for file_path in bz_path.get_file_path(out_path + '/generate_data/'+sub_folder+'/augmentation_label/', ret_full_path=True):
                    file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                    shutil.copy(file_path,
                                out_path + '/original_data_copy/' + sub_folder + '/label/' + file_name + ".npy")
        else:

            for file_path in bz_path.get_file_path(img_path, ret_full_path=True):
                file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                # img = cv2.imread(file_path)
                # cv2.imwrite(out_path + '/original_data_copy/' + sub_folder + '/img/' + file_name + '.jpg', img)
                img=imread(file_path,'rgb')
                imwrite(out_path + '/original_data_copy/' + sub_folder + '/img/' + file_name + '.jpg', img)

            if task.lower() == 'classification':
                for file_path in label_file_path_list:
                    file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                    shutil.copy(file_path, out_path + '/original_data_copy/' + sub_folder + '/label/' + file_name + ".npy")
            elif task.lower() == 'segmentation':
                for file_path in bz_path.get_file_path(label_path, ret_full_path=True):
                    file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                    # label = cv2.imread(file_path, 0)
                    # cv2.imwrite(out_path + '/original_data_copy/' + sub_folder + '/label/' + file_name + ".png", label)
                    label=imread(file_path,'gray')
                    imwrite(out_path + '/original_data_copy/' + sub_folder + '/label/' + file_name + ".png", label)
            elif task.lower() == 'detection':
                for file_path in bz_path.get_file_path(label_path, ret_full_path=True):
                    file_name, ext = bz_path.get_file_name(file_path, return_ext=True)
                    if ext=='txt': #进行格式转换
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        data = []
                        for line in lines:
                            temp = list(map(int, line.strip().split(',')))
                            data.append([temp[1],temp[0],temp[3],temp[2],temp[4]])
                        np.save(out_path + '/original_data_copy/'+ sub_folder + '/label/' + file_name + ".npy",data)
                    elif ext=='npy':
                        shutil.copy(file_path,out_path + '/original_data_copy/'+ sub_folder + '/label/' + file_name + ".npy")

        img_list, label_list = bz_path.get_img_label_path_list(img_copy_path, label_copy_path, ret_full_path=True)
        if task.lower() != 'segmentation' and task.lower() != 'detection':
            label_list = np.array([np.loadtxt(label_path) for label_path in label_list])
        split_train_eval_test_data(img_list, label_list, out_path + '/train/' + sub_folder,
                                   out_path + '/val/' + sub_folder, out_path + '/test/' + sub_folder, task=task)

        # img_list, label_list = bz_path.get_img_label_path_list(img_copy_path, label_copy_path, ret_full_path=True)
        # if task.lower() !='segmentation':
        #     label_list = np.array([np.load(label_path) for label_path in label_list])
        #
        # split_train_eval_test_data(img_list, label_list, out_path + '/train/' + sub_folder, out_path + '/val/' + sub_folder, out_path + '/test/' + sub_folder, task=task)
        #

def get_preprocessing_img_and_label_path_list(
        base_folder,
        generate_data_folder='./generate_data/',
        max_augment_num=None,
        is_balance=True,
        mode='train', task='classification', out_file_extension_list=['jpg', 'txt']):
    """
    获取前处理后的img和label 列表，可以进行balance也可以不进行balance
    :param base_folder:数据路径,其下面包含如1,2,3,4等文件夹，1，2,3,4下包含img，label文件夹
    :param generate_data_folder:生成数据路径
    :param max_augment_num:均衡化后每个类别数目，如果为None则模型按类别最多的数目
    :param is_balance:是否进行均衡
    :param mode:模式，"只能输入train，val，test，TRAIN，VAL，TEST，Train，Val，Test"
                         "中的一个！"
    :return:img，label 的list
    """
    if not os.path.exists(base_folder):
        bz_log.error("路径" + base_folder + "不存在！")
        bz_log.error(base_folder)
        raise ValueError("路径" + base_folder + "不存在！")
    mode_lower = mode.lower()
    if mode_lower not in ['train', 'val', 'test']:
        bz_log.error("mode 只能输入train，val，test，TRAIN，VAL，TEST，"
            "Train，Val，Test中的一个！")
        bz_log.error(mode_lower)
        raise ValueError(
            "mode 只能输入train，val，test，TRAIN，VAL，TEST，"
            "Train，Val，Test中的一个！")

    if is_balance:
        balance_obj = BalanceData(base_folder=base_folder,
                                  generate_data_folder=generate_data_folder,
                                  max_augment_num=max_augment_num, task=task, out_file_extension_list=out_file_extension_list)
        img_list, label_list,is_repeat_data_flag = balance_obj.create_data()

        return img_list, label_list
    else:
        img_list, label_list = bz_path.get_all_subfolder_img_label_path_list(
            base_folder, ret_full_path=True)
        if mode == 'test' or mode == 'validate':
            return img_list, label_list

        shuffle_indices = np.arange(len(img_list))
        np.random.shuffle(shuffle_indices)
        shuffle_img_list = img_list[shuffle_indices]
        shuffle_label_list = img_list[shuffle_indices]
        return shuffle_img_list, shuffle_label_list


def split_train_eval_test_data(img_list, label_list=None, train_folder='split_train_eval_test' + os.sep + 'train',
                               eval_folder='split_train_eval_test' + os.sep + 'eval',
                               test_folder='split_train_eval_test' + os.sep + 'test',
                               ratio=(0.6, 0.1, 0.3), task='Segmentation'):
    '''
    对于image,label数据，实现数据集的按比例自动划分。
    :param img_list: 待划分图像的全路径列表,list或ndarray类型
    :param label_list:待划分标签的全路径列表,list或ndarray或None，如果为None,则只进行img的划分，不进行label的划分
    :param ratio:train,eval,test的分割比例，其加和值严格为1，且每个元素取值为[0,1]
    :param task:任务类型，包括分类、分割、目标检测，默认为分割
    :param train_folder: 划分后的train数据保存文件夹，会在其内部自动创建img、label文件夹，如果没有指定则使用默认路径
    :param eval_folder: 划分后的eval数据保存文件夹，会在其内部自动创建img、label文件夹，如果没有指定则使用默认路径
    :param test_folder: 划分后的test数据保存文件夹，会在其内部自动创建img、label文件夹，如果没有指定则使用默认路径
    输出目录结构为：

    train_folder
        |______img
        |______label


    eval_folder
        |______img
        |______label


    test_folder
        |______img
        |______label
    '''

    # 参数检查部分
    img_list, label_list, ratio, task = __check_param(img_list, label_list, ratio, task)

    # train,eval,test文件夹的创建,out_path是输出路径的集合,[img_path,label_path]*(train,eval,test)格式
    out_path = __mkdirs(train_folder, eval_folder, test_folder)

    # 计算shuffle的索引
    total_num = len(img_list)
    shuffle_index = np.arange(total_num)
    np.random.shuffle(shuffle_index)

    # 计算img_list的、label_list的分段
    train_eval_test_num = [0, int(ratio[0] * total_num), int((ratio[0] + ratio[1]) * total_num), total_num]
    split_img_list = [img_list[shuffle_index[train_eval_test_num[i]:train_eval_test_num[i + 1]]] \
                      for i in np.arange(len(train_eval_test_num) - 1)]

    split_label_list = []
    if label_list is not None:
        split_label_list = [label_list[shuffle_index[train_eval_test_num[i]:train_eval_test_num[i + 1]]] \
                            for i in np.arange(len(train_eval_test_num) - 1)]

    # 进行划分
    if label_list is None:  # 无标签只划分图片的情况
        for i in np.arange(len(split_img_list)):
            __img_copy(split_img_list[i], out_path[i][0])
    elif task in ('classification'):
        for i in np.arange(len(split_img_list)):
            __img_copy(split_img_list[i], out_path[i][0])
            __npy_save(split_img_list[i], split_label_list[i], out_path[i][1])
    elif task in ('detection'):
        for i in np.arange(len(split_img_list)):
            __img_copy(split_img_list[i], out_path[i][0])
            __npy_copy(split_img_list[i], out_path[i][1])
    else:  # segmentation任务的划分
        for i in np.arange(len(split_img_list)):
            __img_copy(split_img_list[i], out_path[i][0])
            __img_copy(split_label_list[i], out_path[i][1])


def __check_param(img_list, label_list, ratio, task):
    # img_list,label_list的参数检查
    if not isinstance(img_list, list) and not isinstance(img_list, np.ndarray):
        bz_log.error("img_list的类型为list或者ndarray类型!")
        bz_log.error("传入的img_list类型：", img_list.dtype)
        raise ValueError("img_list的类型为list或者ndarray类型!")

    if len(img_list) == 0:
        bz_log.error("img_list是空列表，请检查!")
        raise ValueError("img_list是空列表，请检查!")

    if not os.path.exists(img_list[0]):
        bz_log.error("img_list中的文件不存在，可能是非全路径，请检查！")
        bz_log.error("传入的错误img_list路径：",img_list[0])
        raise ValueError("img_list中的文件不存在，可能是非全路径，请检查！")

    if isinstance(img_list, list):
        img_list = np.array(img_list)

    # 对label_list的参数检查
    if label_list is not None:
        # list类型转换ndarray
        if isinstance(label_list, list):
            label_list = np.array(label_list)

        if not isinstance(label_list, list) and not isinstance(label_list, np.ndarray):
            bz_log.error("label_list的类型为list或者ndarray类型！")
            bz_log.error("label_list：", label_list.dtype)
            raise ValueError("label_list的类型为list或者ndarray类型！")

        # img_list,label_list的长度检查
        if len(img_list) != len(label_list):
            bz_log.error("img_list,label_list的长度需要保持一致！")
            bz_log.error("img_list和label_list长度分别为：", len(img_list) , len(label_list))
            raise ValueError("img_list,label_list的长度需要保持一致！")

    # ratio的参数检查
    ratio = np.array(ratio)
    if ratio.sum() != 1 or len(ratio) != 3 or np.any(ratio < 0) or np.any(ratio > 1):
        bz_log.error(" 'ratio'元素取值为[0,1],其加和需为1，请输入正确比例或取值！")
        bz_log.error("ratio值为：", ratio)
        raise ValueError(" 'ratio'元素取值为[0,1],其加和需为1，请输入正确比例或取值！")

    # 任务类型的参数检查
    task = task.lower()
    if task not in ('classification', 'segmentation', 'detection'):
        bz_log.error("task　必须为'classification','Classification','CLASSIFICATION','segmentation',"
                         "'SEGMENTATION','Segmentation','detection','detection'")
        bz_log.error("传入的task为：", task)
        raise ValueError("task　必须为'classification','Classification','CLASSIFICATION','segmentation',"
                         "'SEGMENTATION','Segmentation','detection','detection'")

    # segmentation特有的img_list,label_list的一致性检查(长度一样，但是名称不一致的情况)
    if task == 'segmentation':
        check_img_list = list(map(lambda x: bz_path.get_file_name(x), img_list))
        check_label_list = list(map(lambda x: bz_path.get_file_name(x), label_list))
        diff = list(set(check_img_list) ^ set(check_label_list))
        if len(diff) != 0:
            bz_log.error('img_list、label_list必须保持对应关系！')
            raise ValueError('img_list、label_list必须保持对应关系！')

    return img_list, label_list, ratio, task

def __mkdirs(train_folder, eval_folder, test_folder):
    # 创建3对[img,label]，到对应的输出路径
    type_key = ['img', 'label']
    out_path = []

    for folder in [train_folder, eval_folder, test_folder]:
        temp_list = []
        for tpk in type_key:
            if os.path.exists(folder + os.sep + tpk):
                shutil.rmtree(folder + os.sep + tpk)
            temp_list.append(folder + os.sep + tpk + os.sep)
            os.makedirs(folder + os.sep + tpk + os.sep)
        out_path.append(temp_list)
    return out_path

def __img_copy(src_path_list, dst_path):
    for img in src_path_list:
        shutil.copy(img, dst_path + '.'.join(bz_path.get_file_name(img, return_ext=True)))

def __npy_save(src_img_path_list, src_label_path_list, dst_path):
    for img, label in zip(src_img_path_list, src_label_path_list):
        np.save(dst_path + ''.join(bz_path.get_file_name(img, return_ext=False)) + '.npy', label)

def __npy_copy(src_img_path_list,dst_path):
    for img in src_img_path_list:
        label=img.replace('img','label',1)
        ext=bz_path.get_file_name(label,return_ext=True)[1]
        label=label.replace(ext,'npy')

        shutil.copy(label,dst_path)


def balance_train_val_data(data_path,out_path,max_augment_num,out_file_extension_list,task):
    '''by enfu'''

    #balance train data...
    train_balance_obj = BalanceData(base_folder=data_path+'/train/',
                              generate_data_folder=out_path+'/train_balance/',
                              max_augment_num=max_augment_num, task=task,
                              out_file_extension_list=out_file_extension_list)
    train_img_list, train_label_list= train_balance_obj.create_data()

    # balance val data...
    val_balance_obj = BalanceData(base_folder=data_path+'/val/',
                              generate_data_folder=out_path+'/val_balance/',
                              max_augment_num=None, task=task,
                              out_file_extension_list=out_file_extension_list)

    val_img_list, val_label_list = val_balance_obj.create_data()

    return (train_img_list, train_label_list, val_img_list, val_label_list)