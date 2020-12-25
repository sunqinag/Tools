# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：segmenter.py
#   摘   要：实现图像分割数据集的加载，模型的训练、验证及模型的保存
#   当前版本:2019121816
#   作   者：王茜，崔宗会,陈瑞侠
#   完成日期：2018-12-18
# ---------------------------------

import numpy as np
import os
import shutil
import json
import socket
import collections
from ....file_operator import bz_path, cross_validation
from ....dataset_common import dataset
from ....machine_learning_common import convert_to_pb
from ....machine_learning_common.accuracy import python_accuracy
import time
from diabetic_package.log.log import bz_log
class Segmenter():
    def __init__(self,
                 model_dir,
                 estimator_obj,
                 height_and_width,
                 class_num,
                 crop_height_and_width=(400, 400),
                 channels_list=(1, 1),
                 file_extension_list=('bmp', 'png'),
                 accuracy_weight=(1.0, 1.0),
                 k_fold=1,
                 eval_epoch_step=1,
                 batch_size=1,
                 epoch_num=1,
                 change_loss_fn_threshold=0.3,
                 calculate_saved_model_value_callback=python_accuracy.calculate_class_weighted_recall,
                 is_socket=False,
                 is_early_stop=True):
        """
        :param model_dir: 模型路径
        :param estimator_obj: estimator对象
        :param height_and_width: 图像高度和宽度
        :param crop_height_and_width: 图像crop高度和宽度
        :param channels_list: 图像通道list
        :param file_extension_list: 图像类型list
        :param class_num: 分类个数
        :param accuracy_weight: 精度权重
        :param k_fold: 交差验证折数
        :param batch_size: batch大小
        :param epoch_num: epoch数量
        :param change_loss_fn_threshold:切换loss函数的阈值
        """
        self.img_height = height_and_width[0]
        self.img_width = height_and_width[1]
        self.img_crop_height = crop_height_and_width[0]
        self.img_crop_width = crop_height_and_width[1]
        self.channel_list = channels_list
        self.file_extension_list = file_extension_list
        self.model_dir = model_dir
        self.accuracy_weight = accuracy_weight
        self.estimator_obj = estimator_obj
        self.k_fold = k_fold
        self.eval_epoch_step = eval_epoch_step
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.class_num = class_num
        self.change_loss_fn_threshold = change_loss_fn_threshold
        self.calculate_saved_model_value_callback = calculate_saved_model_value_callback
        self.is_socket = is_socket
        self.is_early_stop = is_early_stop
        if self.channel_list[0] != self.estimator_obj.channel:
            raise ValueError('channel_list的第一个通道数与Unet_Estimator的通道数不相符')

        self.best_checkpoint_dir = os.path.join(self.model_dir,
                                                'best_checkpoint_dir')
        self.best_model_info_path = os.path.join(self.best_checkpoint_dir,
                                                 'best_model_info.json')
        
        self.__init_value_judgment()
        if self.is_socket:
            self.build_socket_connect()

    def fit(self,
            train_features,
            train_labels,
            eval_features=None,
            eval_labels=None):
        """
        :param train_features: 训练图像路径
        :param train_labels: 训练标签路径
        :param eval_features: 验证图像路径
        :param eval_labels: 验证标签路径
        :return:
        """
        # 交叉验证

        # value = 0
        # 切换loss的epoch num
        change_loss_epoch =  round (self.epoch_num * self.change_loss_fn_threshold)

        if self.k_fold > 1:
            print("交叉验证")
            for epoch_index in range(self.epoch_num):
                if epoch_index > self.epoch_num * self.change_loss_fn_threshold:
                    self.estimator_obj.use_background_and_foreground_loss = False
                data_list = cross_validation.create_cross_validation_data(
                    self.k_fold, train_features, train_labels)
                eval_result = {}
                cross_eval_result = {}
                for j in range(self.k_fold):
                    sub_train_features, \
                        sub_train_labels, \
                        sub_eval_features, \
                        sub_eval_labels = data_list[j]
                    self.estimator_obj.train(lambda: self.__train_input_fn(
                        sub_train_features, sub_train_labels))
                    cross_eval_result = self.estimator_obj.evaluate(
                        lambda: self.__eval_input_fn(
                            sub_eval_features, sub_eval_labels))

                    for key, value in cross_eval_result.items():
                        if key not in ['global_step']:
                            if j == 0:
                                eval_result[key] = value / self.k_fold
                            else:
                                eval_result[key] += value / self.k_fold
                eval_result['global_step'] = cross_eval_result['global_step']

                print('\033[1;36m 交叉验证结果:epoch_index=' + str(epoch_index))
                for k, v in eval_result.items():
                    print(k + ' =', v)
                print('\033[0m')

                if self.is_socket:
                    eval_result['epoch_num'] = epoch_index + 1
                    eval_result["class_num"] = self.class_num
                    data_dict = list(eval_result.values())
                    data_dict = str(data_dict).encode('utf-8')
                    self.socket.send(data_dict)

                saved_model_value = self.calculate_saved_model_value_callback(*self.accuracy_weight, **eval_result)
                # 模型保存的条件
                if saved_model_value > self.value_judgment:
                    self.value_judgment = saved_model_value
                    eval_result['value_judgment'] = self.value_judgment
                    self.export_model_dir = self.model_dir + '/export_model_dir'
                    self.export_model(export_model_dir=self.export_model_dir)
                    with open(self.best_checkpoint_dir + '/best_model_info.json',
                              'w') as f:
                        eval_result_dict = {
                            k: str(v) for k, v in eval_result.items()}
                        json.dump(eval_result_dict, f, indent=1)

        else:
            loss_not_decrease_epoch_num = 0
            for epoch_index in range(0, self.epoch_num, self.eval_epoch_step):
                if eval_features is None or eval_labels is None:
                    raise ValueError('非交叉验证时必须输入验证集！')
                if epoch_index > self.epoch_num * self.change_loss_fn_threshold:
                    self.estimator_obj.use_background_and_foreground_loss = False
                self.estimator_obj.train(
                    lambda: self.__train_input_fn(train_features, train_labels))
                eval_result = self.estimator_obj.evaluate(
                    lambda: self.__eval_input_fn(eval_features, eval_labels))
                print("获得验证结果，开始数据传输")
                print('\033[1;36m 验证集结果:epoch_index=' + str(epoch_index))
                for k, v in eval_result.items():
                    print(k + ' =', v)

                print('\033[0m')
                if self.is_socket:
                    eval_result[
                        'epoch_num'] = epoch_index / self.eval_epoch_step + 1
                    eval_result["class_num"] = self.class_num
                    data_dict = list(eval_result.values())
                    data_dict = str(data_dict).encode('utf-8')
                    self.socket.send(data_dict)

                # saved_model_value = self.calculate_saved_model_value_callback(*self.accuracy_weight, **eval_result)
                saved_model_value = eval_result['loss']
                # 模型保存的条件
                if saved_model_value < self.value_judgment:
                    self.value_judgment = saved_model_value
                    eval_result['value_judgment'] = self.value_judgment
                    self.export_model_dir = self.model_dir + '/export_model_dir'
                    self.export_model(
                        export_model_dir=self.export_model_dir,
                        eval_result=eval_result)


                 # early stopping
                if (self.is_early_stop and epoch_index > change_loss_epoch):
                    loss_tolerance = 0.0005
                    if eval_result["loss"] - self.value_judgment >= loss_tolerance:
                        loss_not_decrease_epoch_num += 1
                    else:
                        loss_not_decrease_epoch_num = 0
                    if loss_not_decrease_epoch_num >  5:
                        bz_log.info("early stopping 共训练%d个epoch%d", epoch_index)
                        bz_log.info("is early stop%s", self.is_early_stop)
                        print("early stopping 共训练%d个epoch" %epoch_index)
                        break


        if self.is_socket:
            # self.socket.send(b'exit')
            self.socket.close()

    def predict(self, features):
        """
        :param features: 预测图像路径
        :return:
        """
        predictions = self.estimator_obj.predict(
            lambda: self.__predict_input_fn(features))
        return np.array([pre['classes'] for pre in predictions])

    def export_model(self, export_model_dir, eval_result):
        """
        :param export_model_dir: export模型路径
        :return:
        """
        best_checkpoint_dir = self.model_dir + '/best_checkpoint_dir/'
        self.__save_best_checkpoint(best_checkpoint_dir)
        best_checkpoint = best_checkpoint_dir + os.path.split(
            self.estimator_obj.latest_checkpoint())[1]

        if os.path.exists(self.export_model_dir):
            shutil.rmtree(self.export_model_dir)
        self.estimator_obj.export_model(export_model_dir, best_checkpoint)

        with open(self.best_checkpoint_dir + '/best_model_info.json', 'w+') as f:
            eval_result_dict = {
                k: str(v) for k, v in eval_result.items()}
            json.dump(eval_result_dict, f, indent=1)

        export_model_path = sorted(bz_path.get_subfolder_path(
            self.export_model_dir, ret_full_path=True))[0]
        if os.path.exists(self.export_model_dir + '/frozen_model'):
            shutil.rmtree(self.export_model_dir + '/frozen_model')
        os.mkdir(self.export_model_dir + '/frozen_model')
        out_pb_path = self.export_model_dir + \
            '/frozen_model/frozen_model.pb'
        convert_to_pb.convert_export_model_to_pb(export_model_path,
                                                 out_pb_path,
                                                 output_node_names="classes,"
                                                                   "softmax")
        with open(
                self.export_model_dir + \
                '/frozen_model/model_config.txt', 'w+') as f:
            f.write('model_name:frozen_model.pb' + '\n')
            f.write('input_height:' + str(self.img_crop_height) + '\n')
            f.write('input_width:' + str(self.img_crop_width) + '\n')
            f.write('input_channel:' + str(self.channel_list[0]) + '\n')
            f.write('batch_size:' + str(self.batch_size) + '\n')
            f.write('class_num:' + str(self.estimator_obj.class_num))

    def build_socket_connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("初始化socket")
        # 建立连接:
        try:
            self.socket.connect(('127.0.0.1', 9990))
            print("尝试连接9990")
        except socket.error as e:
            raise ValueError('service connection failed: %s \r\n' % e)
        # 接收欢迎消息:
        print("建立连接")
    def __save_best_checkpoint(self, best_checkpoint_dir):
        """
        :param best_checkpoint_dir: best_checkpoint路径
        :return:
        """
        if (os.path.exists(best_checkpoint_dir)):
            shutil.rmtree(best_checkpoint_dir)
        os.mkdir(best_checkpoint_dir)
        checkpoint_files = bz_path.get_file_path(
            self.model_dir, ret_full_path=True)
        (_, checkpoint_name) = os.path.split(
            self.estimator_obj.latest_checkpoint())
        for path in checkpoint_files:
            if checkpoint_name in path:
                shutil.copy(path, best_checkpoint_dir)
        shutil.copy(self.model_dir + '/checkpoint', best_checkpoint_dir + '/checkpoint')
        os.mkdir(best_checkpoint_dir + '/' + checkpoint_name + '_parameters')
        with open(
                best_checkpoint_dir + '/' + checkpoint_name + '_parameters' + \
                '/UnetEstimator.json', 'w') as f:
            estimator_obj_dict = {
                k: str(v) for k, v in self.estimator_obj.__dict__.items()}
            json.dump(estimator_obj_dict, f, indent=1)

    def __init_value_judgment(self):
        
        if os.path.exists(self.best_model_info_path):
            with open(self.best_model_info_path, 'r') as f:
                best_model_info_dict = json.load(f)
            self.value_judgment = float(best_model_info_dict['value_judgment'])
            bz_log.info("打印加载的value_judgment%f", self.value_judgment)
        else:
            self.value_judgment = 1
            bz_log.info("不存在")

    def __train_input_fn(self, features, labels):
        """
        :param features: 训练图像路径
        :param labels: 训练标签路径
        :return:
        """
        train_dataset = dataset.ImageLabelDataSet(
            img_list=features,
            label_list=labels,
            channels_list=self.channel_list,
            file_extension_list=self.file_extension_list,
            height_width=(self.img_height, self.img_width),
            crop_height_width=(self.img_crop_height, self.img_crop_width),
            batch_size=self.batch_size,
            num_epochs=self.eval_epoch_step,
            shuffle=True,
            mode='train',
            task='segmentation')
        return train_dataset.create_dataset()

    def __eval_input_fn(self, features, labels):
        """
        :param features: 验证图像路径
        :param labels: 验证标签路径
        :return:
        """
        eval_dataset = dataset.ImageLabelDataSetBaseClass(
            img_list=features,
            label_list=labels,
            channels_list=self.channel_list,
            file_extension_list=self.file_extension_list,
            height_width=(self.img_height, self.img_width),
            crop_height_width=(self.img_crop_height, self.img_crop_width),
            batch_size=self.batch_size,
            num_epochs=1,
            mode='evaluate',
            task='segmentation')
        return eval_dataset.create_dataset()

    def __predict_input_fn(self, features):
        """
        :param features: 预测图像路径
        :return:
        """
        predict_dataset = dataset.ImageLabelDataSetBaseClass(
            img_list=features,
            label_list=None,
            channels_list=self.channel_list[0],
            file_extension_list=self.file_extension_list[0],
            height_width=(self.img_height, self.img_width),
            crop_height_width=(self.img_crop_height, self.img_crop_width),
            mode='predict',
            task='segmentation')
        return predict_dataset.create_dataset()
