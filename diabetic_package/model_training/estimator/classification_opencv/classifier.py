# ---------------------------------
#   !Copyright(C) 2019,北京博众
#   All right reserved.
#   文件名称：classifier.py
#   摘   要：创建分类dataset，输入数据，保存模型，控制模型训练过程。
#   当前版本:2019121816
#   作   者：戴卓伦　崔宗会
#   完成日期：2019-12-18
# ---------------------------------

import numpy as np
import os
import shutil
import json
import tensorflow as tf
import socket
from ....dataset_common import dataset
from ....file_operator import bz_path, cross_validation
from ....image_processing_operator import tensorflow_image_processing_map
from ....machine_learning_common import convert_to_pb
from ....machine_learning_common.accuracy import python_accuracy
from diabetic_package.log.log import bz_log

class AlexnetDataSet(dataset.ImageLabelDataSetBaseClass):
    def preprocess_image_map(self, data_dict):
        """
        图像处理的map函数
        :param data_dict:
        :param is_mirroring:
        :return:
        """
        image = data_dict['img']
        label = data_dict['label']
        # image, label = tensorflow_image_processing_map.random_rescale_image_and_label_tf_map(
        #     image, label, min_scale=0.5, max_scale=2.0)
        #
        # image, label = tensorflow_image_processing_map.random_crop_or_pad_image_and_label_tf_map(
        #     image, label=label, crop_height=self.crop_height_width[0], crop_width=self.crop_height_width[1])
        # # image,label=tensorflow_image_processing_map.flip_image_and_label_tf_map(image,label)
        # image, label = tensorflow_image_processing_map.random_rotate_image_and_label_tf_map(image, label,
        #                                                                                     max_angle=40)
        # image, label = tensorflow_image_processing_map.flip_image_and_label_tf_map(image, label)
        # # # image, label = tensorflow_image_processing_map.flip_up_down_image_and_label_tf_map(image, label)
        # # # image, label = tensorflow_image_processing_map.flip_left_right_image_and_label_tf_map(image, label)
        # # # image, label = tensorflow_image_processing_map.transpose_image_image_and_label_tf_map(image, label)
        # # # image, label = tensorflow_image_processing_map.translate_image_and_label_tf_map(image, label, dx=10, dy=-60.0)
        # image, label = tensorflow_image_processing_map.random_translate_image_and_label_tf_map(image, label,
        #                                                                                        max_dx=30,
        #                                                                                        max_dy=30)
        # #
        # # # image = tensorflow_image_processing_map.random_brightness_image_tf_map(image, 10)
        # # # image = tensorflow_image_processing_map.random_contrast_image_tf_map(image, 0, 200)
        # # # image = tensorflow_image_processing_map.random_hue_image_tf_map(image, 0.5)
        # # # image = tensorflow_image_processing_map.random_saturation_image_tf_map(image, 0.5,3.0)
        # image = tensorflow_image_processing_map.add_random_noise_tf_map(image, 0.5, 100)
        # # image = tensorflow_image_processing_map.add_gaussian_noise_tf_map(image, 0.5, 3.0)
        #
        # # image = tensorflow_image_processing_map.add_salt_and_pepper_noise_pyfunc_tf_map(image, 0.001, 255)
        return image, label


class Classifier:
    def __init__(self,
                 estimator_obj,
                 accuracy_weight,
                 class_num,
                 model_dir='./',
                 height_width=(227, 227),
                 crop_height_width=(227, 227),
                 k_fold=1,
                 channels_list=[3],
                 file_extension_list=['bmp'],
                 batch_size=1,
                 epoch_num=1,
                 eval_epoch_step=1,
                 calculate_saved_model_value_callback=python_accuracy.calculate_class_weighted_recall,
                 is_socket=False,
                 is_early_stop=True
                 ):
        """

        :param model_dir: checkpoint保存目录
        :param estimator_obj: estimator对象
        :param height_width: 输入dataset后resize图像目标高宽
        :param crop_height_width: crop图像目标高宽
        :param k_fold: 交叉验证折数，正整数，值为1时不做交叉验证
        :param channels_list: list形式，分类任务传一个元素，为图像通道数
        :param file_extension_list: list形式，分类任务传一个元素，为图像格式
        :param batch_size:
        :param epoch_num:
        """
        # self.model_dir = os.path.join(model_dir, 'model_dir')
        self.model_dir = model_dir
        self.best_checkpoint_dir = os.path.join(model_dir, 'best_checkpoint_dir')
        self.export_model_dir = os.path.join(model_dir, 'export_model_dir')
        self.best_model_info_path = os.path.join(self.best_checkpoint_dir, 'best_model_info.json')
        self.estimator_obj = estimator_obj
        self.height_width = height_width
        self.crop_height_width = crop_height_width
        self.k_fold = k_fold
        self.channels_list = channels_list
        self.file_extension_list = file_extension_list
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.class_num = class_num
        self.value_judgment = - np.inf
        self.accuracy_weight = accuracy_weight
        self.eval_epoch_step = eval_epoch_step
        self.calculate_saved_model_value_callback = calculate_saved_model_value_callback
        self.is_socket = is_socket
        self.is_early_stop = is_early_stop
        self.__init_value_judgment()
        if self.is_socket:
            self.build_socket_connect()
        self.__check_param()


    def fit(self, train_images_path, train_labels, eval_images_path, eval_labels, train_epochs_before_val=1):
        """

        :param train_images_path: nd.array形式，元素为训练集图片全路径
        :param train_labels: nd.array形式，训练集对应labels
        :param eval_images_path: nd.array形式，元素为验证集图片全路径，交叉验证时为None
        :param eval_labels: nd.array形式，训练集对应labels，交叉验证时为None
        :param train_epochs_before_val: 训练几个eopch验证一次
        :return:
        """
        loss_not_decrease_epoch_num = 0
        if train_epochs_before_val < 1:
            raise ValueError("train_epochs_before_val不得小于1")
        if self.k_fold > 1:
            for epoch_index in range(self.epoch_num):
                dataset_list = cross_validation.create_cross_validation_data(
                    self.k_fold, train_images_path, train_labels)
                eval_result = {}
                cross_eval_result = {}
                for j in range(self.k_fold):
                    print('交叉验证第%d/%d次训练开始' % (j + 1, self.k_fold))
                    train_images_path_cross, train_labels_cross, eval_images_path_cross, eval_labels_cross = \
                        dataset_list[j]
                    train_epochs = 0
                    while train_epochs != train_epochs_before_val:
                        self.estimator_obj.train(input_fn=lambda: self.__train_input_fn(
                            train_images_path_cross, train_labels_cross))
                        train_epochs += 1
                    cross_eval_result = self.estimator_obj.evaluate(
                        input_fn=lambda: self.__eval_input_fn(
                            eval_images_path_cross, eval_labels_cross))
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
                # value_judgment = self.__cal_value_judgment(eval_result)
                saved_model_value = self.calculate_saved_model_value_callback(*self.accuracy_weight, **eval_result)
                if saved_model_value > self.value_judgment:
                    self.value_judgment = saved_model_value
                    eval_result['value_judgment'] = saved_model_value
                    # self.__save_best_checkpoint(eval_result)
                    self.export_model(export_model_dir=self.export_model_dir, eval_result=eval_result)
        else:
            if eval_images_path is None or eval_labels is None:
                raise ValueError("非交叉验证训练时，应输入验证集")
            for epoch_index in range(0, self.epoch_num, self.eval_epoch_step):
                train_epochs = 0
                while train_epochs != train_epochs_before_val:
                    self.estimator_obj.train(input_fn=lambda: self.__train_input_fn(
                        train_images_path, train_labels))
                    train_epochs += 1
                eval_result = self.estimator_obj.evaluate(
                    input_fn=lambda: self.__eval_input_fn(
                        eval_images_path, eval_labels))
                print('\033[1;36m 验证集结果:epoch_index=' + str(epoch_index))

                for k, v in eval_result.items():
                    print(k + ' =', v)
                print('\033[0m')
                if self.is_socket:
                    eval_result['epoch_num'] = epoch_index / self.eval_epoch_step + 1
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
                    self.export_model(export_model_dir=self.export_model_dir, eval_result=eval_result)

                # early stopping
                if (self.is_early_stop):
                    loss_tolerance = 0.0005
                    if eval_result["loss"] - self.value_judgment >= loss_tolerance:
                        loss_not_decrease_epoch_num += 1
                    else:
                        loss_not_decrease_epoch_num = 0
                    if loss_not_decrease_epoch_num > 8:
                        print("early stopping 共训练%d个epoch" % epoch_index)
                        break
        if self.is_socket:
            # self.socket.send(b'exit')
            self.socket.close()

    def predict(self, features):
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
                                                 output_node_names=
                                                                   "softmax")
        with open(
                self.export_model_dir + \
                '/frozen_model/model_config.txt', 'w+') as f:
            f.write('model_name:frozen_model.pb' + '\n')
            f.write('input_height:' + str(self.crop_height_width[0]) + '\n')
            f.write('input_width:' + str(self.crop_height_width[1]) + '\n')
            f.write('input_channel:' + str(self.channels_list[0]) + '\n')
            f.write('batch_size:' + str(self.batch_size) + '\n')
            f.write('class_num:' + str(self.class_num))

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

    def __check_param(self):
        if not isinstance(self.accuracy_weight, list) or len(self.accuracy_weight) != self.estimator_obj.class_num:
            raise ValueError("accuracy_weight参数应为长度为分类数目的list")
        if self.channels_list[0] != self.estimator_obj.img_shape[2]:
            raise ValueError('channel_list的第一个通道数与AlexnetEstimator的img通道数不相符')

    def __init_value_judgment(self):
        if os.path.exists(self.best_model_info_path):
            with open(self.best_model_info_path, 'r') as f:
                best_model_info_dict = json.load(f)
            self.value_judgment = float(best_model_info_dict['loss'])
            bz_log.info("打印加载的value_judgment%f", self.value_judgment)
        else:
            self.value_judgment = 1000
            bz_log.info("不存在")

    def __save_best_checkpoint(self, best_checkpoint_dir):
        """

        :param eval_result: eval模式下输出，字典形式
        :return:
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if (os.path.exists(best_checkpoint_dir)):
            shutil.rmtree(best_checkpoint_dir)
        os.makedirs(best_checkpoint_dir)
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
                '/AlexNet_Estimator.json', 'w') as f:
            estimator_obj_dict = {
                k: str(v) for k, v in self.estimator_obj.__dict__.items()}
            json.dump(estimator_obj_dict, f, indent=1)

    def __train_input_fn(self, features, labels):
        train_dataset = AlexnetDataSet(
            img_list=features,
            label_list=labels,
            channels_list=self.channels_list,
            file_extension_list=self.file_extension_list,
            height_width=self.height_width,
            crop_height_width=self.crop_height_width,
            batch_size=self.batch_size,
            num_epochs=self.eval_epoch_step,
            shuffle=True)
        return train_dataset.create_dataset()

    def __eval_input_fn(self, features, labels):
        eval_dataset = AlexnetDataSet(
            img_list=features,
            label_list=labels,
            channels_list=self.channels_list,
            file_extension_list=self.file_extension_list,
            height_width=self.height_width,
            crop_height_width=self.crop_height_width,
            batch_size=self.batch_size,
            num_epochs=1,
            mode='evaluate')
        return eval_dataset.create_dataset()

    def __predict_input_fn(self, features):
        predict_dataset = AlexnetDataSet(
            img_list=features,
            label_list=None,
            channels_list=self.channels_list,
            file_extension_list=self.file_extension_list,
            height_width=self.height_width,
            crop_height_width=self.crop_height_width,
            mode='predict')
        return predict_dataset.create_dataset()
