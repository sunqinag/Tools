# -*- coding: utf-8 -*-
# ----------------------------
#!  Copyright(C) 2020
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：刘恩甫
#   完成日期：2020-x-x
# -----------------------------
import tensorflow as tf
import json
import os
import shutil
import numpy as np
import socket

from .utils import config
from diabetic_package.file_operator import bz_path
from diabetic_package.machine_learning_common import convert_to_pb
from diabetic_package.log.log import bz_log
from .dataset.tfexample_converter import parse_yolo_example

class detector():
    def __init__(self,
                 estimator_obj,
                 model_dir,
                 class_num,
                 eval_epoch_step=1,
                 epoch_num=100,
                 batch_size=4,
                 is_socket=False,
                 is_early_stop=True
                 ):
        self.estimator_obj=estimator_obj
        self.epoch_num=epoch_num
        self.channel=3
        self.batch_size=batch_size
        self.eval_epoch_step=eval_epoch_step
        self.model_dir=model_dir
        self.export_model_dir = self.model_dir+'/export_model_dir/'
        self.best_checkpoint_dir = self.model_dir+'/best_checkpoint_dir'
        self.class_num=class_num
        self.is_socket = is_socket
        self.is_early_stop = is_early_stop

        if self.is_socket:
            self.build_socket_connect()

    def fit(self,train_records,eval_records):
        # 模型保存条件与步数控制
        if os.path.exists(self.model_dir + os.sep + 'best_checkpoint_dir/eval_metrics.json'):
            with open(self.model_dir + os.sep + 'best_checkpoint_dir/eval_metrics.json') as f:
                last_eval_res = json.load(f)
            loss_value = float(last_eval_res['loss'])
        else:
            loss_value = 10000
        loss_not_decrease_epoch_num = 0
        for epoch_index in range(0, self.epoch_num, self.eval_epoch_step):
            self.estimator_obj.train(input_fn=lambda:self.train_input_fn(train_records))
            eval_result= self.estimator_obj.evaluate(input_fn=lambda:self.eval_input_fn(eval_records))
            print('第',epoch_index,'个epoch的验证结果:')
            for k,v in eval_result.items():
                print(k, ':', v)

            #通过socket传输eval_result
            if self.is_socket:
                bz_log.info("epoch_num%d:", epoch_index + 1)
                eval_result['epoch_num'] = epoch_index / self.eval_epoch_step + 1
                eval_result["class_num"] = self.class_num
                data_dict = list(eval_result.values())[1:]
                data_dict = str(data_dict).encode('utf-8')
                self.socket.send(data_dict)

            # 模型更新与保存,两个判断条件并存
            eval_loss_value = self.__get_saved_model_value(eval_result)
            if eval_loss_value <= loss_value:
                #导出模型
                self.export_model()
                with open(self.best_checkpoint_dir + '/eval_metrics.json', 'w') as f:
                    eval_result_dict = {k: str(v) for k, v in eval_result.items()}
                    json.dump(eval_result_dict, f, indent=1)
                loss_value=eval_loss_value

            # early stopping
            if (self.is_early_stop):
                loss_tolerance = 0.0005
                if eval_result["loss"] - loss_value >= loss_tolerance:
                    loss_not_decrease_epoch_num += 1
                else:
                    loss_not_decrease_epoch_num = 0
                if loss_not_decrease_epoch_num > 5:
                    print("early stopping 共训练%d个epoch" % epoch_index)
                    break

        #保存freeze pb模型,在训练终止时进行转换
        self.convert_export_model_to_pb()

    def train_input_fn(self, train_records):
        dataset = tf.data.TFRecordDataset(train_records)
        dataset = dataset.map(self.parse_one, num_parallel_calls=10).\
            apply(tf.data.experimental.shuffle_and_repeat(int(3e2), self.eval_epoch_step)).\
            batch(self.batch_size, drop_remainder=True).prefetch(self.batch_size)
        return dataset

    def eval_input_fn(self,eval_records):
        dataset = tf.data.TFRecordDataset(eval_records)
        dataset = dataset.map(self.parse_one,num_parallel_calls=10).\
            apply(tf.data.experimental.shuffle_and_repeat(int(3e2), self.eval_epoch_step)).\
            batch(self.batch_size, drop_remainder=True).prefetch(self.batch_size)
        return dataset

    def parse_one(self,example_proto):
        '''读取并处理每张图片和对应的label'''
        image_decoded, label_sbbox, label_mbbox, label_lbbox, \
            sbboxes, mbboxes, lbboxes=parse_yolo_example(example_proto,self.class_num)

        return {'img':image_decoded},{'label_sbbox':label_sbbox,
                                      'label_mbbox':label_mbbox,
                                      'label_lbbox':label_lbbox,
                                      'sbboxes':sbboxes,
                                      'mbboxes':mbboxes,
                                      'lbboxes':lbboxes}

    def read_numpy_file(self,label_string):
        label_string=label_string.decode('ascii')
        label_data=np.load(label_string,allow_pickle=True)
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes=\
            [label_data[i].astype(np.float32) for i in range(6)]
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __get_saved_model_value(self, eval_result):
        '''
        依靠eval_result中的指标作为条件进行模型的更新
        :param eval_result:
        :return:
        '''
        eval_recall_value = 0
        for k,v in eval_result.items():
            if 'recall' in k:eval_recall_value += v
        eval_loss_value=eval_result['loss'] #验证loss越来越低的话，保存模型
        return eval_loss_value

    def export_model(self):
        #保存best_checkpoint
        self.__save_best_checkpoint()
        best_checkpoint = self.best_checkpoint_dir+os.sep+\
                          os.path.split(self.estimator_obj.latest_checkpoint())[1]

        #保存export_saved_model
        self.export_saved_model(self.export_model_dir,best_checkpoint)

    def __save_best_checkpoint(self):
        if (os.path.exists(self.best_checkpoint_dir)):
            shutil.rmtree(self.best_checkpoint_dir)
        os.mkdir(self.best_checkpoint_dir)

        checkpoint_files = bz_path.get_file_path(self.model_dir, ret_full_path=True)
        (_, checkpoint_name) = os.path.split(self.estimator_obj.latest_checkpoint())

        for path in checkpoint_files:
            if checkpoint_name in path:shutil.copy(path, self.best_checkpoint_dir)
        shutil.copy(self.model_dir + os.sep + 'checkpoint', self.best_checkpoint_dir)
        os.mkdir(self.best_checkpoint_dir + os.sep + checkpoint_name + '_parameters')
        with open(self.best_checkpoint_dir + os.sep + checkpoint_name + '_parameters' + \
                '/YOLO_Estimator_config.json', 'w') as f:
            estimator_obj_dict = \
                {k: str(v) for k, v in self.estimator_obj.__dict__.items()}
            json.dump(estimator_obj_dict, f, indent=1)

    def export_saved_model(self, export_model_dir, checkpoint_path):
        if os.path.exists(export_model_dir):
            shutil.rmtree(export_model_dir)
        os.mkdir(export_model_dir)

        self.estimator_obj.export_savedmodel(
            export_model_dir,
            serving_input_receiver_fn=self._serving_input_receiver_fn,
            checkpoint_path=checkpoint_path)

    def _serving_input_receiver_fn(self):
        '''
        给export模型添加输入节点
        :return:
        '''
        img = tf.placeholder(dtype=tf.float32,
                             shape=[None,
                                    config.img_shape[0],
                                    config.img_shape[1],
                                    self.channel],
                             name='img')
        features = {'img': img}
        return tf.estimator.export.ServingInputReceiver(features, features)

    def convert_export_model_to_pb(self):
        export_model_path = sorted(bz_path.get_subfolder_path(
            self.export_model_dir, ret_full_path=True))[0]
        if os.path.exists(self.export_model_dir + '/frozen_model'):
            shutil.rmtree(self.export_model_dir + '/frozen_model')
        os.mkdir(self.export_model_dir + '/frozen_model')
        out_pb_path = self.export_model_dir + \
                      '/frozen_model/frozen_model.pb'

        convert_to_pb.convert_export_model_to_pb(
            export_model_path,
            out_pb_path,
            output_node_names="outputs")

        with open(self.export_model_dir + \
                '/frozen_model/model_config.txt', 'w') as f:
            f.write('model_name:frozen_model.pb' + '\n')
            f.write('input_height:' + str(config.img_shape[0]) + '\n')
            f.write('input_width:' + str(config.img_shape[1]) + '\n')
            f.write('input_channel:' + str(self.channel)+ '\n')
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
        print("建立连接")