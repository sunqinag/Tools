# ----------------------------
#!  Copyright(C) 2020
#   All right reserved.
#   文件名称：python_split_train_eval_test_data.py
#   摘   要：自编码器模型,实现只训练正样本对缺陷进行检测的功能
#   当前版本:
#   作   者：陈瑞侠,小孙强
#   完成日期：2020-9-2
# -----------------------------
import tensorflow as tf
import os, shutil
import json
import socket
from diabetic_package.model_training.autoencoder.dataset import Dataset
from diabetic_package.model_training.autoencoder import loss
from diabetic_package.model_training.autoencoder import model
from tensorflow.python.tools.freeze_graph import graph_util

from diabetic_package.file_operator import bz_path
from diabetic_package.log.log import bz_log
from diabetic_package.image_processing_operator.tensorflow_image_processing import random_brightness_image_tf

class Autoencoder:
    def __init__(self,
                 data_dir,
                 model_dir="./model_dir",
                 log_dir="./log",
                 epoch_num=100,
                 batch_size=5,
                 img_size=(512, 512),
                 channel= 1,
                 model_type=model.Model_concat,
                 loss_type=loss.custom_loss,
                 is_socket=False,
                 is_early_stop=True):
        '''
        # model_type总共有三种类型:
        # Model_concat:基础的卷积编码翻卷机解码结构,包含一层全连接层和一个跨层连接
        # Model_no_concat:基础的卷积编码翻卷机解码结构
        # Model_dense:包含基础的卷积编码和反卷积编码及全连接层的编码解码结构
        # loss_type总共有三种类型:
        # euclidean_distance_loss:欧氏距离
        # l2_loss:l2正则化loss
        # custom_loss:自定义loss
        '''
        self.epoch = epoch_num
        self.batch_size = batch_size
        self.image_size = img_size
        self.height = img_size[0]
        self.width = img_size[1]
        self.channel = channel
        self.image_dir = data_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.is_socket = is_socket
        self.early_stop = is_early_stop
        self.model= model_type
        self.loss = loss_type

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        # self.image_num = len(os.listdir(self.image_dir))
        image_list = bz_path.get_file_path(self.image_dir, ".jpg")
        self.image_num = len(image_list)
        self.step_num = int(self.image_num  / self.batch_size)
        self.train_dataset, self.val_dataset = Dataset(self.image_dir,
                                                       self.image_size,
                                                       self.batch_size,
                                                       self.epoch).get_next()
        with tf.name_scope('input_and_aug_TF'):
            self.inputs = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.image_size[0],
                                                self.image_size[1], 1],
                                         name='inputs')
            self.inputs = random_brightness_image_tf(self.inputs, max_delta=10)
        self.logits=self.model(self.inputs)
        with tf.name_scope('Loss'):
            self.loss_op = self.loss(self.inputs, self.logits)

        with tf.name_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_op)

        with tf.name_scope('Saver'):
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        with tf.name_scope("Summery"):
            tf.summary.scalar("loss", self.loss_op)
            # tf.summary.scalar('regularization',self.regularization)
            tf.summary.image("concat_result_train",
                             tf.concat([tf.cast(self.inputs, tf.uint8),
                                        tf.cast(self.logits, tf.uint8)],
                                        axis=2),
                                        max_outputs=6)
            self.summary_op = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()
        self.best_checkpoint_dir = os.path.join(self.model_dir,
                                                'best_checkpoint_dir')
        self.best_model_info_path = os.path.join(self.best_checkpoint_dir,
                                                 'best_model_info.json')
        self.export_model_dir = self.model_dir + '/export_model_dir/frozen_model/'
        self._init_value_judgment()
        if self.is_socket:
            self.build_socket_connect()

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        ###保存模型并添加需要输出的量
        model_file = tf.train.latest_checkpoint(self.model_dir)
        # # 加载之前的训练数据继续训练
        if model_file is not None:
            print('load model:' + model_file)
            self.saver.restore(self.sess, model_file)
        bz_log.info("网络初始化成功,开始训练")
        loss_not_decrease_epoch_num = 0
        summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        for epoch_index in range(self.epoch):
            print('epoch:', epoch_index)
            for step in range(self.step_num):
                global_step = epoch_index*self.image_num+step
                batch_image = self.sess.run(self.train_dataset)
                _, loss_, summary, logits_ = self.sess.run(
                    [self.train_op, self.loss_op, self.summary_op, self.logits],
                    feed_dict={self.inputs: batch_image['img']})

                step_index = step + epoch_index * self.step_num
                print("第"+  str(step_index) +"个step--:")
                print("loss:", loss_)
                summary_writer.add_summary(summary,global_step)
            #验证集
            eval_result = {}
            val_batch = self.sess.run(self.val_dataset)
            val_loss = self.sess.run(
                [self.loss_op],
                feed_dict={
                    self.inputs: val_batch['img']})
            eval_result['epoch_num'] = epoch_index + 1
            eval_result['loss'] = val_loss[0]
            eval_result["class_num"] = 255
            if self.is_socket:
                data_dict = list(eval_result.values())
                data_dict = str(data_dict).encode('utf-8')
                self.socket.send(data_dict)

            saved_model_value = eval_result['loss']
            print("eval loss------", saved_model_value)

            ckpt_file = self.model_dir + '/' + 'mm.ckpt'
            print("更新保存于：", ckpt_file)
            self.saver.save(self.sess, ckpt_file, global_step=epoch_index + 1)
            print(eval_result)

            # 模型保存的条件
            if saved_model_value < self.value_judgment:
                self.value_judgment = saved_model_value
                latest_ckpt = 'mm.ckpt-' + str(epoch_index + 1)
                print("保存最佳模型")
                self.save_best_checkpoint(latest_ckpt, self.best_checkpoint_dir,
                                          eval_result)
                print("导出pb模型")
                self.export_model(self.sess)

            # early stopping
            loss_tolerance = 0.0005
            if eval_result["loss"] - self.value_judgment >= loss_tolerance:
                loss_not_decrease_epoch_num += 1
            else:
                loss_not_decrease_epoch_num = 0
            if loss_not_decrease_epoch_num > 8:
                print("导出pb模型")
                self.export_model(self.sess)
                print("early stopping 共训练%d个epoch" % epoch_index)
                break

        if self.is_socket:
            self.socket.close()

    def export_model(self, sess):
        """
        :param export_model_dir: export模型路径
        :return:
        """
        if os.path.exists(self.export_model_dir):
            shutil.rmtree(self.export_model_dir)
        os.makedirs(self.export_model_dir)
        print("转pb----------")
        output_node_names = 'output/BiasAdd'

        out_pb_path = self.export_model_dir + "frozen_model.pb"
        self.convert_ckpt2pb(sess, out_pb_path,
                             output_node_names)
        with open(
                self.export_model_dir + \
                '/model_config.txt', 'w+') as f:
            f.write('model_name:frozen_model.pb' + '\n')
            f.write('input_height:' + str(self.height) + '\n')
            f.write('input_width:' + str(self.width) + '\n')
            f.write('input_channel:' + str(self.channel) + '\n')
            f.write('batch_size:' + str(self.batch_size))

    def save_best_checkpoint(self, latest_ckpt, best_checkpoint_dir, eval_result):
        """
        :param best_checkpoint_dir: best_checkpoint路径
        :return:
        """
        if (os.path.exists(best_checkpoint_dir)):
            shutil.rmtree(best_checkpoint_dir)
        os.mkdir(best_checkpoint_dir)
        checkpoint_files = bz_path.get_file_path(
            self.model_dir, ret_full_path=True)

        for path in checkpoint_files:
            if latest_ckpt in path:
                shutil.copy(path, best_checkpoint_dir)
        shutil.copy(self.model_dir + '/checkpoint',
                    best_checkpoint_dir + '/checkpoint')

        with open(self.best_checkpoint_dir + '/best_model_info.json',
                  'w+') as f:
            eval_result_dict = {
                k: str(v) for k, v in eval_result.items()}
            json.dump(eval_result_dict, f, indent=1)

    def _init_value_judgment(self):
        if os.path.exists(self.best_model_info_path):
            with open(self.best_model_info_path, 'r') as f:
                best_model_info_dict = json.load(f)
            self.value_judgment = float(
                best_model_info_dict['loss'])
        else:
            self.value_judgment = 100000

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

    def convert_ckpt2pb(self, sess, output_graph, output_node_names):
        '''
        :param input_checkpoint:
        :param output_graph: PB模型保存路径
        :return:
        '''

        output_graph_def = graph_util.convert_variables_to_constants(
            # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(
                ","))  # 如果有多个输出节点，以逗号隔开
        print(output_graph)
        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(
            output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == '__main__':
    data_dir = "../autoencoder/logOK/OK"

    # autoencoder = Autoencoder(data_dir,model_dir,epoch_num,batch_size,channel=1)
    # autoencoder.train()
    Autoencoder(data_dir).train()
    # for image_path in [image_dir+'/'+file for file in os.listdir(image_dir)]:
    #     ckptfile = './model_dir/3510440448.0.ckpt-49.meta'
    #     predict(image_patglobal_setph,ckptfile)
