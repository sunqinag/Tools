# ---------------------------------
#   !Copyright(C) 2019,北京博众
#   All right reserved.
#   文件名称：base_classify_estimator.py
#   摘   要：图像分类的基类
#   当前版本:2019091818
#   作   者：戴卓伦，崔宗会
#   完成日期：2019-09-18
# ---------------------------------

import shutil
import os
import tensorflow as tf
from ....machine_learning_common.accuracy import tensorflow_accuracy
from ....machine_learning_common.loss import loss as loss_module


class BaseClassifyEstimator(tf.estimator.Estimator):
    def __init__(self,
                 label_weight,
                 class_num=2,
                 model_dir='./model',
                 regularizer_scale=(1e-7, 1e-6),
                 optimizer_fn=tf.train.AdamOptimizer,
                 learning_rate=1e-3,
                 tensors_to_log={'probabilities': 'softmax:0'},
                 assessment_list=['accuracy', 'iou', 'recall', 'precision', 'auc']
                 ):
        """

            :param class_num： 分类类别数
            :param model_dir: model_dir前缀
            :param regularizer_scale: l1,l2正则项系数,tumple格式
            :param label_weight: 不同类别样本加权，可为float或list，float scalar将直接乘到loss，list会分类别加权到loss
            :param optimizer_fn： 激活函数，双元素tumple格式，分别表示卷积层和全连接层的激活函数。
            :param learning_rate： 学习率
            :param tensors_to_log： train时输出的log，字典形式
            :return
        """
        # self.model_path = os.path.join(model_dir, 'model_dir')
        # self.model_dir =model_dir
        self.regularizer_scale = regularizer_scale
        self.label_weight = label_weight
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.tensors_to_log = tensors_to_log
        self.class_num = class_num
        self.assessment_list = assessment_list
        self.__check_param()
        os.environ[
            "CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
        session_config = tf.ConfigProto()
        # session_config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 程序最多只能占用指定gpu50%的显存
        session_config.gpu_options.allow_growth = True  # 程序按需申请内存
        run_config = tf.estimator.RunConfig(keep_checkpoint_max=5, session_config=session_config)
        super(BaseClassifyEstimator, self).__init__(
            model_fn=self.__model_fn,
            model_dir=model_dir,
            config=run_config)

    def network(self, features, is_training):
        """

        :param features: network输入层
        :param is_training: 是否在train模式下调用
        :return:
        """
        raise NotImplementedError('请重写子类network方法')

    def serving_input_receiver_fn(self):
        """

        :return:
        """
        raise NotImplementedError('请重写子类serving_input_receiver_fn方法')

    def export_model(self, export_model_dir, checkpoint_path):
        """

        :param export_model_dir: 输出model路径
        :param checkpoint_path: checkpoint地址
        :return:
        """
        if os.path.exists(export_model_dir):
            shutil.rmtree(export_model_dir)
        os.makedirs(export_model_dir)
        self.export_savedmodel(
            export_dir_base=export_model_dir,
            serving_input_receiver_fn=self.serving_input_receiver_fn,
            checkpoint_path=checkpoint_path)

    def _regularizer(self):
        """

        :param weights: tensorflow 自行传递
        :return:
        """
        return tf.contrib.layers.l1_l2_regularizer(
            self.regularizer_scale[0], self.regularizer_scale[1])

    def __check_param(self):
        if not isinstance(self.label_weight, list):
            raise ValueError('输入的label_weight应该为list形式')
        if len(self.label_weight) != self.class_num:
            raise ValueError('输入的label_weight列表长度应与类别个数一致')
        if len(self.regularizer_scale) != 2 or self.regularizer_scale[0] < 0 or self.regularizer_scale[1] < 0:
            raise ValueError('输入的正则项系数列表长度必须为2且两个值都要大于等于0')

    def __generate_estimator_spec(self, mode, logits, labels):
        """

        :param mode: 调用模式
        :param logits: feature层
        :param labels: 用于验证的labels
        :return:
        """
        softmax = tf.nn.softmax(logits, axis=-1, name='softmax')
        predictions = {'classes': tf.argmax(softmax, axis=1, name='classes'),
                       'probabilities': softmax}
        if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
            regularization_loss = tf.losses.get_regularization_loss()
            labels = tf.cast(labels, tf.int32)
            class_weight_loss = loss_module.calculate_class_weighted_loss(logits, labels, self.label_weight)
            # tf.identity(regularization_loss, name='regularization_loss')
            tf.summary.scalar('regularization_loss', regularization_loss)
            # tf.identity(class_weight_loss, name='class_weight_loss')
            tf.summary.scalar('class_weight_loss', class_weight_loss)
            loss = class_weight_loss + regularization_loss
            # tf.identity(loss, name='loss')
            # tf.summary.scalar('loss', loss)
        else:
            loss = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            assessment_dict = tensorflow_accuracy.get_assessment_result(
                labels, predictions['classes'],
                self.class_num, 'train', self.assessment_list)

            optimizer = self.optimizer_fn(learning_rate=self.learning_rate)
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step())
            for key, value in assessment_dict.items():
                tf.summary.scalar(key, value[1])
        else:
            train_op = None
        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = tensorflow_accuracy.get_assessment_result(
                labels, predictions['classes'],
                self.class_num, 'evaluate', self.assessment_list)
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {'result': tf.estimator.export.PredictOutput(
                predictions['classes'])}
        else:
            export_outputs = None
        logging_hook = tf.train.LoggingTensorHook(
            tensors=self.tensors_to_log,
            every_n_iter=50)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=[logging_hook],
            export_outputs=export_outputs)

    def __model_fn(self, features, mode):
        """

        :param features:
        :param mode: 调用模式
        :return:
        """
        input_layer = features['img']
        if mode == tf.estimator.ModeKeys.PREDICT:
            labels = None
        else:
            labels = features['label']
        logits = self.network(
            input_layer, mode == tf.estimator.ModeKeys.TRAIN)
        return self.__generate_estimator_spec(mode, logits, labels)

