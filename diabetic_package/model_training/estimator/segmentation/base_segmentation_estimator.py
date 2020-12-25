# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：base_segmentation_estimator.py
#   摘   要：图像分割的基类
#   当前版本:2019092610
#   作   者：王茜，崔宗会
#   完成日期：2018-09-26
# ---------------------------------

import tensorflow as tf
import os
import shutil

from ....image_processing_operator import tensorflow_image_processing
from ....machine_learning_common.accuracy import tensorflow_accuracy
from diabetic_package.log.log import bz_log
tf.logging.set_verbosity(tf.logging.INFO)


class BaseSegmentationEstimator(tf.estimator.Estimator):
    def __init__(self,
                 class_num=2,
                 model_dir='./model_dir',
                 regularizer_scale=(0, 0),
                 optimizer_fn=tf.train.AdamOptimizer,
                 background_and_foreground_loss_weight=(0.45, 0.55),
                 class_loss_weight=(1, 1, 1),
                 max_img_outputs=6,
                 learning_rate=1e-3,
                 tensor_to_log={'probablities': 'softmax'},
                 assessment_list=['accuracy', 'iou', 'recall', 'precision', 'auc']):
        """
        :param model_dir: 模型路径
        :param regularizer_scale: l1, l2正则系数, tuple或者list格式
        :param learning_rate: 学习率
        :param class_num: 分类个数
        :param optimizer_fn: 优化器函数
        :param background_and_foreground_loss_weight: 前景背景权重
        :param class_loss_weight: 像素类别权重
        :param max_outputs: 图像最大输出个数
            :param assessment_list: 评价模型的list
            如果只使用前后背景的loss
        """
        self.class_num = class_num
        self.regularizer_scale = regularizer_scale
        self.optimizer_fn = optimizer_fn
        self.background_and_foreground_loss_weight = \
            background_and_foreground_loss_weight
        self.class_loss_weight = class_loss_weight
        self.max_img_outputs = max_img_outputs
        self.learning_rate = learning_rate
        self.tensor_to_log = tensor_to_log
        self.assessment_list = assessment_list
        os.environ[
            "CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
        session_config = tf.ConfigProto()
        # session_config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 程序最多只能占用指定gpu50%的显存
        session_config.gpu_options.allow_growth = True  # 程序按需申请内存
        run_config = tf.estimator.RunConfig(keep_checkpoint_max=5, session_config=session_config)
        bz_log.info("begin tf init")
        super().__init__(config=run_config,
                         model_dir=model_dir,
                         model_fn=self.__model_fn)
        bz_log.info("end tf init")
        self.__check_params()

    def network(self, features, is_training):
        raise NotImplementedError('请重写子类的network方法！')

    def get_loss(self, logits, labels):
        raise NotImplementedError('请重写子类的get_loss方法！')

    def export_model(self, export_model_dir, checkpoint_path):
        if os.path.exists(export_model_dir):
            shutil.rmtree(export_model_dir)
        os.mkdir(export_model_dir)

        return self.export_savedmodel(
            export_model_dir,
            serving_input_receiver_fn=self._serving_input_receiver_fn,
            checkpoint_path=checkpoint_path)

    def _regularizer(self, weights):
        """
        :param weights: 权重
        :return:
        """
        return tf.contrib.layers.l1_l2_regularizer(
            self.regularizer_scale[0], self.regularizer_scale[1])(weights)

    def _get_background_and_foreground_loss(self, logits, labels):
        """
        对前后背景的loss进行加权取得总的loss，前景的各个类别权重都为１
        :param logits: 预测结果
        :param labels: 标签
        :return:
        """
        if len(self.background_and_foreground_loss_weight) != 2:
            bz_log.error('classification_weight的长度必须是2！%d', len(self.background_and_foreground_loss_weight))
            raise ValueError('classification_weight的长度必须是2！')
        # 背景
        background_indices = tf.squeeze(
            tf.where(tf.equal(labels, 0)), 1)
        background_labels = tf.gather(labels, background_indices)
        background_logits = tf.gather(logits, background_indices)

        # 前景
        foreground_indices = tf.squeeze(
            tf.where(tf.greater(labels, 0)), 1)
        foreground_labels = tf.gather(labels, foreground_indices)
        foreground_logits = tf.gather(logits, foreground_indices)

        # 背景loss
        background_loss = tf.losses.sparse_softmax_cross_entropy(
            background_labels,
            background_logits,
            reduction=tf.losses.Reduction.MEAN)
        # 前景loss
        foreground_loss = tf.losses.sparse_softmax_cross_entropy(
            foreground_labels,
            foreground_logits,
            weights=1,
            reduction=tf.losses.Reduction.MEAN)

        tf.identity(background_loss, name='background_loss')
        tf.summary.scalar('background_loss', background_loss)
        tf.identity(foreground_loss, name='foreground_loss')
        tf.summary.scalar('foreground_loss', foreground_loss)
        return self.background_and_foreground_loss_weight[0] * background_loss \
            + self.background_and_foreground_loss_weight[1] * foreground_loss

    def _get_background_and_foreground_class_weight_loss(self, logits, labels):
        """
        对前后背景的loss进行加权取得总的loss，前景的各个类别权重都为class_loss_weight
        :param logits: 预测结果
        :param labels: 标签
        :return:
        """
        if len(self.background_and_foreground_loss_weight) != 2:
            bz_log.error('classification_weight的长度必须是2！%d', len(self.background_and_foreground_loss_weight))
            raise ValueError('classification_weight的长度必须是2！')
        # 背景
        background_indices = tf.squeeze(
            tf.where(tf.equal(labels, 0)), 1)
        background_labels = tf.gather(labels, background_indices)
        background_logits = tf.gather(logits, background_indices)

        # 前景
        foreground_indices = tf.squeeze(
            tf.where(tf.greater(labels, 0)), 1)
        foreground_labels = tf.gather(labels, foreground_indices)
        foreground_logits = tf.gather(logits, foreground_indices)

        # 背景loss
        background_loss = tf.losses.sparse_softmax_cross_entropy(
            background_labels,
            background_logits,
            reduction=tf.losses.Reduction.MEAN)
        # 前景loss
        weights = self.__cal_loss_weight(foreground_labels, self.class_loss_weight)
        foreground_loss = tf.losses.sparse_softmax_cross_entropy(
            foreground_labels,
            foreground_logits,
            weights=weights,
            reduction=tf.losses.Reduction.MEAN)

        tf.identity(background_loss, name='background_loss')
        tf.summary.scalar('background_loss', background_loss)
        tf.identity(foreground_loss, name='foreground_loss')
        tf.summary.scalar('foreground_loss', foreground_loss)
        return self.background_and_foreground_loss_weight[0] * background_loss \
            + self.background_and_foreground_loss_weight[1] * foreground_loss

    def _get_class_weighted_loss(self, logits, labels):
        """
        :param logits: 预测结果
        :param labels: 标签
        :param loss_weight: loss权重
        :return:
        """
        weights = self.__cal_loss_weight(labels, self.class_loss_weight)
        class_weight_loss = tf.losses.sparse_softmax_cross_entropy(
            labels,
            logits,
            weights=weights,
            reduction=tf.losses.Reduction.MEAN)

        tf.identity(class_weight_loss, name='class_weight_loss')
        tf.summary.scalar('class_weight_loss', class_weight_loss)
        return class_weight_loss

    def _serving_input_receiver_fn(self):
        raise NotImplementedError('请重写子类的_serving_input_receiver_fn方法！')

    def __check_params(self):
        if len(self.regularizer_scale) != 2:
            bz_log.error('输入的正则项系数长度必须是2！%d', len(self.regularizer_scale))
            raise ValueError('输入的正则项系数长度必须是2！')
        if self.regularizer_scale[0] < 0 or self.regularizer_scale[1] < 0:
            bz_log.error('输入的正则项系数必须大于0！%f', self.regularizer_scale)
            raise ValueError('输入的正则项系数必须大于0！')
        if self.learning_rate <= 0 or self.learning_rate >= 1:
            bz_log.error('输入的学习率必须大于0且小于1！%f', self.learning_rate)
            raise ValueError('输入的学习率必须大于0且小于1！')

    def __generate_estimator_spec(self, features, logits, labels, mode):
        """
        :param logits: 预测结果
        :param labels: 标签
        :param mode: mode方式
        :return:
        """
        # logits维度判断
        if len(logits.shape) != 4:
            bz_log.error('logits维度必须是4维！%d', len(logits.shape))
            raise ValueError('logits维度必须是4维！')
        softmax = tf.nn.softmax(logits, axis=-1, name='softmax')
        classes = tf.cast(
            tf.argmax(softmax, axis=-1), dtype=tf.uint8, name='classes')
        predictions = {
            'classes': classes,
            'probablities': softmax}
        rgb_predictions = tensorflow_image_processing.\
            convert_class_label_to_rgb_label_pyfunc(
                tf.expand_dims(classes, axis=3), class_num=self.class_num)
        logits_reshape = tf.reshape(logits, [-1, self.class_num])

        loss = None
        train_op = None
        eval_metric_ops = None
        export_outputs = None
        if mode == tf.estimator.ModeKeys.TRAIN \
                or mode == tf.estimator.ModeKeys.EVAL:
            labels = tf.cast(labels, dtype=tf.int32)
            rgb_label = tensorflow_image_processing. \
                convert_class_label_to_rgb_label_pyfunc(
                    labels, class_num=self.class_num)
            if features.shape[-1] == 1:
                features = tf.concat([features, features, features], axis=-1)
            labels_reshape = tf.reshape(labels, [-1, ])
            predictions_reshape = tf.reshape(predictions['classes'], [-1, ])

            # 正则项loss
            regularization_loss = tf.losses.get_regularization_loss()
            loss = self.get_loss(logits_reshape, labels_reshape) + \
                regularization_loss
            tf.identity(regularization_loss, name='regularization_loss')
            tf.summary.scalar('regularization_loss', regularization_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.image(
                'concat_result_train',
                tf.concat(
                    [tf.cast(features, dtype=tf.uint8), rgb_predictions[0],
                     rgb_label[0]], axis=2),
                max_outputs=self.max_img_outputs)
            assessment_dict = tensorflow_accuracy.get_assessment_result(
                labels_reshape, predictions_reshape,
                self.class_num, 'train', self.assessment_list)
            optimizer = self.optimizer_fn(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss, global_step=tf.train.get_global_step())
            for key, value in assessment_dict.items():
                tf.identity(value[1], name=key)
                tf.summary.scalar(key, value[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = tensorflow_accuracy.get_assessment_result(
                labels_reshape, predictions_reshape,
                self.class_num, 'evaluate', self.assessment_list)
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {'result': tf.estimator.export.PredictOutput(
                predictions['classes'])}
        logging_hook = tf.train.LoggingTensorHook(
            self.tensor_to_log,
            every_n_iter=50)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          train_op=train_op,
                                          eval_metric_ops=eval_metric_ops,
                                          export_outputs=export_outputs,
                                          training_hooks=[logging_hook],
                                          evaluation_hooks=[logging_hook])

    def __model_fn(self, features, mode):
        """
        :param features: 图像与标签的字典
        :param mode: mode方式
        :return:
        """
        input_layer = features['img']
        if mode == tf.estimator.ModeKeys.PREDICT:
            labels = None
        else:
            labels = features['label']
        logits = self.network(input_layer, mode == tf.estimator.ModeKeys.TRAIN)
        return self.__generate_estimator_spec(input_layer, logits, labels, mode)

    def __cal_loss_weight(self, labels, weights):
        """
        :param labels: 标签
        :param weight: 每一个类别的权重
        :return:
        """
        label_shape = tf.shape(labels)
        weight = tf.zeros(label_shape)
        loss_weight = tf.zeros(label_shape)
        for i in range(len(weights)):
            classes_loss_weight = tf.where(
                tf.equal(labels, i),
                tf.multiply(tf.ones(label_shape), weights[i]),
                weight)
            loss_weight = tf.add_n([loss_weight, classes_loss_weight])
        return loss_weight
