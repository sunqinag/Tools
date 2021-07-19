# -----------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要：
#   当前版本: 0.0
#   作   者：陈瑞侠
#   完成日期: 2020
# -----------------------------------------------------------------
import shutil
import os
import tensorflow as tf
import numpy as np

from diabetic_package.file_operator import bz_path
class ClassifyEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir,
                 best_checkpoint_dir,
                 img_shape,
                 class_num,
                 l1_scale=0.0,
                 l2_scale=0.0,
                 label_smoothing=0
                 ):
        '''
            model_dir: 存储checkpoint的路径
            best_checkpoint_dir: 存储结果最好的checkpoint的路径
            img_shape: 输入图像的尺寸(height, width, channel)
            l1_scale: l1正则项系数
            l2_scale: l2正则项系数
            label_smoothing:使用soft label时,统计的标签存在不准确的概率
        '''
        self.model_path = model_dir
        self.best_checkpoint_dir = best_checkpoint_dir
        self.img_height = img_shape[0]
        self.img_width = img_shape[1]
        self.channel_num = img_shape[2]
        self.class_num = class_num
        self.l1_scale = l1_scale
        self.l2_scale = l2_scale
        self.label_smoothing = label_smoothing
        run_config = tf.estimator.RunConfig(keep_checkpoint_max=5)
        super(ClassifyEstimator, self).__init__(
            model_fn=self.__model_fn,
            model_dir=self.model_path,
            config=run_config)

    def serving_input_receiver_fn(self):
        img = tf.placeholder(dtype=tf.float32,
                             shape=[None,
                                    self.img_height,
                                    self.img_width,
                                    self.channel_num],
                             name='input_img')
        img = tf.reshape(img, shape=[-1,
                                     self.img_height,
                                     self.img_width,
                                     self.channel_num])
        features = {'img': img}
        return tf.estimator.export.ServingInputReceiver(
            features, features)

    def regularizer(self, weights):
        return tf.contrib.layers.l1_l2_regularizer(
            self.l1_scale, self.l2_scale)(weights)

    def train(self,
              input_fn,
              hooks=None,
              steps=None,
              max_steps=None,
              saving_listeners=None):
        super().train(
            input_fn=input_fn)
        # self.save_best_checkpoint()

    def predict(self,
                input_fn,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None,
                yield_single_example=True):
        predictions = super().predict(
            input_fn=lambda: self.__get_single_img_dataset(input_fn),
            checkpoint_path=checkpoint_path)
        return np.array([prediction['classes'] for prediction in predictions])

    def predict_proba(self, input_fn,
                      predict_keys=None,
                      hooks=None,
                      checkpoint_path=None,
                      yield_single_example=True):
        predictions = super().predict(
            input_fn=lambda: self.__get_single_img_dataset(input_fn),
            checkpoint_path=checkpoint_path)
        return np.array(
            [prediction['probabilities'] for prediction in predictions])

    def save_best_checkpoint(self):
        if (os.path.exists(self.best_checkpoint_dir)):
            shutil.rmtree(self.best_checkpoint_dir)
        os.mkdir(self.best_checkpoint_dir)
        checkpoint_files = bz_path.get_file_path(
            self.model_dir, ret_full_path=True)
        (_, checkpoint_name) = os.path.split(
            super().latest_checkpoint())
        for path in checkpoint_files:
            if checkpoint_name in path:
                shutil.copy(path, self.best_checkpoint_dir)
        self.best_checkpoint = self.best_checkpoint_dir + checkpoint_name

    def __get_single_img_dataset(self, patch_set):
        patch_set_file = patch_set.reshape(-1)
        dataset = tf.data.Dataset.from_tensor_slices(patch_set_file)
        dataset = dataset.map(self.__read_and_resize_img)
        return dataset

    def __read_and_resize_img(self, img):
        img_string = tf.read_file(img)
        img_decoded = tf.image.decode_jpeg(img_string)
        image_resized = tf.image.resize_images(
                img_decoded, [self.img_height, self.img_width])
        return ({'img': image_resized})

    def _get_weights(self, label):
        weights = label

        weights = tf.where(tf.equal(weights, 0), 8.1166 * tf.ones_like(weights),
                           weights)
        weights = tf.where(tf.equal(weights, 1), 5.3516 * tf.ones_like(weights),
                           weights)
        # weights = tf.where(tf.equal(weights, 2), 12.6493 * tf.ones_like(weights),
        #                    weights)
        # weights = tf.where(tf.equal(weights, 3), 2.6539 * tf.ones_like(weights),
        #                    weights)
        # weights = tf.where(tf.equal(weights, 4), 4.2719 * tf.ones_like(weights),
        #                    weights)
        return weights

    def __generate_estimator_spec(self, mode, logits, label):
        softmax = tf.nn.softmax(logits, axis=-1, name='softmax')
        predictions = {'classes': tf.argmax(logits, axis=1, name='classes'),
                           'probabilities': softmax}
        if (mode == tf.estimator.ModeKeys.TRAIN or
                    mode == tf.estimator.ModeKeys.EVAL):
            penalty = tf.losses.get_regularization_loss()
            # weigths = self._get_weights(label)

            labels = tf.cast(label, tf.int32)
            labels = tf.one_hot(labels, depth=self.class_num, on_value=1, off_value=0)
            loss = tf.reduce_mean(
                        tf.losses.softmax_cross_entropy(
                            labels, logits,
                            # weights = weigths,
                            label_smoothing=self.label_smoothing)) + penalty

        else:
            loss = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step())
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op, update_ops])

            train_accuracy = tf.metrics.accuracy(
                labels=label, predictions=predictions['classes'])[1]
            tf.summary.scalar('train_accuracy', train_accuracy)
        else:
            train_op = None
        if mode == tf.estimator.ModeKeys.EVAL:
            # indexs = tf.squeeze(tf.where(tf.equal(label, self.class_num)), axis=1)
            # scrash_label = tf.gather(label, indexs)
            # scrash_predictions = tf.gather(predictions['classes'], indexs)

            # indexs1 = tf.squeeze(tf.where(tf.equal(label, self.class_num)), axis=1)
            # normal_label = tf.gather(label, indexs1)
            # normal_predictions = tf.gather(predictions['classes'], indexs1)

            eval_metric_ops = {
                    'accuracy': tf.metrics.accuracy(
                        labels=label, predictions=predictions['classes']),
                    # 'scrash_accuracy':tf.metrics.accuracy(
                    #     labels=scrash_label,
                    #     predictions=scrash_predictions),
                    # 'normal_accuracy': tf.metrics.accuracy(
                    # labels=normal_label,
                    # predictions=normal_predictions)
            }
        else:
            eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {'result': tf.estimator.export.PredictOutput(
                    predictions['classes'])}
        else:
            export_outputs = None

        tensors_to_log = {'probabilities': 'softmax'}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=50)

        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops,
                training_hooks= [logging_hook],
                export_outputs=export_outputs)

    def __model_fn(self, features, labels, mode):
        input_layer = features['img']
        logits = self.network(
            input_layer, mode == tf.estimator.ModeKeys.TRAIN)

        return self.__generate_estimator_spec(mode, logits, labels)