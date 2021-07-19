import os
import shutil
import numpy as np
import tensorflow as tf
from diabetic_package.model_training.estimator.LW_classification import dataset
import random
from diabetic_package.model_training.estimator.LW_classification.simple_cnn import CNNEstimator
from diabetic_package.machine_learning_common import convert_to_pb
from diabetic_package.file_operator import bz_path



tf.logging.set_verbosity(tf.logging.INFO)

class  NET_train():

    def __init__(self,
                 model_dir_base,
                 class_num,
                 img_size = (64,64),
                 channel_num = 3,
                 batch_size=20,
                 epoch_num = 500,
                 # regularizer_scale = 0.0,
                 # Is_evalute = True,
                 ext=['.jpg', 'txt']
                 ):
        '''
            model_dir_base:   模型保存的根目录
            creat_way:        训练数据混合的方式,为1则表示正负样本总数按照一定比例混合,
                              为0则表示每个batch中正负样本按照一定比例混合
            sample_proportion:正负样本的混合比例,当creat_way为1时,输入的为大
                              于等于1的整数倍,当creat_way为0时输入为0到1的小数
        '''
        self.img_width = img_size[0]
        self.img_height = img_size[1]
        self.channel_num = channel_num
        self.class_num = class_num
        self.ext = ext
        self.model_dir_base = model_dir_base
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        # self.regularizer_scale = regularizer_scale
        # self.Is_evalute = Is_evalute

        self.best_checkpoint_dir = (
            self.model_dir_base + '/best_checkpoint/')
        if not os.path.exists(self.best_checkpoint_dir):
            os.makedirs(self.best_checkpoint_dir)
        self.cnn_model_dir = (
                self.model_dir_base )
        self.export_model_dir = (
                self.model_dir_base + '/export_model_dir/')
        if not os.path.exists(self.best_checkpoint_dir):
            os.makedirs(self.cnn_model_dir)

        self.estimator = CNNEstimator(
            model_dir=self.cnn_model_dir,
            best_checkpoint_dir=self.best_checkpoint_dir,
            img_shape=[self.img_height, self.img_width, self.channel_num],
            class_num=self.class_num)
    def predict(self, patch_set, best_checkpoint):
        predictions = self.estimator.predict(patch_set,
                                             checkpoint_path=best_checkpoint)
        return np.array([prediction for prediction in predictions])

    def predict_proba(self, patch_set, checkpoint_path):
        predictions = self.estimator.predict_proba(patch_set,
                                                   checkpoint_path=checkpoint_path)
        return np.array(
            [prediction for prediction in predictions])

    def _train(self, img_set_file, label_set):
        # self.save_hyperparameter()
        best_accuracy = 0
        accuracy_tolerance = 0.0005
        accuracy_not_increase_epoch_num = 0
        accuracy = 0
        for i in range(self.epoch_num):
            print('epoch num:', i)
            self.estimator.train(
                input_fn=lambda: self.__train_input_fn(
                    img_set_file,
                    label_set))
            # self.best_checkpoint = self.estimator.best_checkpoint
            # self.__export_model()
            print('验证集验证：')
            eval_metric = self.estimator.evaluate(
                        input_fn=self._evalute_input_fn)
            print(eval_metric)
            if eval_metric['accuracy'] > accuracy:
                accuracy = eval_metric['accuracy']
                print(accuracy)
                self.estimator.save_best_checkpoint()
                self.best_checkpoint = self.estimator.best_checkpoint
                print(self.best_checkpoint)
        # self.__export_model()
                self.best_checkpoint = self.estimator.best_checkpoint
                self.__export_model()
            if eval_metric['accuracy'] - best_accuracy <= accuracy_tolerance:
                accuracy_not_increase_epoch_num += 1
            else:
                best_accuracy = eval_metric['accuracy']
                # self.estimator.save_best_checkpoint()
                self.best_checkpoint = self.estimator.best_checkpoint
                self.__export_model()
                accuracy_not_increase_epoch_num = 0

            if accuracy_not_increase_epoch_num > 20:
                print('early stopping 共计训练%d个epoch' % i)
                print("best_accuracy:", best_accuracy)
                print(self.estimator.latest_checkpoint())
                break
    def __export_model(self):
        if os.path.exists(self.export_model_dir):
            shutil.rmtree(self.export_model_dir)
        os.mkdir(self.export_model_dir)

        self.estimator.export_savedmodel(
            self.export_model_dir,
            self.estimator.serving_input_receiver_fn,
            checkpoint_path=self.best_checkpoint)

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
            f.write('input_height:' + str(self.img_height) + '\n')
            f.write('input_width:' + str(self.img_width) + '\n')
            f.write('input_channel:' + str(self.channel_num) + '\n')
            f.write('batch_size:' + str(self.batch_size) + '\n')
            f.write('class_num:' + str(self.class_num))

    def __decode_img(self, features, label):
        img_path = features['img']

        encode_dir = tf.read_file(img_path)
        # if self.ext[0] == ".jpg":
        #     img = tf.image.decode_jpeg(encode_dir)
        # elif self.ext[0] == ".png":
        img = tf.image.decode_png(encode_dir)
        # elif self.ext[0] == ".bmp":
        #     img = tf.image.decode_bmp(encode_dir)
        img = tf.image.resize_images(img, [self.img_height, self.img_width])
        img = tf.cast(img, dtype=tf.float32)
        return ({'img': img}, label)

    def __tensor_slices_to_dataset(self, data_tensor):
        datasets = tf.data.Dataset.from_tensor_slices(data_tensor)
        dataset = datasets.map(
            self.__decode_img,
            num_parallel_calls=20)
        return dataset

    def __create_dataset(self, feature_set, label_set):
        train_data_tensor = ({'img': feature_set}, label_set)
        datasets = tf.data.Dataset.from_tensor_slices(train_data_tensor)
        dataset = datasets.map(
            self.__decode_img,
            num_parallel_calls=20)
        return dataset

    def __train_input_fn(self, feature_set, label_set):
        dataset = self.__create_dataset(feature_set, label_set)
        dataset = dataset.map(self.__img_mirroring)
        # dataset = dataset.map(self.__img_random_crop)

        # random_num = tf.random_uniform([1], 0, 2)
        # if tf.equal(random_num, 0):
        #     dataset = dataset.map(self.__img_mirroring)
        # else:
        #     dataset = dataset.map(self.__img_random_crop)
        #
        dataset = dataset.shuffle(len(feature_set)).batch(
            self.batch_size,drop_remainder=True)

        return dataset

    def __create_eval_dataset(self):
        shuffle_inds = np.arange(len(self.eval_set))
        random.shuffle(shuffle_inds)
        feature = self.eval_set[shuffle_inds]
        label = self.eval_label_set[shuffle_inds]
        eval_data_tensor = ({'img': feature}, label)
        self.evalute_set = self.__tensor_slices_to_dataset(eval_data_tensor)

    def evalute(self, img_set_file, label_set):
        self.eval_set = img_set_file
        self.eval_label_set = label_set
        eval_metric = self.estimator.evaluate(
            input_fn=self._evalute_input_fn)
        print(eval_metric)

    def _evalute_input_fn(self):
        self.__create_eval_dataset()
        dataset = self.evalute_set.batch(self.batch_size)
        return dataset

    def _read_resize_img(self, img_path, label):
        img_string = tf.read_file(img_path['img'])
        img_decoded = tf.image.decode_jpeg(img_string)
        image_resized = tf.image.resize_images(
            img_decoded, [self.img_height, self.img_width])
        return ({'img': image_resized}, label)

    def train_cnn_classifier(self, train_set_dir, eval_set_dir):
        print('开始训练...')
        feature_set, label_set = dataset.create_data(train_set_dir, self.ext)
        self.eval_set, self.eval_label_set = dataset.create_data(eval_set_dir, self.ext)
        # self.classes_ = np.array([0, 1])  # 我们处理的是二分类问题
        feature_set = feature_set.reshape(-1)
        self._train(
            feature_set, label_set)

        print('训练结束!')

    def __img_mirroring(self, features, labels):
        distort_random = \
            tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
        mirror = tf.less(distort_random, 0.5)
        mirror_mask = tf.boolean_mask([0, 1], mirror)
        img_mirror = tf.reverse(features['img'], mirror_mask)
        return {'img': img_mirror}, labels

    def __img_random_crop(self, features, labels, min_scale=0.9, max_scale=1.0):
        if min_scale <= 0:
            raise ValueError('min_scale必须大于0！')
        if max_scale <= 0:
            raise ValueError('max_scale必须大于0！')
        if min_scale > max_scale:
            raise ValueError('min_scale必须小于max_scale!')
        shape = tf.to_float(tf.shape(features['img']))
        scale = tf.random_uniform(
            [1], min_scale, max_scale, dtype=tf.float32)[0]
        new_height = tf.to_int32(shape[0] * scale)
        new_width = tf.to_int32(shape[1] * scale)
        img_crop = tf.random_crop(features['img'], [new_height, new_width, 1])
        img_resize = tf.image.resize_images(
            img_crop, [self.img_height, self.img_width])
        return {'img': img_resize}, labels


if __name__ == '__main__':
    root_model_dir = './model_dir_手机_nin_cnn/'
    train_set_dir = './data/手机_jpg/train/'
    eval_set_dir = './data/手机_jpg/test/'
    class_num = 6
    ext='.jpg'
    print('开始训练')
    cnn_classify = NET_train(root_model_dir, class_num,ext=ext)
    # train
    cnn_classify.train_cnn_classifier(train_set_dir,eval_set_dir)
    # test
    # eval_set, eval_label_set = dataset.create_data(eval_set_dir, ext)
    # cnn_classify.evalute(eval_set, eval_label_set)
    #
    #
    # from diabetic_package.file_operator import bz_path
    #
    # best_checkpoint = './model_dir_cat_dog/best_checkpoint/model.ckpt-16940'
    # predict_img_list = np.array(sorted(
    #     bz_path.get_file_path(eval_set_dir, ret_full_path=True)))
    # predictions = cnn_classify.predict(predict_img_list, best_checkpoint)
    # for predict_img_path, prediction in zip(predict_img_list, predictions):
    #     img_name = os.path.split(predict_img_path)[1]
    #     file_name= os.path.split(predict_img_path)[0]
    #     label_falg = os.path.split(file_name)[-1]
    #     # print(label_falg)
    #     shutil.copy(predict_img_path, './result_cat_dog/' + str(prediction) + '/' + label_falg + "_" + str(prediction) + "_" + img_name)
