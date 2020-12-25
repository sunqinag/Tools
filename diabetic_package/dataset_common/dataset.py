# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：dataset.py
#   摘   要：创建分类分割在训练验证和测试时的dataset
#   当前版本:2019121916
#   作   者：崔宗会，陈瑞侠
#   完成日期：2019-12-19
# -----------------------------
import tensorflow as tf
import numpy as np


from ..image_processing_operator import tensorflow_image_processing_map, tensorflow_image_processing


class ImageLabelDataSetBaseClass:
    def __init__(self, img_list, label_list, channels_list, file_extension_list, height_width, crop_height_width=(400, 400),
                 batch_size=1, num_epochs=1, shuffle=False, mode='train', task='classification'):
        """
        dataset基类
        :param img_list: 图像路径列表
        :param label_list: 标签list,使用ndarray或者None
        :param channels_list: 解码图像的通道数如[3,1]
        :param file_extension_list: 解码图像的格式,目前支持bmp,jpg,jpeg,png,gif
        :param height_width: 进行resize后的高和宽
        :param crop_height_width: 进行图像crop的高宽
        :param batch_size: batch的大小
        :param num_epochs: epoch数
        :param shuffle: 是否shuffle
        :param mode: 设定dataset模式,包含'train','evaluate','predict'三种，也可填入数字１，２，３分别与三种模式对应
        :param task:设定任务类型，包含'classification','segmentation',也可填入数字1,2与两种任务对应

        """

        self.img_path_list, self.label_list = img_list, label_list
        self.height_width = height_width
        self.crop_height_width = crop_height_width

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.channels_list = channels_list
        self.file_extension_list = file_extension_list
        self.mode = mode
        self.task = task

        self._check_params()

    # def create_dataset(self):
    #     # 构造数据queue
    #
    #     if self.mode == 'predict':
    #         self.label_list = np.zeros_like(self.img_path_list)
    #     dataset_src = ({'img_list': self.img_path_list}, {'label_list': self.label_list})
    #     dataset = tf.data.Dataset.from_tensor_slices(dataset_src)
    #     dataset_map = dataset.map(self._read_img_label,num_parallel_calls=4)
    #     dataset_map = dataset_map.map(self._preprocess_image_map_outer,num_parallel_calls=4)
    #     if self.shuffle:
    #         dataset = dataset_map.batch(
    #             self.batch_size, drop_remainder=False).shuffle(
    #             buffer_size=len(self.img_path_list)).repeat(self.num_epochs)
    #     else:
    #         dataset = dataset_map.batch(
    #             self.batch_size).repeat(self.num_epochs)
    #
    #     dataset = dataset.cache()
    #     dataset = dataset.prefetch(buffer_size=None)
    #
    #     return dataset

    def create_dataset(self):
        # 构造数据queue

        if self.mode == 'predict':
            self.label_list = np.zeros_like(self.img_path_list)
        dataset_src = ({'img_list': self.img_path_list}, {'label_list': self.label_list})
        dataset = tf.data.Dataset.from_tensor_slices(dataset_src)
        dataset_map = dataset.map(self._read_img_label, num_parallel_calls=4)
        dataset_map = dataset_map.map(self._preprocess_image_map_outer, num_parallel_calls=4)
        if self.shuffle:
            dataset = dataset_map.batch(
                self.batch_size, drop_remainder=False).shuffle(
                buffer_size=len(self.img_path_list)).repeat(self.num_epochs)
        else:
            dataset = dataset_map.batch(
                self.batch_size).repeat(self.num_epochs)
        dataset = dataset.prefetch(buffer_size=10 * self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_data = iterator.get_next()
        if self.mode == 'predict':
            batch_data = {'img': batch_data['img']}
        return batch_data


    def preprocess_image_map(self, data_dict):
        """
        图像处理的map函数
        :param data_dict:传入的是{'img':image,'label':label}字典
        :param is_classification:是否分类任务
        :return:返回经过图像处理后的字典
        """
        if self.mode == 'train':
            raise NotImplementedError('train 模式必须在子类重写该方法!')
        return data_dict

    def _preprocess_image_map_outer(self, data_dict):
        image = data_dict['img']
        label = data_dict['label']
        image_map = tf.reshape(image, [-1, image.shape[0], image.shape[1], image.shape[2]])
        label_map = None

        if self.task == 'segmentation' and (self.mode == 'train' or self.mode == 'evaluate'):
            label_map = tf.reshape(label, [-1, label.shape[0], label.shape[1], label.shape[2]])
        if self.mode == 'train':
            image_map, label_map = self.preprocess_image_map({'img': image_map, 'label': label_map})

        image_map_type=image_map.dtype
        image_map = tf.image.resize_bilinear(image_map, size=[self.crop_height_width[0], self.crop_height_width[1]])
        image_map=tf.cast(image_map,image_map_type)
        image_map = tf.reshape(image_map,
                               shape=[self.crop_height_width[0], self.crop_height_width[1], self.channels_list[0]])


        if self.mode == 'predict':
            return {'img': image_map}
        elif self.task == 'classification':
            return {'img': image_map, 'label': label}
        else:
            label_map_type = label_map.dtype
            label_map = tf.image.resize_nearest_neighbor(label_map,
                                                         size=[self.crop_height_width[0], self.crop_height_width[1]])
            label_map = tf.cast(label_map, label_map_type)
            label_map = tf.reshape(label_map,
                                   shape=[self.crop_height_width[0], self.crop_height_width[1], self.channels_list[1]])
            return {'img': image_map, 'label': label_map}

    def _read_img_label(self, img_path_list, labels):
        img_list = img_path_list['img_list']
        img = tensorflow_image_processing.read_image_and_resize_tf(
            img_list, self.channels_list[0], self.file_extension_list[0],
            height=self.height_width[0], width=self.height_width[1], method=tf.image.ResizeMethod.BILINEAR)
        if self.task == 'segmentation' and (self.mode == 'train' or self.mode == 'evaluate'):
            label_list = labels['label_list']
            label = tensorflow_image_processing.read_image_and_resize_tf(
                label_list, self.channels_list[1], self.file_extension_list[1],
                height=self.height_width[0], width=self.height_width[1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return {'img': img, 'label': label}
        else:
            return {'img': img, 'label': labels['label_list']}

    def _check_params(self):
        mode_tuple = ('train', 'evaluate', 'predict')
        if isinstance(self.mode, str):
            if self.mode.lower() not in mode_tuple:
                raise ValueError("mode　必须为'train','evaluate','predict','TRAIN','EVALUATE','PREDICT',或者1,2,3！")
            self.mode = self.mode.lower()
        if isinstance(self.mode, int):
            if self.mode not in (1, 2, 3):
                raise ValueError("mode　必须为'train','evaluate','predict','TRAIN','EVALUATE','PREDICT',或者1,2,3！")
            self.mode = mode_tuple[self.mode - 1]

        task_tuple = ('classification', 'segmentation')
        if isinstance(self.task, str):
            if self.task.lower() not in task_tuple:
                raise ValueError("task　必须为'classification','segmentation','CLASSIFICATION','SEGMENTATION',或者1,2！")
            self.task = self.task.lower()

        if isinstance(self.task, int):
            if self.task not in (1, 2):
                raise ValueError("task　必须为'classification','segmentation','CLASSIFICATION','SEGMENTATION',或者1,2！")
            self.task = task_tuple[self.task - 1]

        if len(self.img_path_list) == 0:
            raise ValueError('路径' + self.img_path_list + '中图像不存在!')

        if (self.mode == 'train' or self.mode == 'evaluate'):
            if (not isinstance(self.img_path_list, np.ndarray) and self.img_path_list is None) or \
                    (not isinstance(self.label_list, np.ndarray) and self.label_list is None):
                raise ValueError('train和evaluate模式必须同时包含img和label!')
            elif len(self.img_path_list) != len(self.label_list):
                raise ValueError('img和label数目不一致!')

        if self.task == 'segmentation' and (len(self.channels_list) != 2 or len(self.file_extension_list) != 2):
            raise ValueError("分割任务必须给定参数channels_list和参数img_format_list两个值!")

        if len(self.height_width) != 2 or len(self.crop_height_width) != 2:
            raise ValueError('height_width和crop_height_width必须包含两个值！')


class ImageLabelDataSet(ImageLabelDataSetBaseClass):
    pass

    def preprocess_image_map(self, data_dict):
        """
        图像处理的map函数
        :param data_dict:
        :param is_mirroring:
        :return:
        """
        image = data_dict['img']
        label = data_dict['label']
        # image, label = tensorflow_image_processing_map.random_flip_image_and_label_tf_map(image, label)
        # image, label = tensorflow_image_processing_map.random_rotate_image_and_label_tf_map(image, label, max_angle=30, rate=0.1)
        # image, label = tensorflow_image_processing_map.random_translate_image_and_label_tf_map(image, label, max_dx=30,
        #                                                                                        max_dy=30, rate=0.1)
        # image, label = tensorflow_image_processing_map.random_rescale_and_crop_image_and_label_tf_map(
        #     image, label, crop_width=self.crop_height_width[1], crop_height=self.crop_height_width[0], rate=0.5,
        #     min_scale=0.8, max_scale=1.2)
        image, label = tensorflow_image_processing_map.random_crop_or_pad_image_and_label_tf_map(
            image, label, self.crop_height_width[0], self.crop_height_width[1], rate=0.5)




        # image, label = tensorflow_image_processing_map.flip_up_down_image_and_label_tf_map(image, label, rate=0.1)
        # image, label = tensorflow_image_processing_map.flip_left_right_image_and_label_tf_map(image, label, rate=0.1)
        # image, label = tensorflow_image_processing_map.transpose_image_and_label_tf_map(image, label, rate=0.1)
        # image, label = tensorflow_image_processing_map.translate_image_and_label_tf_map(image, label, dx=10, dy=-60.0, rate=0.1)
        #
        # image, label = tensorflow_image_processing_map.rotate_image_and_label_tf_map(image, label, angle=30, rate=0.1)
        # # #
        # image = tensorflow_image_processing_map.adjust_random_brightness_image_tf_map(image, 0.5, rate=0.1)
        # image = tensorflow_image_processing_map.adjust_random_contrast_image_tf_map(image, 0.8, 1.2, rate=0.1)
        # image = tensorflow_image_processing_map.adjust_random_hue_image_tf_map(image, 0.2, rate=0.1)
        # image = tensorflow_image_processing_map.adjust_random_saturation_image_tf_map(image, 0.0, 0.1, rate=0.1)
        # image = tensorflow_image_processing_map.add_random_noise_tf_map(image, 0.5, 10, rate=0.1)
        # image = tensorflow_image_processing_map.add_gaussian_noise_tf_map(image, 0.1, 1.0, 1.0)
        #
        # image = tensorflow_image_processing_map.add_salt_and_pepper_noise_pyfunc_tf_map(image, 0.002, 255, rate=0.1)
        # image = tensorflow_image_processing_map.adjust_brightness_image_tf_map(image, delta=100, rate=0.1)
        # image = tensorflow_image_processing_map.adjust_contrast_image_tf_map(image, contrast_factor=0.5, rate=0.1)
        # image = tensorflow_image_processing_map.adjust_saturation_image_tf_map(image, 0.2, rate=0.1)
        # # image=tf.cast(image,tf.uint8)
        # # label=tf.cast(label,tf.uint8)
        return image, label
