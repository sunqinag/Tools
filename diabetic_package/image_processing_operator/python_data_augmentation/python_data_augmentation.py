# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：data augmentation
#   摘   要：对图像数据进行增强，包括随机旋转，随机裁剪，随机噪声，平移变换，图像反转,并保存增强后的图像和标签
#   当前版本:2019121918
#   作   者：王茜, 崔宗会
#   完成日期：2019-12-19

import numpy as np
import functools
import os
import shutil
import datetime
import filecmp
import random
from . import python_base_data_augmentation
from ...file_operator import bz_path
from ..python_image_processing import imread,imwrite
from diabetic_package.log.log import bz_log


def load_detection_txt_label(detection_txt_label_path):
    if not os.path.exists(detection_txt_label_path):
        bz_log.error("路径(" + detection_txt_label_path + ")对应txt文件不存在!")
        bz_log.error("传入的txt文件路径：", detection_txt_label_path)
        raise ValueError("路径(" + detection_txt_label_path + ")对应txt文件不存在!")
    _, ext = bz_path.get_file_name(detection_txt_label_path, True)
    if ext != 'txt':
        bz_log.error("路径(" + detection_txt_label_path + ")不是txt文件!")
        bz_log.error("错误的txt文件路径：",detection_txt_label_path)
        raise ValueError("路径(" + detection_txt_label_path + ")不是txt文件!")

    label = []
    file = open(detection_txt_label_path, 'r+')
    for line in file.readlines():
        line=list(map(float, line.strip().split(',')))
        label.append([line[1], line[0], line[3], line[2], line[4]]) #坐标顺序改变
    label = np.array(label)
    return label


def save_detection_txt_label(detection_txt_label_path, label):
    _, ext = bz_path.get_file_name(detection_txt_label_path, True)
    if ext != 'txt':
        bz_log.error("路径(" + detection_txt_label_path + ")不是txt文件!")
        bz_log.error("错误的txt文件路径：", detection_txt_label_path)
        raise ValueError("路径(" + detection_txt_label_path + ")不是txt文件!")

    if os.path.exists(detection_txt_label_path):
        os.remove(detection_txt_label_path)

    with open(detection_txt_label_path, 'w+') as f:
        for one_label in label:
            one_label_str = functools.reduce(lambda x, y: x + y, [str(i) + '_' for i in one_label])
            f.write(one_label_str[:-1] + '\n')


class DataAugmentation:
    def __init__(self,
                 img_list,
                 label_list,
                 augmentation_ratio,
                 generate_data_folder='./generate_data',
                 channel=3,
                 out_file_extension_list=['jpg', 'npy'],
                 task='classification', augmentation_split_ratio=4):
        """
        对图像和对应标签进行数据增强,若augmentation_ratio小于4,则进行一次数据增强,否则进行
        二次增强,会在原始图像相同的目录下创建两个文件夹augmentation_img和augmentation_label存储增强后的图像和label
        支持分类、分割、目标检测，分类输入的是img_path_list和npy或者txt的label_path_list,
        分割输入的是img_path_list和label_path_list
        目标检测输入的是img_path_list和和npy或者txt的label_path_list，数据为nX5，对于txt存储的标签每行代表一个目标框有５个值，用下划线隔开，依次为
        左上角点的行列右下角点的行列，以及类别
        :param img_list: 输入img_list
        :param label_list: 输入label_list
        :param augmentation_ratio: 增强图像倍数
        :param generate_data_folder:生成文件路径
        :param channel: 图像channel
        :param out_file_extension_list:输出文件格式
        :param task: 增强任务,只能是"classification", "segmentation"和"detection"
        :param augmentation_split_ratio:一次增强和二次增强分割点
        return: 返回增强后的图像和标签的list
        """

        self.img_list = img_list
        self.label_list = label_list
        self.generate_data_folder = generate_data_folder
        self.augmentation_img_list = np.array([])
        self.augmentation_label_list = np.array([])
        self.augmentation_ratio = augmentation_ratio
        self.channel = channel
        self.out_file_extension_list = out_file_extension_list
        self.augmentation_split_ratio = augmentation_split_ratio
        self.task = task.lower()

        self.__check_params()
        self.color_flag = 'RGB'
        self.is_enhance_image_only = False
        if self.channel == 1:
            self.color_flag = 'Gray'
        self.__init_augmentation_data_fn()

    def augment_data(self):
        self.__create_augmentation_data_dir()
        return self.augmentation()

    # def augment_data(self):
    #     # crx更改逻辑，只进行一次数据增强
    #     if self.is_repeat_data:
    #         return self._repeat_use_data()
    #     else:
    #         self.__create_augmentation_data_dir()
    #         # if self.augmentation_ratio == 1:
    #         #     return self._one_ratio_augmentation()
    #         # else:
    #         #     return self._multiply_ratio_augmentation()
    #         return self.augmentation()

    def augmentation(self):
        # crx只做一次增强
        bz_log.info("进行数据增强")
        for i in range(len(self.img_list)):
            img_name, img_extension = bz_path.get_file_name(self.img_list[i], True)
            for j in range(int(self.augmentation_ratio)):
                random = np.random.randint(0, 5)
                augment_name = list(self.augment_fn_dict.keys())[random]
                img_path, label = self.__create_data_fn(self.img_list[i],
                                                        self.label_list[i],
                                                        self.augment_fn_dict[augment_name],
                                                        img_name,
                                                        self.out_file_extension_list,
                                                        False)
                self.augmentation_img_list = np.append(self.augmentation_img_list, img_path)
                self.augmentation_label_list = np.append(self.augmentation_label_list, label)
        np.save(self.data_list_npy_path + '/img.npy', self.augmentation_img_list)
        if self.is_enhance_image_only:
            shutil.rmtree(self.augmentation_label_dir)
            return self.augmentation_img_list
        np.save(self.data_list_npy_path + '/label.npy', self.augmentation_label_list)

        print('数据增强完成！')
        return self.augmentation_img_list, self.augmentation_label_list

    def _one_ratio_augmentation(self):
        bz_log.info("进行一次数据增强")
        for i in range(len(self.img_list)):
            img_name, img_extension = bz_path.get_file_name(self.img_list[i], True)

            img_path, label = self.__create_data_fn(self.img_list[i],
                                                    self.label_list[i],
                                                    self.__copy,
                                                    img_name,
                                                    self.out_file_extension_list,
                                                    False)
            self.augmentation_img_list = np.append(self.augmentation_img_list, img_path)
            self.augmentation_label_list = np.append(self.augmentation_label_list, label)
        np.save(self.data_list_npy_path + '/img.npy', self.augmentation_img_list)
        if self.is_enhance_image_only:
            shutil.rmtree(self.augmentation_label_dir)
            return self.augmentation_img_list
        np.save(self.data_list_npy_path + '/label.npy', self.augmentation_label_list)

        print('数据增强完成！')
        return self.augmentation_img_list, self.augmentation_label_list

    def _multiply_ratio_augmentation(self):
        bz_log.info("进行多次数据增强")
        img_num = len(self.img_list)
        augment_mode = len(self.augment_fn_dict)
        first_augmentation_ratio = np.minimum(self.augmentation_ratio, self.augmentation_split_ratio)
        bz_log.info("获得第一次增强比率%f", first_augmentation_ratio)
        num = np.int32((first_augmentation_ratio - 1) * img_num / augment_mode)
        remainder = (first_augmentation_ratio - 1) * img_num % augment_mode
        num_list = np.ones(shape=augment_mode, dtype=np.int32) * num
        num_list[0] += remainder
        if self.augmentation_ratio <= self.augmentation_split_ratio:
            # 一次增强
            self.__first_augment_data(num_list)
        else:
            # 一次增强
            self.__first_augment_data(num_list)
            # 二次增强
            second_augmentation_num = np.int64((self.augmentation_ratio - self.augmentation_split_ratio) * img_num)

            self.__second_augment_data(second_augmentation_num)

        if self.is_enhance_image_only:
            self.augmentation_img_list = self._one_ratio_augmentation()
            print('数据增强完成！')
            return self.augmentation_img_list
        # 增加原始图像
        self.augmentation_img_list, self.augmentation_label_list = self._one_ratio_augmentation()

        print('数据增强完成！')
        return self.augmentation_img_list, self.augmentation_label_list

    def __init_augmentation_data_fn(self):
        self.augment_fn_dict = {
            'flip':
                python_base_data_augmentation.random_flip_image_and_label,
            'translate': functools.partial(
                python_base_data_augmentation.random_translation_image_and_label, max_dist=0.1),
            'rotate': functools.partial(
                python_base_data_augmentation.random_rotate_image_and_label, min_angle=-10, max_angle=10),
            'scale_crop': functools.partial(
                python_base_data_augmentation.random_scale_and_crop_image_and_label,
                min_scale=0.8,
                max_scale=1.2,
                crop_height_scale=0.8,
                crop_width_scale=0.8),
            'shear': functools.partial(
                python_base_data_augmentation.random_shear_image_and_label, min_scale=0.2, max_scale=-0.2)}
        # 去掉if，因为逻辑错误，没有标签不能按照分割图像进行处理
        if self.is_enhance_image_only:
            self.__create_data_fn = \
                self.__create_segmentation_data
        else:
            if self.task == 'classification':
                self.__create_data_fn = \
                    self.__create_classification_data
            elif self.task == 'segmentation':
                self.__create_data_fn = \
                    self.__create_segmentation_data
            elif self.task == 'detection':
                self.__create_data_fn = \
                    self.__create_detection_data

    def __check_params(self):
        # 去掉
        # if self.label_list is None:
        #     self.is_enhance_image_only = True
        #     self.label_list = self.img_list
        #     self.task = 'segmentation'
        #     self.out_file_extension_list[1] = self.out_file_extension_list[0]

        if (not isinstance(self.img_list, np.ndarray) or \
                not isinstance(self.label_list, np.ndarray)):
            bz_log.error('img_list和label_list类型必须是np.ndarray!')
            raise ValueError('img_list和label_list类型必须是np.ndarray!')
        if (len(self.img_list) != len(self.label_list)):
            bz_log.error('原始img和label数量必须相等!%d%d', len(self.img_list), len(self.label_list))
            raise ValueError('原始img和label数量必须相等!')
        if self.augmentation_ratio < 1:
            bz_log.error('增强后图像数量必须大于原始图像数量!%f', self.augmentation_ratio)
            raise ValueError('增强后图像数量必须大于原始图像数量!')
        if self.channel != 1 and self.channel != 3:
            bz_log.error('图像channel必须是1或者3!%d', self.channel)
            raise ValueError('图像channel必须是1或者3!')
        if self.task != 'classification' and self.task != 'segmentation' \
                and self.task != 'detection':
            bz_log.error('图像增强任务只能是分类/分割或目标检测!%s', self.task)
            raise ValueError('图像增强任务只能是分类/分割或目标检测!')
        if self.out_file_extension_list[0] not in ['bmp', 'jpg', 'jpeg', 'png']:
            bz_log.error('输出图像目前只支持bmp,jpg,jpeg,png 四种格式%s', self.out_file_extension_list[0])
            raise ValueError('输出图像目前只支持bmp,jpg,jpeg,png 四种格式！')

        if self.task == 'segmentation' and (self.out_file_extension_list[1] not in ['bmp', 'jpg', 'jpeg', 'png'] or \
                                            self.out_file_extension_list[0] not in ['bmp', 'jpg', 'jpeg', 'png']):
            bz_log.error('对于分割任务，输出图像目前只支持bmp,jpg,jpeg,png 四种格式！%s%s',self.out_file_extension_list[1], self.out_file_extension_list[0] )
            raise ValueError('对于分割任务，输出图像目前只支持bmp,jpg,jpeg,png 四种格式！')
        if (self.task == 'classification' or self.task == 'objection') and (
            self.out_file_extension_list[0] not in ['bmp', 'jpg', 'jpeg', 'png'] or \
                self.out_file_extension_list[1] not in ['npy', 'txt']):
            bz_log.error("对于分类或分割任务，out_file_extension_list[0]只能为['bmp', 'jpg', 'jpeg', 'png'] 的一种格式，"
                             "out_file_extension_list[1]只能为['npy', 'txt']中的一种格式！%s%s",self.out_file_extension_list[0], self.out_file_extension_list[1] )
            raise ValueError("对于分类或分割任务，out_file_extension_list[0]只能为['bmp', 'jpg', 'jpeg', 'png'] 的一种格式，"
                             "out_file_extension_list[1]只能为['npy', 'txt']中的一种格式！")
        self.label_save_function = np.save
        if self.out_file_extension_list[1] == 'txt' and self.task == 'detection':
            self.label_save_function = save_detection_txt_label
        elif self.out_file_extension_list[1] == 'txt' and self.task != 'detection':
            self.label_save_function = np.savetxt

        if not os.path.exists(self.generate_data_folder):
            os.makedirs(self.generate_data_folder)

        if os.path.exists(self.generate_data_folder + '/augmentation_information.txt'):
            os.rename(self.generate_data_folder + '/augmentation_information.txt',
                      self.generate_data_folder + '/augmentation_information_old.txt')
            self.__write_parameter_information()
            self.is_repeat_data = filecmp.cmp(self.generate_data_folder + '/augmentation_information.txt',
                                              self.generate_data_folder + '/augmentation_information_old.txt')
            os.remove(self.generate_data_folder + '/augmentation_information_old.txt')
        else:
            self.__write_parameter_information()
            self.is_repeat_data = False

    def __create_augmentation_data_dir(self):

        self.data_list_npy_path = self.generate_data_folder + '/augmentation_data_list_npy/'
        self.augmentation_img_dir = self.generate_data_folder + '/augmentation_img/'
        self.augmentation_label_dir = self.generate_data_folder + '/augmentation_label/'

        if os.path.exists(self.data_list_npy_path):
            shutil.rmtree(self.data_list_npy_path)
        os.mkdir(self.data_list_npy_path)

        if os.path.exists(self.augmentation_img_dir):
            shutil.rmtree(self.augmentation_img_dir)
        os.mkdir(self.augmentation_img_dir)

        if os.path.exists(self.augmentation_label_dir):
            shutil.rmtree(self.augmentation_label_dir)
        os.mkdir(self.augmentation_label_dir)
        bz_log.info("完成数据增强文件路径的创建")

    def __first_augment_data(self, augmentation_num_list):
        """
        :param augmentation_num_list: 一次增强图像数目list
        :return:
        """
        bz_log.info("开始第一次数据增强")
        replace = False
        if len(self.img_list) < np.max(augmentation_num_list):
            replace = True

        for i, (augment_name, augment_fn) in enumerate(
                self.augment_fn_dict.items()):
            num_index = np.random.choice(
                range(len(self.img_list)), augmentation_num_list[i], replace=replace)
            for j in range(augmentation_num_list[i]):
                img_path = self.img_list[num_index][j]
                label = self.label_list[num_index][j]

                origin_img_name, img_extension = bz_path.get_file_name(img_path, True)

                img_name = origin_img_name + '_' + augment_name + '_' + self.__get_timestamp()
                augmentation_img_path, augmentation_label = \
                    self.__create_data_fn(
                        img_path, label, augment_fn, img_name, self.out_file_extension_list, True)
                self.augmentation_img_list = np.append(self.augmentation_img_list, augmentation_img_path)
                self.augmentation_label_list = np.append(self.augmentation_label_list, augmentation_label)

        bz_log.info("完成第一次数据增强")
    def __second_augment_data(self, augmentation_num):
        """
        :param augmentation_num: 二次增强图像数目
        :return:返回一次增强后图像和标签的list
        """
        bz_log.info("开始进行第二次增强")
        num_index = np.random.choice(range(len(self.augmentation_img_list)),
                                     augmentation_num,
                                     replace=True)
        for i in range(augmentation_num):
            img_path = self.augmentation_img_list[num_index][i]
            label = self.augmentation_label_list[num_index][i]
            random = np.random.randint(0, 5)
            augment_name = list(self.augment_fn_dict.keys())[random]

            origin_img_name, img_extension = bz_path.get_file_name(img_path, True)
            img_name = origin_img_name + '_' + augment_name + '_' + self.__get_timestamp()
            augmentation_img_path, augmentation_label = self.__create_data_fn(
                img_path, label, self.augment_fn_dict[augment_name], img_name, self.out_file_extension_list, True)
            self.augmentation_img_list = np.append(
                self.augmentation_img_list, augmentation_img_path)
            self.augmentation_label_list = np.append(
                self.augmentation_label_list, augmentation_label)
        bz_log.info("完成第二次数据增强")
    def __create_segmentation_data(self,
                                   img_path,
                                   label_path,
                                   augment_fn,
                                   img_name,
                                   file_extension_list,
                                   is_adding_noise=False):
        """
        :param img_path: 分割单张图像的路径
        :param label_path: 分割单张label的路径
        :param augment_fn: 增强函数
        :param img_name: 增强后图像的名字
        :param file_extension_list:图像和label的后缀
        :param is_adding_noise: 是否给图像加噪声
        :return:分割增强的图像和标签
        """
        img=imread(img_path,self.color_flag)
        if is_adding_noise:
            img = self.__random_add_noise_to_img(img)
        label=imread(label_path,'Gray')
        rdstr = ''.join(random.sample(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'], 5))
        augmentation_img, augmentation_label = augment_fn(img, label)
        timesamp = self.__get_timestamp()
        augmentation_img_path = self.augmentation_img_dir + os.sep + img_name + rdstr + timesamp + '.' + file_extension_list[0]
        imwrite(augmentation_img_path,augmentation_img)
        augmentation_label_path = self.augmentation_label_dir + os.sep + img_name + rdstr + timesamp + '.' + file_extension_list[1]
        imwrite(augmentation_label_path, augmentation_label)
        return augmentation_img_path, augmentation_label_path

    def __create_classification_data(self,
                                     img_path,
                                     label_path,
                                     augment_fn,
                                     img_name,
                                     file_extension_list,
                                     is_adding_noise=False):
        """
        :param img_path: 分类单张图像的路径
        :param label: 分类标签
        :param augment_fn: 增强函数
        :param img_name: 增强后图像的名字
        :param file_extension_list:图像和label的后缀
        :param is_adding_noise: 是否给图像增加噪声
        :return:分类增强的图像和标签
        """
        img = imread(img_path, self.color_flag)
        if is_adding_noise:
            img = self.__random_add_noise_to_img(img)

        label_ext = bz_path.get_file_name(label_path, True)[1]
        if label_ext == 'txt':
            label = np.loadtxt(label_path)
        else:
            label = np.load(label_path)
        label = int(label)
        timesamp = self.__get_timestamp()

        rdstr = ''.join(random.sample(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'], 5))

        augmentation_img, augmentation_label = augment_fn(img, label)
        augmentation_img_path = self.augmentation_img_dir + os.sep + \
                                img_name + rdstr + timesamp + '.' + file_extension_list[0]
        imwrite(augmentation_img_path, augmentation_img)

        augmentation_label_path = self.augmentation_label_dir + os.sep + \
                                  img_name + rdstr + timesamp + '.' + file_extension_list[1]
        self.label_save_function(augmentation_label_path, np.array([augmentation_label]))

        return augmentation_img_path, augmentation_label_path

    def __create_detection_data(self,
                                img_path,
                                label_path,
                                augment_fn,
                                img_name,
                                file_extension_list,
                                is_adding_noise=False):
        """
        :param img_path: 分割单张图像的路径
        :param label_path: 分割单张label的路径
        :param augment_fn: 增强函数
        :param img_name: 增强后图像的名字
        :param file_extension_list:图像和label的后缀
        :param is_adding_noise: 是否给图像加噪声
        :return:目标检测增强的图像和标签
        """
        img=imread(img_path,self.color_flag)
        if is_adding_noise:
            img = self.__random_add_noise_to_img(img)
        label_ext = bz_path.get_file_name(label_path, True)[1]
        if label_ext == 'txt':
            label = load_detection_txt_label(label_path)
        else:
            label = np.load(label_path)
            #update by enfu.
            # label = load_detection_txt_label(label_path)

        timesamp = self.__get_timestamp()
        rdstr = ''.join(random.sample(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m',"n","o","p"], 5))
        augmentation_img, augmentation_label = augment_fn(img, label)
        augmentation_img_path = self.augmentation_img_dir + os.sep + \
            img_name + rdstr + timesamp + '.' + file_extension_list[0]

        imwrite(augmentation_img_path,augmentation_img)
        augmentation_label_path = self.augmentation_label_dir + os.sep + \
                                  img_name + rdstr + timesamp + '.' + file_extension_list[1]

        self.label_save_function(augmentation_label_path, augmentation_label)

        return augmentation_img_path, augmentation_label_path

    def __copy(self, img, label):
        return img, label

    def __random_add_noise_to_img(self, img):
        """
        :param img: 单张图像
        :return:返回增加噪声后的图像
        """
        noise_mode_list = [['gaussian'],
                           ['salt'],
                           ['s&p'],
                           ['localvar'],
                           ['speckle']]
        mean_var_list = [[0.0001, 0.0001],
                         [0.0001, 0.00005],
                         [0.0001, 0.0001],
                         [0.0001, 0.0001],
                         [0.0001, 0.0001]]
        random_noise = np.random.randint(0, 6)
        if random_noise == 5:
            return img
        else:
            img_noise = python_base_data_augmentation.add_noise(
                img.copy(),
                noise_mode_list[random_noise],
                mean=mean_var_list[random_noise][0],
                var=mean_var_list[random_noise][1])
            return np.uint8(img_noise[0] * 255)

    def _repeat_use_data(self):
        print('增强数据复用！')
        self.data_list_npy_path = self.generate_data_folder + '/augmentation_data_list_npy/'
        self.augmentation_img_list = np.load(
            self.data_list_npy_path + 'img.npy')
        if self.is_enhance_image_only:
            return self.augmentation_img_list

        self.augmentation_label_list = np.load(
            self.data_list_npy_path + 'label.npy')
        return self.augmentation_img_list, self.augmentation_label_list

    def __get_timestamp(self):
        ts = str(int(1000 * datetime.datetime.now().timestamp()))
        return ts[-6:]

    def __write_parameter_information(self):
        augmentation_information_file = open(self.generate_data_folder + '/augmentation_information.txt', 'w+')
        augmentation_information_file.write('augmentation_ratio=' + str(self.augmentation_ratio) + '\n')
        augmentation_information_file.write('generate_data_folder=' + self.generate_data_folder + '\n')
        augmentation_information_file.write('channel=' + str(self.channel) + '\n')
        augmentation_information_file.write('out_file_extension_list=' + str(self.out_file_extension_list) + '\n')
        augmentation_information_file.write('task=' + self.task + '\n')
        augmentation_information_file.write('img_list=' + str(self.img_list) + '\n')
        augmentation_information_file.write('label_list=' + str(self.label_list) + '\n')
        augmentation_information_file.close()