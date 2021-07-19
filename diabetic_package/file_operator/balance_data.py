# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：balance_data
#   摘   要：对不均衡的各类数据进行增强到统一数量返回均衡后的结果
#   当前版本:2019121817
#   作   者：孙强
#   完成日期：2019-12-18

import numpy as np
import os, math, filecmp, shutil

from . import bz_path
from ..image_processing_operator.python_data_augmentation import python_data_augmentation
from diabetic_package.log.log import bz_log

class BalanceData:
    def __init__(self, base_folder, generate_data_folder, out_file_extension_list, channel=3, max_augment_num=None,
                 task='classification'):
        '''
        对不均衡的各类数据进行增强到统一数量返回均衡后的结果，请确保数据的文件夹结构为：
        父文件夹-->各子类文件夹-->img和label请务必确保这两个文件夹命名与上述完全一致
        :param base_folder: 父文件夹路径,其子文件夹为各类别文件夹如1,2,3等
        :param generate_data_folder： 生成数据集文件夹路径
        :param out_file_extension_list： 输出文件格式
        :param max_augment_num： 各类最大生成样本数量
        :param task: 任务模式，可选择参数为L：’classification‘，'detection'，’segmentation‘
        :return: 样本均衡后的img_list和label_list
        '''
        self.base_folder = base_folder
        self.generate_data_folder = generate_data_folder
        self.channel = channel
        self.max_augment_num = max_augment_num
        self.src_max_augment_num = max_augment_num
        self.task = task.lower()
        self.out_file_extension_list = out_file_extension_list
        self.sub_folders = sorted(os.listdir(base_folder))
        self.generate_data_folders = []
        self.is_repeat_data = False

        self.__check_params()

    def create_data(self):
        self.__check_info_txt()
        img_list, label_list, augmentation_ratios, generate_data_folders = self.__get_augment_params()
        bz_log.info('生成均衡数据')
        img_list, label_list= self.__augment_data(img_list=img_list, label_list=label_list,
            augmentation_ratios=augmentation_ratios,
            channel=self.channel,
            generate_data_folders=generate_data_folders)

        return img_list.flatten(), label_list.flatten()

    def __get_augment_params(self):
        self.img_nums = []
        self.img_list = []
        self.label_list = []
        for folder in self.sub_folders:
            imgs, labels = bz_path.get_train_val_img_label_path_list(self.base_folder + os.sep + folder + os.sep + 'img',
                                                           self.base_folder + os.sep + folder + os.sep + 'label',
                                                           ret_full_path=True)
            single_class_img_num = len(imgs)
            if single_class_img_num == 0:
                bz_log.error('文件夹' + self.base_folder + os.sep + folder + '中样本个数为0，请增加对应类别样本或者删除该文件夹！')
                raise ValueError('文件夹' + self.base_folder + os.sep + folder + '中样本个数为0，请增加对应类别样本或者删除该文件夹！')
            self.img_nums.append(single_class_img_num)
            self.img_list.append(imgs)
            self.label_list.append(labels)
            self.generate_data_folders.append(self.generate_data_folder + os.sep + folder)

        format_list = [format_cls for format_cls in ['bmp', 'jpg', 'png', 'jpeg'] if format_cls in str(self.img_list)]

        if len(format_list) > 1:
            bz_log.error('img中出现不同格式后缀')
            bz_log.error(format_list)
            raise ValueError('img中出现不同格式后缀')
        if self.max_augment_num is None:
            self.max_augment_num = np.max(self.img_nums)
        self.augmentation_ratios = [math.ceil(self.max_augment_num / i) for i in self.img_nums]
        self.augmentation_nums = [self.img_nums[i] * self.augmentation_ratios[i] for i in
                                  np.arange(len(self.augmentation_ratios))]
        return self.img_list, self.label_list, self.augmentation_ratios, self.generate_data_folders

    def __check_params(self):
        if not os.path.exists(self.base_folder):
            bz_log.error(str(self.base_folder) + '文件夹不存在!')
            bz_log.error(str(self.base_folder))
            raise ValueError(str(self.base_folder) + '文件夹不存在!')
        if len(self.out_file_extension_list) != 2:
            bz_log.error("参数out_file_extension_list输入的长度必须为2!")
            bz_log.error(len(self.out_file_extension_list))
            raise ValueError("参数out_file_extension_list输入的长度必须为2!")
        if self.task == 'classification' or self.task == 'detection':
            if (self.out_file_extension_list[0] not in ['bmp', 'jpg', 'png', 'jpeg']) or (
                    self.out_file_extension_list[1] not in ['txt', 'npy']):
                bz_log.error('输入out_file_extension_list参数不正确，请重新检查！')
                bz_log.error(self.out_file_extension_list)
                raise ValueError('输入out_file_extension_list参数不正确，请重新检查！')
        if self.channel != 1 and self.channel != 3:
            bz_log.error("channel必须为1或3")
            bz_log.error(self.channel)
            raise ValueError("channel必须为1或3")
        if self.max_augment_num is not None and self.max_augment_num < 1:
            bz_log.error('单类别最大数量不能小于1！')
            bz_log.error(self.max_augment_num)
            raise ValueError('单类别最大数量不能小于1！')
        if self.task != 'classification' and self.task != 'detection' and self.task != 'segmentation':
            bz_log.error("task参数错误，必须为‘classification’，'detection'或者‘segmentation’请检查参数拼写！")
            bz_log.error(self.task)
            raise ValueError("task参数错误，必须为‘classification’，'detection'或者‘segmentation’请检查参数拼写！")

    def __check_info_txt(self):
        if not os.path.exists(self.generate_data_folder):
            os.makedirs(self.generate_data_folder)
        if not os.path.exists(self.generate_data_folder + os.sep + 'img_list.npy') or not os.path.exists(
                self.generate_data_folder + os.sep + 'label_list.npy') or not os.path.exists(
            self.generate_data_folder + os.sep + 'result_info.txt'):
            bz_log.info('信息文件丢失，正在重新生成数据')
            shutil.rmtree(self.generate_data_folder)
            self.is_repeat_data = False
        if os.path.exists(self.generate_data_folder + os.sep + 'result_info.txt'):
            if os.path.exists(self.generate_data_folder + os.sep + 'result_info_old.txt'):
                os.remove(self.generate_data_folder + os.sep + 'result_info_old.txt')
            os.rename(self.generate_data_folder + os.sep + 'result_info.txt',
                      self.generate_data_folder + os.sep + 'result_info_old.txt')
            self.__write_result_info()
            self.is_repeat_data = filecmp.cmp(self.generate_data_folder + os.sep + 'result_info.txt',
                                              self.generate_data_folder + os.sep + 'result_info_old.txt')
            if not self.is_repeat_data:
                shutil.rmtree(self.generate_data_folder)

    def __augment_data(self, img_list, label_list, augmentation_ratios, channel, generate_data_folders):
        '''
        调用数据增强模块将所有类别按照相应的增强比例增强
        :param base_folder: 原始数据文件夹路径
        :param sub_folders: 原始数据文件夹下各类别文件夹
        :param generate_data_folder: 增强文件夹路径
        :param augmentation_nums: 各类别进行图像增强后的数量
        :param augmentation_ratios: 各类别进行图像增强的比例
        :param generate_data_folders: 增强文件夹下各类别文件夹
        :param max_augment_num: 最大增强数量
        :return:
        '''
        bz_log.info("调用数据增强模块将所有类别按照相应的增强比例增强")
        generate_img_list = []
        generate_label_list = []
        for i in range(len(self.sub_folders)):
            augmentation_data = python_data_augmentation.DataAugmentation(img_list=img_list[i],
              label_list=label_list[i],
              augmentation_ratio=augmentation_ratios[i],
              channel=channel,
              task=self.task,
              generate_data_folder=generate_data_folders[i],
              out_file_extension_list=self.out_file_extension_list)
            aug_single_class_img_list, aug_single_class_label_list = augmentation_data.augment_data()

            shuffle_indices = np.arange(len(aug_single_class_img_list))
            np.random.shuffle(shuffle_indices)
            shuffle_aug_single_class_img_list = aug_single_class_img_list[shuffle_indices]
            shuffle_aug_single_class_label_list = aug_single_class_label_list[shuffle_indices]

            generate_img_list.append(
                shuffle_aug_single_class_img_list.tolist()[:self.max_augment_num])
            generate_label_list.append(
                shuffle_aug_single_class_label_list.tolist()[:self.max_augment_num])
            if self.augmentation_nums[i] != self.max_augment_num:
                for index in range(self.max_augment_num, len(aug_single_class_img_list)):
                    os.remove(shuffle_aug_single_class_label_list[index])
                    os.remove(shuffle_aug_single_class_img_list[index])
        generate_img_list = np.array(generate_img_list)
        generate_label_list = np.array(generate_label_list)
        img_list_npy_path = self.generate_data_folder + os.sep + 'img_list'
        label_list_npy_path = self.generate_data_folder + os.sep + 'label_list'
        np.save(img_list_npy_path, generate_img_list)
        np.save(label_list_npy_path, generate_label_list)
        self.all_img_nums = generate_img_list.size
        self.__write_result_info()
        return generate_img_list, generate_label_list

    def __write_result_info(self, ):
        bz_log.info("记录当前训练的数据信息")
        with open(self.generate_data_folder + os.sep + 'result_info.txt', 'w') as f:
            f.write('base_folder=' + str(self.base_folder) + '\n')
            f.write('generate_data_folder=' + str(self.generate_data_folder) + '\n')
            f.write('out_file_extension_list=' + str(self.out_file_extension_list) + '\n')

            f.write('src_max_augment_num=' + str(self.src_max_augment_num) + '\n')
            f.write('task=' + self.task + '\n')
            f.write('channel=' + str(self.channel) + '\n')

# if __name__ == '__main__':
#     img_list, label_list = BalanceData(
#         # base_folder='/media/bzsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/孙强电脑备份/AOI/0822',
#         base_folder='/media/bzsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/孙强电脑备份/AOI/indus_data_base_class',
#         generate_data_folder='/media/bzsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/孙强电脑备份/AOI/balance_generate',
#         out_file_extension_list=['jp', 'txt'],
#         max_augment_num=30,
#         # task='segmentation'
#         # task='detection'
#     ).create_data()
