# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: YOLOv3
File Name: dataset.py
Author: xtcsun
Create Date: 2020/12/25
-------------------------------------------------
"""
import os
import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset



class DataSet(Dataset):
    '''
    注意:本dataset中图像是BGR格式读取,label以txt形式保存
    '''

    def __init__(self, train_dir, img_size=(640, 640), max_objects=50):
        '''
        :param train_dir: 训练或验证文件夹
        :param img_size: 统一输入尺度,默认(416,416)
        :param max_objects: 每个图像中最多目标数量,也是pading的最大数量
        '''
        self.image_dir = train_dir + os.sep + 'images'
        self.label_dir = train_dir + os.sep + 'labels'
        self.img_size = img_size
        self.max_objects = max_objects
        self.image_list, self.label_list = self.get_img_and_label_list(train_dir)

    def get_img_and_label_list(self, train):
        image_list = np.load(train + os.sep + 'img_list.npy').flatten()
        label_list = np.load(train + os.sep + 'label_list.npy').flatten()
        return image_list, label_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = cv2.imread(image_path)

        # print(image_path)
        label_path = self.label_list[index]

        label = np.load(label_path).reshape((-1, 5)).astype(np.float32)
        label = self.xyxyid2idxyxy(label)
        resize_image, resize_label = self.__resize_image_label(image, label, self.img_size)

        sample = self.__to_tensor(resize_image, resize_label)
        return sample

    def xyxyid2idxyxy(self, label):
        label_copy = np.zeros(label.shape, dtype=np.float32)
        label_copy[:, 1:] = label[:, :4]
        label_copy[:, 0] = label[:, 4]
        return label_copy

    def __resize_image_label(self, image, label, resize_size):
        height, width = image.shape[:2]
        scale_x = resize_size[0] / width
        scale_y = resize_size[1] / height

        image = cv2.resize(image, resize_size)

        label[:, 1:] *= [scale_x, scale_y, scale_x, scale_y]

        return image, label

    def __to_tensor(self, image, label):
        '''
        将label从(N,5)扩展到统一尺度,(max_objects,5)
        :return:
        '''
        # 代测试
        # image = image.astype(np.float32)
        # image /= 255.0
        image = np.transpose(image, (2, 0, 1))

        label = self.__label_normal(label)

        filled_labels = np.zeros((self.max_objects, 5), np.float32)
        filled_labels[range(len(label))[:self.max_objects]] = label[:self.max_objects]
        return {'image': torch.from_numpy(image).float(), 'label': torch.from_numpy(filled_labels).float()}

    def __label_normal(self, label):
        '''
        将label归一化
        :param label:
        :return:
        '''
        h, w = self.img_size
        label[:, 1] /= w
        label[:, 3] /= w
        label[:, 2] /= h
        label[:, 4] /= h

        h_ = label[:, 4] - label[:, 2]
        w_ = label[:, 3] - label[:, 1]
        label[:, 1] += w_ / 2
        label[:, 2] += h_ / 2
        label[:, 3] = w_
        label[:, 4] = h_
        return label


if __name__ == '__main__':
    train_dir = 'out_path/val'
    dataset = DataSet(train_dir)
    # dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=2,shuffle=True,num_workers=4)
    for sample in dataset:
        image = sample['image'].cpu().numpy()
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8).copy()
        h, w = image.shape[:2]
        label = sample['label']
        # print(label)

        for t in range(label.shape[0]):
            target = label[t]
            if target[4] < 0:
                print(target)
                break
        for l in label:
            if l.sum() == 0:
                continue

            x1 = int((l[1] - l[3] / 2) * w)
            y1 = int((l[2] - l[4] / 2) * h)
            x2 = int((l[1] + l[3] / 2) * w)
            y2 = int((l[2] + l[4] / 2) * h)
            # cv2.rectangle(image, (l[1],l[2]), (l[3], l[4]), (0, 0, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("step{}_{}.jpg".format(1, 1), image)
        cv2.waitKey(0)
