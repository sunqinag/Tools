# -*- coding: utf-8 -*-
# ----------------------------
#!  Copyright(C) 2021
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：陈瑞侠
#   完成日期：2021-3-4
# -----------------------------
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from diabetic_package.file_operator import bz_path

class CreateData(Dataset):  # 继承Dataset
    def __init__(self,
                 root_dir,
                 input_size,
                 batch_size,
                 transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.input_size = input_size
        self.batch_size = batch_size

        self.images, self.labels = np.sort(
            bz_path.get_all_subfolder_img_label_path_list(root_dir))
        if self.batch_size > len(self.images):
            self.images = np.append(self.images, self.images)
            self.labels = np.append(self.labels, self.labels)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # print("index--------", index)
        img = cv2.imread(self.images[index])
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        label = np.loadtxt(self.labels[index])
        img = img.transpose((2,0,1)) #NHWC -> NCHW
        sample = {'image': img, 'label': torch.tensor(label,dtype=torch.float32)}  # 根据图片和标签创建字典

        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        return sample  # 返回该样本

# if __name__=='__main__':
#     data_dir = 'E:/Python Project/PyTorch/dogs-vs-cats/train'
#     data = CreateData(data_dir,transform=None)#初始化类，设置数据集所在路径以及变换
#     dataloader = DataLoader(data,batch_size=128,shuffle=True)#使用DataLoader加载数据
#     for i_batch,batch_data in enumerate(dataloader):
#         print(i_batch)#打印batch编号
#         print(batch_data['image'].size())#打印该batch里面图片的大小
#         print(batch_data['label'])#打印该batch里面图片的标签
