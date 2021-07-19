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

class ClassDataset(Dataset):
    def __init__(self,
                 img_list,
                 label_list,
                 height_width=(256,256),
                 transform=None):
        self.img_list=img_list
        self.label_list=label_list
        self.height_width=height_width
        self.transform = transform  # 变换

    def __len__(self):  # 返回整个数据集的大小
        return len(self.img_list)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        img = cv2.imread(self.img_list[index])
        img = cv2.resize(img, self.height_width, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        label = np.loadtxt(self.label_list[index])
        img = img.transpose((2, 0, 1))  # NHWC -> NCHW
        sample = {'image': img,
                  'label': torch.tensor(label, dtype=torch.float32)}  # 根据图片和标签创建字典

        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        return sample  # 返回该样本