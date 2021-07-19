# 导入相关模块
from torch.utils.data import DataLoader, Dataset
import numpy as np
from diabetic_package.file_operator import bz_path
import cv2
import torch
from skimage import io,transform
from diabetic_package.image_processing_operator.python_data_augmentation.python_data_augmentation import python_base_data_augmentation


class CreateData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, input_size, task, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images, self.labels = np.sort(bz_path.get_all_subfolder_img_label_path_list(root_dir))
        self.input_size = input_size

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        img = cv2.imread(self.images[index])
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        label = np.loadtxt(self.labels[index])
        img = img.transpose((2,0,1)) #NHWC -> NCHW
        sample = {'image': img, 'label': torch.tensor(label,dtype=torch.float32)}  # 根据图片和标签创建字典

        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        return sample  # 返回该样本


class ClassDataset(Dataset):
    def __init__(self, img_list,label_list, task, height_width=(256,256),transform=None):
        self.img_list=img_list
        self.label_list=label_list
        self.height_width=height_width
        self.transform = transform  # 变换
        self.task = task

    def __len__(self):  # 返回整个数据集的大小
        return len(self.img_list)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # print("index--------", index)
        img = cv2.imread(self.img_list[index])
        img = cv2.resize(img, self.height_width, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))  # NHWC -> NCHW

        if self.task == "classification" or self.task == "lw_classification":
            label = np.loadtxt(self.label_list[index])
            sample = {'image': img,
                      'label': torch.tensor(label,
                                            dtype=torch.float32)}  # 根据图片和标签创建字典
        elif self.task == "autoencoder":
            img = cv2.imread(self.img_list[index], 0)
            img = cv2.resize(img, self.height_width,
                             interpolation=cv2.INTER_CUBIC)
            img = img.astype(np.float32) / 255
            label = img.copy()
            label = torch.tensor(label, dtype=torch.float32)
            label = label.unsqueeze(0)  # 增加维度

            if self.mode == "train":
                img = python_base_data_augmentation.add_noise(img, ['gaussian'],
                                                              0.0001, 0.0001)[0]
            img = torch.tensor(img, dtype=torch.float32)
            img = img.unsqueeze(0)
            sample = {'image': img, 'label': label}  # 根据图片和标签创建字典


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
