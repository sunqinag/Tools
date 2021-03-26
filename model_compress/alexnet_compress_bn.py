#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : alexnet_compress_bn.py
@Author: 孙强
@Date  : 2021/3/23 下午4:52
@Desc  : 
'''
import os
import torch
import torch.nn as nn
import logging

from diabetic_package.model_training.torch.classification.ClassNet import BZ_Alexnet

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s', )
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 数据集
data_dir = '../model-compression/data/class_catdog/train'

# 输入通道数
input_channels = 3
# 剪枝率
percent = 0.35
# 正常|规整剪枝
nrmal_regular = 1
# 剪枝层数，0默认所有
layers = 0
# 要压缩的model
spase_model = '../model-compression/module/best_checkpoint_dir/classification_model_279972.pth'
# 剪枝后保存的model
save_path = '../save_compress_model/compress_alexnet.pth'

model = BZ_Alexnet(num_classes=2).cuda()

model.load_state_dict(torch.load(spase_model))
print('模型加载完成')

print('旧模型', model)

# *******************************预剪枝**************************************
cfg = [input_channels]
start_linear = True
retain_index_list = []
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.clone()
        weight_sort, index = torch.sort(weight_copy, descending=True)
        thre_index = int(len(weight_copy) * percent)
        if thre_index%2!=0:
            thre_index+=1
        retain_index = index[:thre_index]
        cfg.append(thre_index)
        retain_index_list.append(retain_index)
    elif start_linear and isinstance(m, nn.Linear):
        node = m.weight.data.shape[1]
        cfg.append(node)
        start_linear = False
    elif isinstance(m, nn.BatchNorm1d):
        weight_copy = m.weight.data.clone()
        weight_sort, index = torch.sort(weight_copy, descending=True)
        thre_index = int(len(weight_copy) * percent)
        if thre_index%2!=0:
            thre_index+=1
        retain_index = index[:thre_index]
        cfg.append(thre_index)
        retain_index_list.append(retain_index)

print("预剪枝完成")
new_model = BZ_Alexnet(num_classes=2, cfg=cfg).cuda()

i = 0
for [new_m, m] in zip(new_model.modules(), model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        print('BatchNorm2d')
        weight_copy = m.weight.data.clone()
        new_m.weight.data = m.weight.data[retain_index_list[i]].clone()
        new_m.bias.data = m.bias.data[retain_index_list[i]].clone()
        new_m.running_mean = m.running_mean[retain_index_list[i]].clone()
        new_m.running_var = m.running_var[retain_index_list[i]].clone()
        i += 1
    elif isinstance(m, nn.Conv2d):
        print('Conv2d')
        weight_copy = m.weight.data.clone()
        new_m.weight.data = weight_copy[retain_index_list[i], :, :, :].clone()
        new_m.bias.data = m.bias.data[retain_index_list[i]].clone()
    elif isinstance(m, nn.BatchNorm1d):
        print('BatchNorm1d')
        weight_copy = m.weight.data.clone()
        new_m.weight.data = m.weight.data[retain_index_list[i]].clone()
        new_m.bias.data = m.bias.data[retain_index_list[i]].clone()
        new_m.running_mean = m.running_mean[retain_index_list[i]].clone()
        new_m.running_var = m.running_var[retain_index_list[i]].clone()
        i += 1
    elif isinstance(m, nn.Linear):
        print('Linear')
        new_m.weight.data = m.weight.data[retain_index_list[i], :].clone()
print('模型剪枝完成')

# ***********************************剪枝后模型保存**************************************
torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, save_path)
# torch.save(new_model.state_dict(),save_path)
print("模型保存完成")


# 微调：
print('开始 fine tuning')
from diabetic_package.model_training.torch.classification.classifier import Classifier
from src.dataset import get_img_and_label
train_dir = '/media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/PycharmProjects/模型压缩/model-compression/data/class_catdog/train'
val_dir = '/media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/PycharmProjects/模型压缩/model-compression/data/class_catdog/val'

class_num=2
ckpt = torch.load(save_path)
new_model = BZ_Alexnet(num_classes=class_num,percent=percent,cfg=ckpt['cfg']).cuda()
new_model.state_dict(ckpt['state_dict'])

print('new model',new_model)
train_img_list, train_label_list = get_img_and_label(train_dir)
val_img_list, val_label_list = get_img_and_label(val_dir)


# from src.test import test
# test(new_model=new_model,model_dir=save_path,compress=True)

height_width = (256, 256)
obj = Classifier(class_num=class_num,
                 accuracy_weight=[1, 1],
                 label_weight=[1, 1],
                 batch_size=32,
                 transfer_checkpoint_path=save_path,
                 model_dir='../Fine_tuning_dir',
                 height_width=height_width,
                 epoch_num=50,
                 eval_epoch_step=5,
                 keep_max=1,
                 cfg=cfg,
                 new_model=new_model)
obj.fit(
    train_images_path=train_img_list,
    train_labels=train_label_list,
    eval_images_path=val_img_list,
    eval_labels=val_label_list
)
