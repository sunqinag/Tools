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
import os
import cv2
import torch
import shutil
import numpy as np
from torch.utils.data import DataLoader

from diabetic_package.model_training.torch.LW_classification import create_dataset
from diabetic_package.model_training.torch.LW_classification import model \
    as class_model
from diabetic_package.machine_learning_common.accuracy import python_accuracy

class ModelLoader():
    def __init__(self, model_dir,
                 class_num,
                 batch_size=10,
                 result_dir="./result/",
                 height_width=(256,256)):
        self.height_width = height_width
        self.result_dir = result_dir
        self.batch_size = batch_size
        self.class_num = class_num

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = class_model.LW_Classnet(class_num=class_num)
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.cuda()

    def predict(self,data_dir):

        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir)
        os.makedirs(self.result_dir)
        for i in range(self.class_num):
            os.makedirs(self.result_dir + str(i))

        data = create_dataset.CreateData(data_dir,
                                        input_size=(self.height_width[0],
                                                    self.height_width[1]))
        dataloader = DataLoader(data,
                                batch_size=self.batch_size,
                                shuffle=True)  # 使用DataLoader加载数据
        pred_classes = []
        true_labels = []
        self.model.eval()
        with torch.no_grad():
            for i_batch, batch_data in enumerate(dataloader):
                images = batch_data['image'].to(device=self.device,
                                                dtype=torch.float32)
                labels = batch_data['label'].to(device=self.device,
                                                dtype=torch.int64)

                output = self.model(images)
                output = output.squeeze()
                softmax = torch.softmax(output, dtype=torch.float32, dim=1)
                batch_classes = torch.argmax(softmax,dim=1).cpu().numpy()

                images_cpu = images.cpu().detach().numpy()
                for i in range(len(batch_classes)):
                    idex = i_batch * len(batch_data) + i
                    img = images_cpu[i]
                    img = img.squeeze()
                    img = img.transpose((1, 2, 0))  # NHWC -> NCHW
                    cv2.imwrite(self.result_dir + str(batch_classes[i]) + "/" \
                                + str(idex) + ".jpg", img)

                pred_classes = np.append(pred_classes, batch_classes)
                true_labels = np.append(true_labels, labels.cpu().numpy())

            assessment_dict = ['accuracy', 'recall', 'precision']
            assessment_dict = python_accuracy.get_assessment_result(true_labels,
                                                                    pred_classes,
                                                                    self.class_num,
                                                                    'eval',
                                                                    assessment_dict)
            return assessment_dict

if __name__ =="__main__":
    model_dir = "/home/crx/code_project/deeplearn_example/gongye_lwc/all_models_dir/model_17.pth"
    data_dir = "/home/crx/code_project/deeplearn_example/data/gongye_split/val"
    result_dir = "/home/crx/code_project/deeplearn_example/result/"
    class_num = 10
    model = ModelLoader(model_dir, class_num, result_dir=result_dir)
    model.predict(data_dir)


