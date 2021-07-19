# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: YOLOv3_change
File Name: teset.py
Author: xtcsun
Create Date: 2020/12/31
-------------------------------------------------
"""
import torch
from .nets.model_main import ModelMain
from .nets.yolo_loss import YOLOLoss
from .cfg import cfg
import os
import cv2
import numpy as np
import torch.nn as nn
from .utils import non_max_suppression

# class_dict = {0:'aeroplane',
#               1:'bicycle',
#               2:'bird',
#               3:'boat',
#               4:'bottle',
#               5:'bus',
#               6:'car',
#               7:'cat',
#               8:'chair',
#               9:'cow',
#               10:'diningtable',
#               11:'dog',
#               12:'horse',
#               13:'motorbike',
#               14:'person',
#               15:'pottedplant',
#               16:'sheep',
#               17:'sofa',
#               18:'train',
#               19:'tvmonitor'}

# class_dict={0:'background',
#             1:'barcode',
#             2:'erweima',
#             3:'qr'}

class_dict = {0: (0, 255, 0),
              1: (0, 0, 255),
              2: (0, 255, 255)}
model_path = 'model_dir/model.pth'

test_dir = 'out_path/test/images/'
image_dir = test_dir

image_list = sorted([image_dir + file for file in os.listdir(image_dir)])
# label_list = sorted([label_dir + file for file in os.listdir(label_dir)])

net = ModelMain(cfg)

net.eval()

net.cuda()

model_data = torch.load(model_path)
net.load_state_dict(model_data)

yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLoss(cfg["yolo"]["anchors"][i],
                                cfg["yolo"]["classes"], (cfg["img_w"], cfg["img_h"])))
# YOLO loss with 3 scales
yolo_losses = []
for i in range(3):
    yolo_losses.append(YOLOLoss(cfg["yolo"]["anchors"][i],
                                cfg["yolo"]["classes"], (cfg["img_w"], cfg["img_h"])))

for image in image_list:
    image_name = os.path.basename(image)

    image = cv2.imread(image)
    image_src = cv2.resize(image, (416, 416))
    image = np.transpose(image_src, (2, 0, 1))

    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image.astype(np.float32)).cuda()
    outputs = net(image)

    ####demo
    output_list = []
    for i in range(3):
        output_list.append(yolo_losses[i](outputs[i]))
    outputs = torch.cat(output_list, 1)

    output = non_max_suppression(outputs, cfg["yolo"]["classes"], conf_thres=0.2, nms_thres=0.2)

    for detections in output:
        if detections is not None:
            detections = detections.cpu().numpy()
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                print((x1, y1), (x2, y2))
                # cls_name = class_dict[int(cls_pred)]+str(conf)
                cls = class_dict[int(cls_pred)]
                cv2.rectangle(image_src, (x1, y1), (x2, y2), cls)

    image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)

    cv2.imwrite('save/' + image_name, image_src)
