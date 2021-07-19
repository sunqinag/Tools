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
import torch.nn as nn

class LW_Classnet(nn.Module):
    def __init__(self,class_num):
        super(LW_Classnet, self).__init__()
        self.conv1 = self.conv_layer(3, 32)
        self.conv2 = self.conv_layer(32, 64)
        self.conv3 = self.conv_layer(64, 128)
        self.conv4 = self.conv_layer2(128, 256)
        self.conv5 = nn.Conv2d(256, class_num, 3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(class_num)

    def conv_layer(self, In, out):
        conv_layer = nn.Sequential(
        nn.Conv2d(In, out, 3, stride=2, padding=3),
        nn.BatchNorm2d(out),
        nn.ReLU(inplace=True),
        nn.Conv2d(out, out, 1, padding=0),
        nn.BatchNorm2d(out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, stride=2))
        return conv_layer

    def conv_layer2(self, In, out):
        conv_layer = nn.Sequential(
            nn.Conv2d(In, out, 3, stride=2, padding=3),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, 1, padding=0),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        )
        return conv_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        return x






