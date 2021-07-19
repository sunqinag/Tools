# ----------------------------
#!  Copyright(C) 2021
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：陈瑞侠
#   完成日期：2021-3-4
# -----------------------------
import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AE, self).__init__()
        self.conv1 = self.conv(in_channels, 64)
        self.conv2 = self.conv(64, 32)
        self.conv3 = self.conv(32, 32)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1 ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.pool = nn.Sequential(nn.MaxPool2d(2, stride=2))

        self.dense1 = self.dense(16384, 1024)
        self.dense2 = self.dense(1024, 512)
        self.dense3 = self.dense(512, 1024)
        self.dense4 = self.dense(1024, 16384)

        self.deconv1 = self.convtransepose(16,32)
        self.deconv2 = self.convtransepose(48, 32)
        self.deconv3 = self.convtransepose(32, 64)
        self.deconv4 = self.convtransepose(64, out_channels)

    def conv(self, In, out):
        conv_layer = nn.Sequential(
        nn.Conv2d(In, out, 3, stride=1, padding=1),
        nn.BatchNorm2d(out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, stride=2))
        return conv_layer

    def dense(self, In, out):
        dense_layer = nn.Sequential(
        nn.Linear(In, out),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True)
        )
        return dense_layer

    def convtransepose(self, In, out):
        conv_layer = nn.Sequential(
        nn.ConvTranspose2d(In, out,2, stride=2, padding=0),
        nn.BatchNorm2d(out),
        nn.ReLU(inplace=True)
        )
        return conv_layer

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        pool = self.pool(conv4)
        flat = pool.view(pool.size(0), -1)
        dense1 = self.dense1(flat)
        dense2 = self.dense2(dense1)
        dense3 = self.dense3(dense2)
        dense4 = self.dense4(dense3)
        reshape = dense4.view(pool.size())
        deconv1 = self.deconv1(reshape)
        concat = torch.cat((conv4, deconv1), 1)
        deconv2 = self.deconv2(concat)
        deconv3 = self.deconv3(deconv2)
        logits = self.deconv4(deconv3)
        return logits

