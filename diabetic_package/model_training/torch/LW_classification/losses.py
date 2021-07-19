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
import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss

__all__=["CrossEntropyLoss", "CrossEntropyLoss2dLabelSmooth"]

class CrossEntropyLoss(_WeightedLoss):
    """
    standard pytorch weighted nn.CrossEntropyLoss
    """
    def __init__(self, weight=None, ignore_label=255, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(
                        weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        :param output: torch.tensor (NXC)
        :param target: torch.tensor(N)
        :return: scalar
        """
        return self.nll_loss(output, target)

class CrossEntropyLoss2dLabelSmooth(_WeightedLoss):
    """
    :param target:N
    :param n_classes:int
    :param eta:float
    ":return
        NXC onhot smothed vector
    """
    def __init__(self,
                 weight=None,
                 ignore_label=255,
                 epsilon=0.1,
                 reduction="mean"):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.nll_loss = nn.CrossEntropyLossW(
                        weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        :param output: torch.tensor (NXC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        n_classes = output.size(1)
        targets = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
        targets = (1-self.epsilon) * targets + self.epsilon / n_classes
        return self.nll_loss(output, targets)
