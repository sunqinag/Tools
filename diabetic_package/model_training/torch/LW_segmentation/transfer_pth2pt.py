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
from diabetic_package.model_training.torch.LW_segmentation.model.DABNet \
    import DABNet

def transfer_pth2pt(model, result_dir, classes):
    net = DABNet(classes=classes)
    net.load_state_dict(model.state_dict())
    net.eval()

    example = torch.rand(1, 3, 400, 400)
    ts = torch.jit.trace(net, example)

    ts.save(result_dir + "/frozen.pt")
