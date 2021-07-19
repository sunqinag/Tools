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
from torch.utils.data import DataLoader

from diabetic_package.model_training.torch.autoencoder import create_dataset
from diabetic_package.model_training.torch.autoencoder import model  as AE

class ModelLoader():
    def __init__(self,
                 model_dir,
                 batch_size=10,
                 channels_list=(1,1),
                 result_dir="./result/",
                 height_width=(512,512)):
        self.channels_list = channels_list
        self.height_width = height_width
        self.result_dir = result_dir
        self.batch_size = batch_size

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AE.AE(in_channels=self.channels_list[0],
                    out_channels=self.channels_list[1]).to(device=self.device)
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.cuda()

    def predict(self,data_dir):
        data = create_dataset.CreateData(data_dir,
                                         input_size=(self.height_width[0],
                                                    self.height_width[1]))
        dataloader = DataLoader(data, batch_size=self.batch_size,
                                       shuffle=True)  # 使用DataLoader加载数据
        self.model.eval()

        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir)
        os.makedirs(self.result_dir)

        for i_batch, batch_data in enumerate(dataloader):
            images = batch_data['image'].to(device=self.device,
                                            dtype=torch.float32)
            labels = batch_data['label'].to(device=self.device,
                                            dtype=torch.float32)
            with torch.no_grad():
                images = images.cuda()
                output = self.model(images)
            # torch.cuda.synchronize()

            output = output.cpu().detach().numpy()
            for i in range(len(output)):
                img = output[i]
                img = img.squeeze()
                idex = i_batch*len(batch_data) + i
                cv2.imwrite(self.result_dir + "/" + str(idex) + ".jpg", img * 255)

if __name__=="__main__":
    data_dir = "/home/crx/code_project/autoencoder/logOK/val/"
    model_dir = "/home/crx/code_project/autoencoder_pytorch/all_model_dir/model_93.pth"
    result_dir = "/home/crx/code_project/deeplearn_example/AE_result/"
    model = ModelLoader(model_dir, result_dir=result_dir)
    model.predict(data_dir)
