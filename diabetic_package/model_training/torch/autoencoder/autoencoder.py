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
import glob
import torch
import socket
import json
import shutil
from torchsummary import summary
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from diabetic_package.log.log import bz_log
from diabetic_package.model_training.torch.autoencoder import create_dataset
from diabetic_package.model_training.torch.autoencoder import model

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Autoencoder():
    def __init__(self,
                 data_dir,
                 epoch_num,
                 batch_size,
                 class_num=256,
                 height_width=(512, 512),
                 model_dir='./',
                 channels_list=[3, 1],
                 lr=0.001,
                 is_socket=False,
                 is_early_stop=False
                 ):
        self.data_dir = data_dir
        self.class_num = class_num
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.channels_list = channels_list
        self.height_width = height_width
        self.model_dir = model_dir
        self.best_checkpoint_dir = os.path.join(model_dir,
                                                'best_checkpoint_dir')
        self.export_model_dir = os.path.join(model_dir, 'export_model_dir')
        self.best_model_info_path = os.path.join(self.best_checkpoint_dir,
                                                 'best_model_info.json')
        self.lr = lr
        self.is_socket = is_socket
        self.is_early_stop = is_early_stop
        self.eval_epoch_step = 5

        if self.is_socket:
            self.build_socket_connect()
        self.__init_value_judgment()
        self.__init_global_step()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.create_network()

    def fit(self):
        train_data = create_dataset.CreateData(root_dir=self.data_dir + "/train",
                        input_size=(self.height_width[0],self.height_width[1]),
                                               batch_size=self.batch_size)
        val_data = create_dataset.CreateData(self.data_dir + "/val",
                        input_size=(self.height_width[0],self.height_width[1]),
                                             batch_size=self.batch_size)

        self.train_loader = DataLoaderX(train_data,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                        pin_memory=True)#使用DataLoader加载数据
        self.val_loader = DataLoaderX(val_data,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                      pin_memory=True)#使用DataLoader加载数据

        self.model = model.AE(in_channels=self.channels_list[0],
                     out_channels=self.channels_list[1]).to(device=self.device)
        # 打印网络结构
        summary(self.model,(self.channels_list[0], self.height_width[0],
                                                        self.height_width[1]))
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=self.lr)
        #损失函数
        self.criterion =  torch.nn.MSELoss().cuda()
        loss_not_decrease_epoch_num = 0
        for epoch_index in range(self.epoch_num):
            self.model.train()
            train_epochs = 0
            while train_epochs != self.eval_epoch_step:
                epoch_loss = self.train()
                print("第%d个epoch, train loss: %f" %
                                    (epoch_index + train_epochs, epoch_loss))
                train_epochs += 1

            eval_result = self.val()
            print('\033[1;36m 验证集结果:epoch_index=' + str(epoch_index))
            for k, v in eval_result.items():
                print(k + ' =', v)
            print('\033[0m')

            if self.is_socket:
                eval_loss = {
                    'epoch_num': epoch_index + 1,
                    'loss': eval_result['loss']}
                data_dict = list(eval_loss.values())
                data_dict = str(data_dict).encode('utf-8')
                self.socket.send(data_dict)

            # 模型保存的条件
            saved_model_value = eval_result['loss']
            if saved_model_value < self.value_judgment:
                self.value_judgment = saved_model_value
                eval_result['value_judgment'] = self.value_judgment
                eval_result['global_step'] = self.global_step
                self.export_model_dir = self.model_dir + '/export_model_dir'
                self.export_model(eval_result, epoch_index)

            # early stopping
            if (self.is_early_stop):
                loss_tolerance = 0.0005
                print("diff", eval_result["loss"] - self.value_judgment)
                if eval_result["loss"] - self.value_judgment >= loss_tolerance:
                    loss_not_decrease_epoch_num += 1
                    print("loss_not_decrease_epoch_num", loss_not_decrease_epoch_num)
                else:
                    loss_not_decrease_epoch_num = 0
                if loss_not_decrease_epoch_num > 5:
                    print("early stopping 共训练%d个epoch" % epoch_index)
                    break

        if self.is_socket:
            self.socket.close()

    def create_network(self):
        if 0!=self.global_step:
            bz_log.info("恢复模型...")
            AEnet =  model.AE(in_channels=self.channels_list[0],
                    out_channels=self.channels_list[1]).to(device=self.device)
            AEnet.load_state_dict(torch.load(os.path.abspath(
                self.model_dir+'/model_'+str(self.global_step)+'.pth')))
        else:
            bz_log.info("从头开始训练...")
            AEnet = model.AE(in_channels=self.channels_list[0],
                    out_channels=self.channels_list[1]).to(device=self.device)
        AEnet.to(device=self.device)
        return AEnet

    def train(self):
        epoch_loss = 0
        for i_batch, batch_data in enumerate(self.train_loader):
            images = batch_data['image'].to(device=self.device,
                                            dtype=torch.float32)
            labels = batch_data['label'].to(device=self.device,
                                            dtype=torch.float32)
            output = self.model(images)
            batch_loss = self.criterion(output, labels)
            self.optimizer.zero_grad()
            # 反向传播
            batch_loss.backward()
            # 只有用了optimizer.step()，模型才会更新
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            self.global_step += self.batch_size
        return epoch_loss

    def val(self):
        """
        args:
          val_loader: loaded for validation dataset
          model: model
        return: loss
        """
        # evaluation mode
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i_batch, batch_data in enumerate(self.val_loader):
                images = batch_data['image'].to(device=self.device,
                                                dtype=torch.float32)
                labels = batch_data['label'].to(device=self.device,
                                                dtype=torch.float32)
                output = self.model(images)
                batch_loss = self.criterion(output, labels)
                epoch_loss += batch_loss.item()
            assessment_dict={'loss': epoch_loss}
        return assessment_dict

    def export_model(self,eval_result, epoch_index):
        """
        :param export_model_dir: export模型路径
        :return:
        """
        best_checkpoint_dir = self.model_dir + '/best_checkpoint_dir/'
        self.__save_best_checkpoint(best_checkpoint_dir, epoch_index)
        with open(
                self.best_checkpoint_dir + '/best_model_info.json', 'w+') as f:
            eval_result_dict = {k: str(v) for k, v in eval_result.items()}
            json.dump(eval_result_dict, f, indent=1)
        best_checkpoint = glob.glob(best_checkpoint_dir + '*.pth')[0]

        if os.path.exists(self.export_model_dir):
            shutil.rmtree(self.export_model_dir)
        if os.path.exists(self.export_model_dir + '/frozen_model'):
            shutil.rmtree(self.export_model_dir + '/frozen_model')
        os.makedirs(self.export_model_dir + '/frozen_model')
        out_pt_path = self.export_model_dir + '/frozen_model/frozen_model.pt'
        self.__convert_pth_to_pT(best_checkpoint, out_pt_path)
        with open(
           self.export_model_dir + '/frozen_model/model_config.txt', 'w+') as f:
            f.write('model_name:frozen_model.pt' + '\n')
            f.write('input_height:' + str(self.height_width[0]) + '\n')
            f.write('input_width:' + str(self.height_width[1]) + '\n')
            f.write('input_channel:' + str(self.channels_list[0]) + '\n')
            f.write('batch_size:' + str(self.batch_size) + '\n')
            f.write('class_num:' + str(self.class_num))

    def __save_best_checkpoint(self, best_checkpoint_dir, epoch_index):
        """

        :param eval_result: eval模式下输出，字典形式
        :return:
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if (os.path.exists(best_checkpoint_dir)):
            shutil.rmtree(best_checkpoint_dir)
        os.makedirs(best_checkpoint_dir)

        model_file_name = self.model_dir + '/model_' + str(epoch_index) + '.pth'
        # Individual Setting for save model !!!
        torch.save(self.model.state_dict(), model_file_name)

        shutil.copy(model_file_name, best_checkpoint_dir)
        os.mkdir(best_checkpoint_dir + '/' + str(epoch_index) + '_parameters')
        with open(best_checkpoint_dir + '/' + str(epoch_index) +
                               '_parameters' + '/model_config.json', 'w') as f:
            classifier_config_dict = {k: str(v) for k, v in self.__dict__.items()}
            json.dump(classifier_config_dict, f, indent=1)

    def __convert_pth_to_pT(self,best_checkpoint, out_pt_path):
        example = torch.rand(1, self.channels_list[0], self.height_width[0],
                                                       self.height_width[1])
        net = model.AE(in_channels=self.channels_list[0],
                       out_channels=self.channels_list[1])
        net.load_state_dict(self.model.state_dict())
        net.eval()
        traced_script_module = torch.jit.trace(net, example)
        traced_script_module.save(out_pt_path)

    def build_socket_connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("初始化socket")
        # 建立连接:
        try:
            self.socket.connect(('127.0.0.1', 9990))
            print("尝试连接9990")
        except socket.error as e:
            raise ValueError('service connection failed: %s \r\n' % e)
        # 接收欢迎消息:
        print("建立连接")

    def __init_value_judgment(self):
        if os.path.exists(self.best_model_info_path):
            with open(self.best_model_info_path, 'r') as f:
                best_model_info_dict = json.load(f)
            self.value_judgment = float(best_model_info_dict['loss'])
            bz_log.info("打印加载的value_judgment: %f", self.value_judgment)
        else:
            self.value_judgment = 10000
            bz_log.info("value_judgment不存在")


    def __init_global_step(self):
        checkpoint_files = glob.glob(self.model_dir + '/*.pth')
        if checkpoint_files:
            self.global_step = max(list(map(
                        lambda x: int(x.split('_')[-1].split('.pth')[0]),
                        checkpoint_files)))
        else:
            self.global_step = 0

    def to_img(self, x):
        x = x.clamp(0, 1)
        x = x.view(x.size(0), self.channels_list[0], self.height_width[0],
                                                     self.height_width[1])
        return x


# if __name__=="__main__":
#     data_dir = "/home/crx/code_project/autoencoder/logOK/"
#     task = "autoencoder"
#     class_num = 256
#     epoch_num = 100
#     batch_size = 10
#     model_dir = "./all_model_dir"
#     height_width = (512, 512)
#     channels_list = [1, 1]
#
#     model = Autoencoder(data_dir,
#                        class_num,
#                        epoch_num,
#                        batch_size,
#                        height_width=height_width,
#                        model_dir=model_dir,
#                        channels_list=channels_list)
#     model.fit()




