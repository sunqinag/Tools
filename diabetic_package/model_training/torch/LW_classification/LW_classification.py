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
import glob
import torch
import socket
import json
import shutil
import numpy as np
from torchsummary import summary
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from diabetic_package.log.log import bz_log
from diabetic_package.model_training.torch.LW_classification.losses \
                                                    import CrossEntropyLoss
from diabetic_package.model_training.torch.LW_classification \
                                                    import create_dataset
from diabetic_package.model_training.torch.LW_classification \
                                                 import model as class_model
from diabetic_package.machine_learning_common.accuracy import python_accuracy

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class LW_Classification():
    def __init__(self,
                 data_dir,
                 class_num,
                 epoch_num,
                 batch_size,
                 lr=0.1,
                 height_width=(64, 64),
                 model_dir='./',
                 channels_list=[3],
                 assessment_list=['accuracy', 'recall', 'precision'],
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
        self.assessment_list = assessment_list
        self.is_socket = is_socket
        self.is_early_stop = is_early_stop
        self.eval_epoch_step = 5

        if self.is_socket:
            self.build_socket_connect()

        self.__init_value_judgment()
        self.__init_global_step()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.create_network()

    def fit(self):
        import time
        t1 = time.time()

        train_data = create_dataset.CreateData(self.data_dir +  "/train",
                                               input_size=(64,64),
                                               batch_size=self.batch_size)
        val_data = create_dataset.CreateData(self.data_dir + "/val",
                                             input_size=(64,64),
                                             batch_size=self.batch_size)
        self.train_loader = DataLoaderX(train_data,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        drop_last=False,
                                        pin_memory=True)
        self.val_loader = DataLoaderX(val_data,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      pin_memory=True)
        self.model = class_model.LW_Classnet(
                                class_num=self.class_num).to(device=self.device)
        # 输出网络结构
        summary(self.model, (self.channels_list[0], self.height_width[0],
                                                        self.height_width[1]))

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      lr=self.lr)
        #损失函数
        criterion = CrossEntropyLoss(ignore_label=self.class_num)
        self.criterion = criterion.cuda()
        loss_not_decrease_epoch_num = 0
        self.value_judgment = 100000
        for epoch_index in range(self.epoch_num):#loop over the dataset multiple times
            self.model.train()
            train_epochs = 0
            while train_epochs != self.eval_epoch_step:
                epoch_loss = self.train()
                train_epochs += 1
            print("第%d个epoch, train loss: %f" %
                  (epoch_index, epoch_loss))

            eval_result = self.val()
            print('\033[1;36m 验证集结果:epoch_index=' + str(epoch_index))
            for k, v in eval_result.items():
                print(k + ' =', v)
            print('\033[0m')

            # 模型保存的条件
            saved_model_value = eval_result['loss']
            if saved_model_value < self.value_judgment:
                self.value_judgment = saved_model_value
                eval_result['value_judgment'] = self.value_judgment
                eval_result['global_step'] = self.global_step
                self.export_model_dir = self.model_dir + '/export_model_dir'
                self.export_model(eval_result, epoch_index)
                print("最新最优验证集结果：")
                for k, v in eval_result.items():
                    print(k + ' =', v)
            if self.is_socket:
                eval_loss = {'epoch_num': epoch_index + 1, 'loss': eval_result['loss']}
                data_dict = list(eval_loss.values())
                data_dict = str(data_dict).encode('utf-8')
                self.socket.send(data_dict)
            # early stopping
            if (self.is_early_stop):
                loss_tolerance = 0.0005
                if eval_result[
                    "loss"] - self.value_judgment >= loss_tolerance:
                    loss_not_decrease_epoch_num += 1
                else:
                    loss_not_decrease_epoch_num = 0
                if loss_not_decrease_epoch_num > 10:
                    print("early stopping 共训练%d个epoch" % epoch_index)
                    break
        if self.is_socket:
            self.socket.close()
        print("training time===============>", time.time() - t1)

    def train(self):
        epoch_loss = 0
        # prefetchers = data_prefetcher(self.train_loader)
        # images, labels = prefetchers.next()
        # iteration = 0
        # while images is not None:
        #     iteration += 1
        #     # 训练代码
        #     output = self.model(images)
        #     output = output.squeeze()
        #     batch_loss = self.criterion(output, labels.long())
        #     self.optimizer.zero_grad()
        #     # 反向传播
        #     batch_loss.backward()
        #     # 只有用了optimizer.step()，模型才会更新
        #     self.optimizer.step()
        #     # epoch_loss += batch_loss.item()
        #     images, labels = prefetcher.next()

        # for images, labels in self.train_loader:
        #     images = torch.from_numpy(images)
            # images = images.transpose((3, 1, 2))  # NHWC -> NCHW
            # labels = torch.from_numpy(labels)
        for i_batch, batch_data in enumerate(self.train_loader):
            images = batch_data['image'].to(device=self.device,
                                            dtype=torch.float32)
            labels = batch_data['label'].to(device=self.device,
                                            dtype=torch.int64)
            # print(images)
            output = self.model(images)
            output = output.squeeze()
            batch_loss = self.criterion(output, labels.long())
            self.optimizer.zero_grad()
            # 反向传播
            batch_loss.backward()
            # 只有用了optimizer.step()，模型才会更新
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        return epoch_loss / len(self.train_loader)

    def create_network(self):
        if 0!=self.global_step:
            bz_log.info("恢复模型...")
            AEnet =  class_model.LW_Classnet(
                               class_num=self.class_num).to(device=self.device)
            AEnet.load_state_dict(torch.load(os.path.abspath(
                     self.model_dir + '/model_'+str(self.global_step)+'.pth')))
        else:
            bz_log.info("从头开始训练...")
            AEnet = class_model.LW_Classnet(
                                class_num=self.class_num).to(device=self.device)
        AEnet.to(device=self.device)
        return AEnet

    def val(self):
        """
        args:
          val_loader: loaded for validation dataset
          model: model
        return: accuracy, recall, precision, loss
        """
        # evaluation mode

        pred_classes = []
        true_labels = []
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data['image'].to(device=self.device,
                                                dtype=torch.float32)
                labels = batch_data['label'].to(device=self.device,
                                                dtype=torch.int64)

                output = self.model(images)
                output = output.squeeze()
                # criterion包含softmax
                batch_loss = self.criterion(output, labels.long())
                epoch_loss += batch_loss.item()
                softmax = torch.softmax(output, dtype=torch.float32, dim=1)
                batch_classes = torch.argmax(softmax,dim=1).cpu().numpy()
                pred_classes = np.append(pred_classes, batch_classes)
                true_labels = np.append(true_labels, labels.cpu().numpy())
            assessment_dict = python_accuracy.get_assessment_result(
                              true_labels, pred_classes, self.class_num,
                                                   'eval', self.assessment_list)
            epoch_loss = epoch_loss / len(self.val_loader)
            assessment_dict.update({'loss': epoch_loss})
        return assessment_dict

    def export_model(self,eval_result, epoch_index):
        """
        :param export_model_dir: export模型路径
        :return:
        """
        best_checkpoint_dir = self.model_dir + '/best_checkpoint_dir/'
        self.__save_best_checkpoint(best_checkpoint_dir, epoch_index)
        with open(self.best_checkpoint_dir + '/best_model_info.json', 'w+') as f:
            eval_result_dict = {k: str(v) for k, v in eval_result.items()}
            json.dump(eval_result_dict, f, indent=1)
        best_checkpoint = glob.glob(best_checkpoint_dir+'*.pth')[0]

        if os.path.exists(self.export_model_dir):
            shutil.rmtree(self.export_model_dir)
        if os.path.exists(self.export_model_dir + '/frozen_model'):
            shutil.rmtree(self.export_model_dir + '/frozen_model')
        os.makedirs(self.export_model_dir + '/frozen_model')
        out_pt_path = self.export_model_dir + '/frozen_model/frozen_model.pt'
        self.__convert_pth_to_pT(best_checkpoint,out_pt_path)
        with open(
           self.export_model_dir + '/frozen_model/model_config.txt', 'w+') as f:
            f.write('model_name:frozen_model.pt' + '\n')
            f.write('input_height:' + str(self.height_width[0]) + '\n')
            f.write('input_width:' + str(self.height_width[1]) + '\n')
            f.write('input_channel:' + str(self.channels_list[0]) + '\n')
            f.write('batch_size:' + str(self.batch_size) + '\n')
            f.write('class_num:' + str(self.class_num))

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
        torch.save(self.model.state_dict(), model_file_name)
        shutil.copy(model_file_name, best_checkpoint_dir)
        os.mkdir(best_checkpoint_dir + '/' + str(epoch_index) + '_parameters')
        with open(best_checkpoint_dir + '/' + str(epoch_index) + '_parameters'
                  + '/lw_classifier_config.json', 'w') as f:
            classifier_config_dict = {k: str(v) for k, v in self.__dict__.items()}
            json.dump(classifier_config_dict, f, indent=1)

    def __convert_pth_to_pT(self,best_checkpoint,out_pt_path):
        example = torch.rand(1, self.channels_list[0], self.height_width[0],
                                                        self.height_width[1])
        net = class_model.LW_Classnet(class_num=self.class_num)
        checkpoint = torch.load(best_checkpoint)
        net.load_state_dict(checkpoint)
        net.eval()
        traced_script_module = torch.jit.trace(net, example)
        traced_script_module.save(out_pt_path)

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
            self.global_step = max(list(
                map(lambda x: int(x.split('_')[-1].split('.pth')[0]),
                    checkpoint_files)))
        else:
            self.global_step = 0


# if __name__=="__main__":
#     # import argparse
#     import sys
#     args = sys.argv
#     print("args=====>",args)  # 通过sys.argv传递参数
#     # parser = argparse.ArgumentParser(description='manual to this script')
#     # parser.add_argument("--data_path", type=str, default="0")
#     # parser.add_argument("--class_num", type=int, default=2)
#     # parser.add_argument("--epoch_num", type=int, default=50)
#     # parser.add_argument("--batch_size", type=int, default=20)
#     # parser.add_argument("--model_dir", type=str, default=None)
#     # parser.add_argument("--is_socket", type=bool, default=False)
#     # parser.add_argument("--is_early_stop", type=bool, default=False)
#     # args = parser.parse_args()
#     # print("args--", args)
#     model = LW_Classification(
#         args.data_path,
#         args.class_num,
#         args.epoch_num,
#         args.batch_size,
#         model_dir=args.all_models_dir,
#         is_socket=args.is_socket,
#         is_early_stop=args.is_early_stop
#     )
#     model.fit()

#


