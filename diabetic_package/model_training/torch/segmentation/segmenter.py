# ----------------------------
#!  Copyright(C) 2021
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：刘恩甫,陈瑞侠
#   完成日期：2021-3-4
# -----------------------------
import numpy as np
import os
import shutil
import json
import socket
import glob
import torch
import time
from torch import optim,nn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
# from torch.utils.tensorboard import SummaryWriter

from diabetic_package.log.log import bz_log
from diabetic_package.model_training.torch.segmentation.SegmenterDataset \
                                                        import SegmentDataset
from diabetic_package.model_training.torch.segmentation.SegmenterNet \
    import bz_unet

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Segmenter:
    def __init__(self,
                 class_num,
                 accuracy_weight,
                 transfer_checkpoint_path,
                 class_loss_weight,
                 learning_rate=1e-3,
                 model_dir='./',
                 height_width=(500, 500),
                 channels_list=(3, 1),
                 file_extension_list=('jpg', 'png'),
                 epoch_num=1,
                 eval_epoch_step=1,
                 batch_size=1,
                 keep_max=5,
                 regularizer_scale=1e-7,
                 optimizer_fn=optim.Adam,
                 # assessment_list=['accuracy', 'recall', 'precision'],
                 assessment_list=[],
                 background_and_foreground_loss_weight=(0.4, 0.6),
                 use_background_and_foreground_loss=True,
                 # tensors_to_log=['logits'],
                 tensors_to_log=[],
                 is_socket=False,
                 is_early_stop=True,
                 change_loss_fn_threshold = 0.3,
                 ):
        self.model_dir = model_dir
        self.best_checkpoint_dir = os.path.join(model_dir, 'best_checkpoint_dir')
        self.export_model_dir = os.path.join(model_dir, 'export_model_dir')
        self.best_model_info_path = os.path.join(self.best_checkpoint_dir,
                                                 'best_model_info.json')
        self.height_width = height_width
        self.channels_list = channels_list
        self.file_extension_list = file_extension_list
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.class_num = class_num
        self.accuracy_weight = accuracy_weight
        self.eval_epoch_step = eval_epoch_step
        self.is_socket = is_socket
        self.is_early_stop = is_early_stop
        self.transfer_checkpoint_path = transfer_checkpoint_path
        self.learning_rate = learning_rate
        self.class_loss_weight = class_loss_weight
        self.keep_max = keep_max #保存模型的最大数目
        self.regularizer_scale = regularizer_scale[0]
        self.background_and_foreground_loss_weight = background_and_foreground_loss_weight
        self.use_background_and_foreground_loss = use_background_and_foreground_loss
        self.tensors_to_log = tensors_to_log
        self.change_loss_fn_threshold = change_loss_fn_threshold
        self.__init_value_judgment()
        self.__init_global_step()
        if self.is_socket:
            self.build_socket_connect()

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = self.create_network()
        # self.writer = self.__init_SummaryWriter()
        self.optimizer = optimizer_fn(self.net.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.regularizer_scale)
        self.assessment_list=assessment_list

    def fit(self, train_images_path, train_labels, eval_images_path, eval_labels):
        """
        :param train_images_path: nd.array形式，元素为训练集图片全路径
        :param train_labels: nd.array形式，训练集对应labels
        :param eval_images_path: nd.array形式，元素为验证集图片全路径，交叉验证时为None
        :param eval_labels: nd.array形式，训练集对应labels，交叉验证时为None
        :return:
        """
        change_loss_epoch = round(self.epoch_num * self.change_loss_fn_threshold)
        loss_not_decrease_epoch_num = 0
        train_loader,val_loader=self.create_dataloader(train_images_path,
                                                       train_labels,
                                                       eval_images_path,
                                                       eval_labels)

        for epoch_index in range(0, self.epoch_num, self.eval_epoch_step):
            start_time=time.time()
            #切换前后背景loss为类别加权loss
            if epoch_index > self.epoch_num * self.change_loss_fn_threshold:
                self.use_background_and_foreground_loss = False

            train_epochs = 0
            while train_epochs != self.eval_epoch_step:
                epoch_loss=self.train(train_loader)
                train_epochs += 1
            print("第%d个epoch, train loss: %f" % (epoch_index,
                                                 epoch_loss))

            eval_result=self.eval(val_loader)

            self.__save_pth() #保存模型

            print('\033[1;36m 验证集结果:epoch_index=' + str(epoch_index))
            for k, v in eval_result.items():
                print(k + ' =', v)
            print('\033[0m')
            print("epoch time:",time.time()-start_time)

            if self.is_socket:
                eval_loss = {"epoch_num":epoch_index / self.eval_epoch_step + 1,
                             'loss':eval_result['loss']}
                data_dict = list(eval_loss.values())
                data_dict = str(data_dict).encode('utf-8')
                self.socket.send(data_dict)

            # 模型保存的条件
            saved_model_value = eval_result['loss']
            if saved_model_value < self.value_judgment:
                self.value_judgment = saved_model_value
                eval_result['value_judgment'] = self.value_judgment
                eval_result['global_step']=self.global_step
                self.export_model_dir = self.model_dir + '/export_model_dir'
                self.export_model(eval_result=eval_result)

            # early stopping
            if (self.is_early_stop and epoch_index > change_loss_epoch):
                loss_tolerance = 0.0005
                if eval_result["loss"] - self.value_judgment >= loss_tolerance:
                    loss_not_decrease_epoch_num += 1
                else:
                    loss_not_decrease_epoch_num = 0
                if loss_not_decrease_epoch_num > 5:
                    print("early stopping 共训练%d个epoch" % epoch_index)
                    break

        if self.is_socket:
            # self.socket.send(b'exit')
            self.socket.close()

        # self.writer.close()

    def create_dataloader(self,
                          train_images_path,
                          train_labels,
                          eval_images_path,
                          eval_labels,
                          num_workers=0,
                          pin_memory=True):
        train_dataset = SegmentDataset(train_images_path,
                                       train_labels,
                                       self.height_width)
        val_dataset = SegmentDataset(eval_images_path,
                                     eval_labels,
                                     self.height_width)
        train_loader = DataLoaderX(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last = True)
        val_loader = DataLoaderX(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=False)
        return train_loader,val_loader

    def create_network(self):
        '''bz unet'''
        if 0==self.global_step and self.transfer_checkpoint_path:
            bz_log.info("导入迁移模型...")
            net = bz_unet(self.channels_list[0],2)
            net.load_state_dict(torch.load(self.transfer_checkpoint_path))
            net.outconv=nn.Conv2d(8, self.class_num, 1, 1)
        elif 0!=self.global_step:
            bz_log.info("从恢复点导入模型...")
            net = bz_unet(self.channels_list[0],self.class_num)
            net.load_state_dict(torch.load(os.path.abspath(self.model_dir +
                                       '/model_'+str(self.global_step)+'.pth')))
        else:
            bz_log.info("从scratch开始训练...")
            net = bz_unet(self.channels_list[0],self.class_num)
        net.to(device=self.device)
        return net

    #version 0
    # def train(self,train_loader):
    #     self.net.train() #train mode
    #     epoch_loss=0
    #     for batch in train_loader:
    #         batch_imgs=batch['image'].to(device=self.device, dtype=torch.float32)
    #         mask_type = torch.float32 if self.class_num == 1 else torch.long
    #         batch_true_labels=batch['label'].to(device=self.device, dtype=mask_type)
    #
    #         mask_logits = self.net(batch_imgs)
    #         batch_loss = self.criterion(mask_logits, batch_true_labels) / self.batch_size
    #         # self.writer.add_scalar('Loss/train', batch_loss.item(), self.global_step)
    #         epoch_loss += batch_loss.item()
    #
    #         self.optimizer.zero_grad()
    #         batch_loss.backward()
    #
    #         self.optimizer.step()
    #         self.global_step += self.batch_size
    #     return epoch_loss
    #
    # def eval(self,val_loader):
    #     self.net.eval() #eval mode
    #     epoch_loss,epoch_pred_class,epoch_true_labels = 0,np.array([]),np.array([])
    #
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             batch_imgs=batch['image'].to(device=self.device, dtype=torch.float32)
    #             mask_type = torch.float32 if self.class_num == 1 else torch.long
    #             batch_true_labels=batch['label'].to(device=self.device, dtype=mask_type)
    #
    #             mask_logits = self.net(batch_imgs)
    #             batch_loss = self.criterion(mask_logits, batch_true_labels)/self.batch_size
    #             # self.writer.add_scalar('Loss/eval', batch_loss.item(), self.global_step)
    #             epoch_loss += batch_loss.item()
    #
    #             #以下部分暂不考虑
    #         #     batch_classes=torch.argmax(torch.softmax(logits,
    #         # dtype=torch.float32,dim=1),dim=1).cpu().numpy()
    #         #     if self.tensors_to_log:
    #         #         print("batch验证:(bsoftmax and blabels):")
    #         #         print(batch_classes)
    #         #         print(batch_true_labels.cpu().numpy())
    #         #     epoch_pred_class =np.concatenate((epoch_pred_class,batch_classes))
    #         #     epoch_true_labels=np.concatenate((epoch_true_labels,
    #         # batch_true_labels.cpu().numpy()))
    #         #
    #         # assessment_dict = get_assessment_result(epoch_true_labels,
    #         # epoch_pred_class, self.class_num, 'eval',self.assessment_list)
    #         # assessment_dict.update({'loss':epoch_loss})
    #
    #         assessment_dict={}
    #         assessment_dict['loss']=epoch_loss
    #         return assessment_dict
    #
    # def __init_criterion(self):
    #     if self.use_background_and_foreground_loss :
    #         self.criterion = self.__background_forground_weight_loss()
    #     else:
    #         self.criterion = self.__class_weight_loss()
    #
    # def __background_forground_weight_loss(self):
    #     if len(self.background_and_foreground_loss_weight) != 2:
    #         bz_log.error('classification_weight的长度必须是2！%d',
    #                      len(self.background_and_foreground_loss_weight))
    #         raise ValueError('classification_weight的长度必须是2！')
    #     expand_background_and_foreground_loss_weight = \
    #                              np.ones_like(self.class_loss_weight) * \
    #                              self.background_and_foreground_loss_weight[1]
    #     expand_background_and_foreground_loss_weight[0] = \
    #                                 self.background_and_foreground_loss_weight[0]
    #     return torch.nn.CrossEntropyLoss(
    #         weight=torch.from_numpy(
    #         expand_background_and_foreground_loss_weight).float()).to(self.device)
    #
    # def __class_weight_loss(self):
    #     return torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(
    #                         self.class_loss_weight)).float()).to(self.device)

    #version 1
    def train(self, train_loader):
        self.net.train()  # train mode
        epoch_loss = 0
        for batch in train_loader:
            batch_imgs = batch['image'].to(device=self.device, dtype=torch.float32)
            mask_type = torch.float32 if self.class_num == 1 else torch.long
            batch_true_labels = batch['label'].to(device=self.device, dtype=mask_type)

            mask_logits = self.net(batch_imgs)
            batch_loss = self.__get_loss(mask_logits,batch_true_labels)
            # self.writer.add_scalar('Loss/train', batch_loss.item(), self.global_step)
            epoch_loss += batch_loss.item()
            self.optimizer.zero_grad()
            batch_loss.backward()
            # torch.nn.utils.clip_grad_value_(self.net.parameters(), 0.1)  # 梯度截断
            self.optimizer.step()
            self.global_step += self.batch_size
        return epoch_loss

    def eval(self, val_loader):
        self.net.eval()  # eval mode
        epoch_loss, epoch_pred_class, epoch_true_labels = 0, np.array([]), np.array([])

        with torch.no_grad():
            for batch in val_loader:
                batch_imgs = batch['image'].to(device=self.device, dtype=torch.float32)
                mask_type = torch.float32 if self.class_num == 1 else torch.long
                batch_true_labels = batch['label'].to(device=self.device, dtype=mask_type)

                mask_logits = self.net(batch_imgs)
                batch_loss = self.__get_loss(mask_logits,batch_true_labels)
                # self.writer.add_scalar('Loss/eval', batch_loss.item(), self.global_step)
                epoch_loss += batch_loss.item()

            assessment_dict = {}
            assessment_dict['loss'] = epoch_loss
            return assessment_dict

    def __get_loss(self,logits,labels):
        return self.__background_forground_weight_loss(logits,labels)\
            if self.use_background_and_foreground_loss else \
            self.__class_weight_loss(logits,labels)

    def __background_forground_weight_loss(self,logits,labels):
        if len(self.background_and_foreground_loss_weight) != 2:
            bz_log.error('classification_weight的长度必须是2！%d',
                         len(self.background_and_foreground_loss_weight))
            raise ValueError('classification_weight的长度必须是2！')
        logits,labels=logits.permute(0,2,3,1).contiguous().\
                          view(-1,self.class_num),labels.view(-1)  #由分割向分类的loss形式转换
        background_labels, background_logits, foreground_labels, foreground_logits = \
            labels[labels == 0], logits[labels == 0], \
            labels[labels != 0], logits[labels != 0]
        background_loss, foreground_loss = \
            torch.nn.CrossEntropyLoss()(background_logits, background_labels), \
                torch.nn.CrossEntropyLoss()(foreground_logits, foreground_labels)
        return self.background_and_foreground_loss_weight[0] * background_loss + \
               self.background_and_foreground_loss_weight[1] * foreground_loss

    def __class_weight_loss(self,logits,labels):
        return torch.nn.CrossEntropyLoss(weight=torch.from_numpy(
            np.array(self.class_loss_weight)).float())\
                   .to(self.device)(logits,labels)

    def export_model(self,eval_result):
        """
        :param export_model_dir: export模型路径
        :return:
        """
        best_checkpoint_dir = self.model_dir + '/best_checkpoint_dir/'
        self.__save_best_checkpoint(best_checkpoint_dir)
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
        self.__convert_export_model_to_pb(best_checkpoint,out_pt_path)
        with open(self.export_model_dir + '/frozen_model/model_config.txt', 'w+') as f:
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
            self.global_step = max(list(map(lambda x:
        int(x.split('all_models_dir')[1].split('_')[-1].split('.pth')[0]),
                                                             checkpoint_files)))
        else:
            self.global_step = 0

    def __save_best_checkpoint(self, best_checkpoint_dir):
        """
        :param eval_result: eval模式下输出，字典形式
        :return:
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if (os.path.exists(best_checkpoint_dir)):
            shutil.rmtree(best_checkpoint_dir)
        os.makedirs(best_checkpoint_dir)

        checkpoint_files = glob.glob(self.model_dir+'/*.pth')
        latest_epoch_id=max(list(map(lambda x:int(
        x.split('all_models_dir')[-1].split('_')[-1].split('.pth')[0]),
                                                        checkpoint_files)))
        checkpoint_name='model_'+str(latest_epoch_id)+'.pth'
        for path in checkpoint_files:
            if checkpoint_name in path:
                shutil.copy(path, best_checkpoint_dir)
        os.mkdir(best_checkpoint_dir + '/' + checkpoint_name + '_parameters')
        with open(best_checkpoint_dir + '/' + checkpoint_name + '_parameters' +
                                            '/classifier_config.json', 'w') as f:
            classifier_config_dict = {k: str(v) for k, v in self.__dict__.items()}
            json.dump(classifier_config_dict, f, indent=1)

    def __convert_export_model_to_pb(self,best_checkpoint,out_pt_path):
        example = torch.rand(1, self.channels_list[0], self.height_width[0],
                                                        self.height_width[1])
        net = bz_unet(self.channels_list[0], self.class_num)
        net.load_state_dict(torch.load(best_checkpoint, map_location='cpu'))
        net.eval()
        traced_script_module = torch.jit.trace(net, example)
        traced_script_module.save(out_pt_path)

    def __save_pth(self):
        '''模型保存,确保最多保存self.keep_max个模型'''
        torch.save(self.net.state_dict(), self.model_dir +
                                              f'/model_{self.global_step}.pth')
        if len(glob.glob(self.model_dir + '/*.pth')) > self.keep_max:
            for pth in list(map(lambda x: self.model_dir + '/model_' + str(x) + '.pth',
                     sorted(list(map(lambda x: int(x.split(
                    'all_models_dir')[1].split('_')[-1].split('.pth')[0]),
                    glob.glob(self.model_dir + '/*.pth'))))[:-self.keep_max])):
                os.remove(pth)

    # def __init_SummaryWriter(self):
    #     if glob.glob(self.model_dir+'/events.out.tfevents.*'):
    #         for e in glob.glob(self.model_dir+'/events.out.tfevents.*'):
    #             os.remove(e)
    #     writer=SummaryWriter(log_dir=self.model_dir)
    #     return writer