# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: YOLOv3_change
File Name: train.py
Author: 孙强
Create Date: 2020/12/25
-------------------------------------------------
"""
import glob
import os
import time
import torch
import socket
import torch.optim as optim
import json
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader

# from tensorboardX import SummaryWriter
from .nets.model_main import ModelMain
from .nets.yolo_loss import YOLOLoss
from .dataset import DataSet
from .cfg import cfg
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Train():

    def __init__(self, data_dir, batch_size, model_dir, epoch_num, class_num, is_early_stop=False, is_socket=False):
        self.is_early_stop = is_early_stop
        self.is_socket = is_socket
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.class_num = class_num
        self.best_checkpoint_dir = os.path.join(model_dir, 'best_checkpoint_dir')
        self.export_model_dir = os.path.join(model_dir, 'export_model_dir')
        self.best_model_info_path = os.path.join(self.best_checkpoint_dir,
                                                 'best_model_info.json')
        train_dir = data_dir + os.sep + 'train_balance'
        val_dir = data_dir + os.sep + 'val_balance'

        if class_num == 1:
            cfg["yolo"]["classes"] = class_num + 1
        cfg["yolo"]["classes"] = class_num

        cfg['epochs'] = epoch_num
        cfg['model_dir'] = model_dir
        if self.is_socket:
            self.build_socket_connect()

        if not os.path.exists(cfg['model_dir']):
            os.mkdir(cfg['model_dir'])
        if not os.path.exists(cfg['log_dir']):
            os.mkdir(cfg['log_dir'])

        start = time.time()

        # 加载网络
        net = ModelMain(cfg)
        net.train(True)

        # 优化器
        optimizer = self.get_optimizer(cfg,net)

        # 数据并行
        net = net.cuda()

        # 模型预加载
        if os.listdir(cfg['model_dir']):
            print('断点重训')
            ckpt_file = glob.glob(cfg['model_dir'] + os.sep + 'model_*.pth')[0]
            state_dict = torch.load(ckpt_file)
            net.load_state_dict(state_dict)

        # YOLO loss with 3 scales
        yolo_losses = []
        for i in range(3):
            yolo_losses.append(YOLOLoss(cfg["yolo"]["anchors"][i],
                                        cfg["yolo"]["classes"], (cfg["img_w"], cfg["img_h"])))

        train_dataset = DataSet(train_dir, img_size=(cfg['img_h'], cfg['img_h']))

        # DataLoader
        train_dataloader = DataLoaderX(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       pin_memory=True,
                                       drop_last=True)

        val_dataset = DataSet(val_dir, img_size=(cfg['img_h'], cfg['img_h']))

        # DataLoader
        val_dataloader = DataLoaderX(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     pin_memory=True,
                                     drop_last=False)

        # tensorboard 可视化
        # cfg["tensorboard_writer"] = SummaryWriter(cfg["log_dir"])

        # 开始训练
        print("Start training.")
        save_value = 9999
        self.global_step = 0
        loss_not_decrease_epoch_num = 0
        for epoch in range(cfg['epochs']):
            self.epoch = epoch
            # train 循环5次进入验证
            for i in range(5):
                train_loss = self.train_op(train_dataloader, optimizer, net,yolo_losses, mode='train')
                print("train_loss", train_loss)
            torch.cuda.empty_cache()

            eval_loss = self.train_op(val_dataloader, optimizer, net,yolo_losses, mode='eval')
            print("eval_loss ", eval_loss)
            if save_value > train_loss:
                save_value = train_loss

                eval_result = {'value_judgment':eval_loss,
                               'global_step':self.global_step}

                self.__save_pth(net) #保存模型

                self.export_model_dir = self.model_dir + '/export_model_dir'
                self.export_model(eval_result=eval_result)


                end = time.time()
                used_time = (end - start) / 60
                # torch.save(net.state_dict(), cfg['model_dir'] + os.sep + 'model.pth')
                with open(cfg['model_dir'] + '/used_time.txt', 'w') as f:
                    f.write(str(format(used_time, '.2f')) + ' mins.\n')

            # ealy stop
            if self.is_early_stop and epoch:
                loss_tolerance = 0.0005
                if eval_loss - save_value >= loss_tolerance:
                    loss_not_decrease_epoch_num += 1
                else:
                    loss_not_decrease_epoch_num = 0
                if loss_not_decrease_epoch_num > 5:
                    print("early stopping 共训练%d个epoch%d", epoch)

            if self.is_socket:
                eval_dict = {
                    'epoch_num': epoch + 1,
                    'loss': eval_loss}
                # eval_result['epoch_num'] = epoch_index + 1
                # eval_result["class_num"] = self.class_num
                data_dict = list(eval_dict.values())
                data_dict = str(data_dict).encode('utf-8')
                self.socket.send(data_dict)

                print("epoch:{}, train_loss={},eval_loss={}".format(epoch, train_loss, eval_loss))


        if self.is_socket:
            self.socket.close()

    def get_optimizer(self, cfg,net):
        # 只冻结主干网络
        base_params = list(map(id, net.backbone.parameters()))
        logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": cfg["lr"]}, ]
        optimizer = optim.Adam(params, weight_decay=cfg["weight_decay"], lr=cfg['lr'])
        return optimizer

    def pth2pt(self, model_dir, cfg):
        model_path = model_dir + os.sep + 'model.pth'
        # 加载网络
        net = ModelMain(cfg)

        model_data = torch.load(model_path)
        net.load_state_dict(model_data)

        net.cuda()
        net.eval()

        traced_script_module = torch.jit.trace(net, torch.ones(1, 3, cfg['img_h'], cfg['img_w']).cuda())
        os.makedirs(model_dir + '/export_model_dir/frozen_model/')
        traced_script_module.save(model_dir + '/export_model_dir/frozen_model/frozen_model.pt')
        print('model frozen 完成!! 保存于 ', model_dir + '/export_model_dir/frozen_model/frozen_model.pt')

        with open(
                model_dir + os.sep + '/export_model_dir/frozen_model/model_config.txt', 'w+') as f:
            f.write('model_name:frozen_model.pt' + '\n')
            f.write('input_height:' + str(cfg['img_h']) + '\n')
            f.write('input_width:' + str(cfg['img_w']) + '\n')
            f.write('input_channel:' + str(self.input_channel) + '\n')
            f.write('batch_size:' + str(self.batch_size) + '\n')
            f.write('class_num:' + str(cfg["yolo"]["classes"]))

    def train_op(self, dataloader, optimizer,net,yolo_losses, mode='train'):
        final_step, total_loss = 1, 0
        for samples in tqdm(dataloader):
            self.global_step += 1
            images, labels = samples["image"].cuda(), samples["label"].cuda()
            self.input_channel = images.shape[1]
            optimizer.zero_grad()
            outputs = net(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]

            losses = []
            for _ in range(len(losses_name)):
                losses.append([])
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]

            loss = losses[0]
            total_loss += loss.cpu().detach().numpy()
            # 训练过程才有反向传播
            if mode == "train":
                loss.backward()
                optimizer.step()

            # loss_ = loss.item()
            # lr = optimizer.param_groups[0]['lr']
            # cfg['tensorboard_writer'].add_scalar(mode + '_lr',
            #                                      lr,
            #                                      self.global_step)

            # for i, name in enumerate(losses_name):
            #     value = loss_ if i == 0 else losses[i]
            #     cfg["tensorboard_writer"].add_scalar(mode + '_' + name,
            #                                          value,
            #                                          self.global_step)
            final_step += 1
        return total_loss / final_step

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

    def export_model(self, eval_result):
        """
        :param export_model_dir: export模型路径
        :return:
        """
        best_checkpoint_dir = self.model_dir + '/best_checkpoint_dir/'
        self.__save_best_checkpoint(best_checkpoint_dir)
        with open(self.best_checkpoint_dir + '/best_model_info.json', 'w+') as f:
            eval_result_dict = {k: str(v) for k, v in eval_result.items()}
            json.dump(eval_result_dict, f, indent=1)
        best_checkpoint = glob.glob(best_checkpoint_dir + '*.pth')[0]

        if os.path.exists(self.export_model_dir):
            shutil.rmtree(self.export_model_dir)
        if os.path.exists(self.export_model_dir + '/frozen_model'):
            shutil.rmtree(self.export_model_dir + '/frozen_model')
        os.makedirs(self.export_model_dir + '/frozen_model')
        out_pt_path = self.export_model_dir + '/frozen_model/frozen_model.pt'
        self.__convert_export_model_to_pb(best_checkpoint, out_pt_path)
        with open(self.export_model_dir + '/frozen_model/model_config.txt', 'w+') as f:
            f.write('model_name:frozen_model.pt' + '\n')
            f.write('input_height:' + str(cfg['img_h']) + '\n')
            f.write('input_width:' + str(cfg['img_w']) + '\n')
            f.write('input_channel:' + str(self.input_channel) + '\n')
            f.write('batch_size:' + str(self.batch_size) + '\n')
            f.write('class_num:' + str(self.class_num))

    def __convert_export_model_to_pb(self,best_checkpoint,out_pt_path):
        example = torch.rand(1, self.input_channel, cfg['img_h'],cfg['img_w']).cuda()
        # 加载网络
        net = ModelMain(cfg).cuda()
        net.load_state_dict(torch.load(best_checkpoint))
        net.eval()
        traced_script_module = torch.jit.trace(net, example)
        traced_script_module.save(out_pt_path)

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
        # latest_epoch_id = self.epoch
        checkpoint_name='model_'+str(latest_epoch_id)+'.pth'
        for path in checkpoint_files:
            if checkpoint_name in path:
                shutil.copy(path, best_checkpoint_dir)
        os.mkdir(best_checkpoint_dir + '/' + checkpoint_name + '_parameters')
        with open(best_checkpoint_dir + '/' + checkpoint_name + '_parameters' +
                                            '/detection_config.json', 'w') as f:
            classifier_config_dict = {k: str(v) for k, v in self.__dict__.items()}
            json.dump(classifier_config_dict, f, indent=1)
            
    def __save_pth(self,net):
        '''模型保存,确保最多保存self.keep_max个模型'''
        torch.save(net.state_dict(), self.model_dir +
                                              f'/model_{self.global_step}.pth')
        # if len(glob.glob(self.model_dir + '/*.pth')) > self.keep_max:
        #     for pth in list(map(lambda x: self.model_dir + '/model_' + str(x) + '.pth',
        #              sorted(list(map(lambda x: int(x.split(
        #             'all_models_dir')[1].split('_')[-1].split('.pth')[0]),
        #             glob.glob(self.model_dir + '/*.pth'))))[:-self.keep_max])):
        #         os.remove(pth)


if __name__ == '__main__':
    data_dir = '/home/xtcsun/PycarmProhects/barcode_调试/out_path'
    Train(
        data_dir=data_dir,
        batch_size=10,
        model_dir='all_models_dir',
        epoch_num=50,
        class_num=4,
    )
