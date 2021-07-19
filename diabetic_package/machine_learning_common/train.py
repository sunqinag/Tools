# -*- coding: utf-8 -*-
# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会,陈瑞侠
#   完成日期：2021-x-x
# -----------------------------
import os
import shutil
import filecmp
import glob
import pynvml
import numpy as np
import tensorflow as tf
from diabetic_package.log.log import bz_log
from diabetic_package.file_operator import bz_path
from diabetic_package.data_preprocessing.data_preprocessing import balance_train_val_data
from diabetic_package.model_training.estimator.yolov3_detection.utils import config
from diabetic_package.model_training.estimator.yolov3_detection.model import darknet
from diabetic_package.model_training.estimator.yolov3_detection import yolov3_estimator
from diabetic_package.model_training.estimator.yolov3_detection import detector
from diabetic_package.model_training.estimator.yolov3_detection.dataset.make_labels import (
    make_label_func)
from diabetic_package.model_training.torch.autoencoder.autoencoder import Autoencoder
from diabetic_package.model_training.torch.LW_classification.LW_classification import (
    LW_Classification)
from diabetic_package.model_training.torch.LW_segmentation.LW_segmentation import (
    train_model, parse_args)

from diabetic_package.model_training.torch.classification import classifier
from diabetic_package.model_training.torch.segmentation import segmenter





def train(
    save_model_path,
    epoch_num,
    class_weight,
    class_num,
    balance_num=20,
    data_path="./data",
    task="classification",
    is_socket=0,
    is_early_stop=0,
    out_path="./out_path",
    min_example_num=20,
    out_file_extension_list=["jpg", "png"],
    transfer_checkpoint_path=None,
):
    bz_log.info("start")
    bz_log.info("data_path %s", data_path)
    bz_log.info("save_model_path %s", save_model_path)
    bz_log.info("epoch_num %d", epoch_num)
    for i in range(len(class_weight)):
        bz_log.info("class_weight %d", class_weight[i])
    bz_log.info("class_num %d", class_num)
    bz_log.info("balance_num %d", balance_num)
    bz_log.info("task %s", task)
    if is_socket <= 0:
        is_socket = False
    else:
        is_socket = True

    if is_early_stop == "0":
        is_early_stop = False
    else:
        is_early_stop = True
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory = int(meminfo.free / (1024 * 1024 * 1024))
    bz_log.info("memory %d", memory)

    # 模型保存路径
    all_models_dir = save_model_path + "/all_models_dir"
    best_checkpoint_dir = all_models_dir + "/best_checkpoint_dir"
    export_model_dir = all_models_dir + "/export_model_dir"
    pre_train_chpt_path = "./diabetic_package/pre_train_model"
    # 迁移模型
    if not os.path.exists(all_models_dir):
        os.makedirs(all_models_dir)
        if task == "segmentation":
            transfer_checkpoint_path = (
                pre_train_chpt_path + "/segmentation_pytorch/unet.pth"
            )
            # transfer_checkpoint_path = None
        elif task == "classification":
            # transfer_checkpoint_path = pre_train_chpt_path + "/classfication/model.ckpt-0"
            # transfer_checkpoint_path = None
            # transfer_checkpoint_path = pre_train_chpt_path + "/classification/alexnet.pth"
            transfer_checkpoint_path = pre_train_chpt_path + "/classification/resnet50.pth"
        elif task == "detection":
            transfer_checkpoint_path = (
                pre_train_chpt_path + "/detection/yolov3_coco_demo.ckpt"
            )

    else:
        if not os.path.exists(best_checkpoint_dir) or not os.path.exists(
            export_model_dir
        ):
            shutil.rmtree(all_models_dir)
            os.mkdir(all_models_dir)
            if task == "segmentation":
                # transfer_checkpoint_path = (
                    # pre_train_chpt_path + "/segmentation_pytorch/unet.pth")
                transfer_checkpoint_path = None
            elif task == "classification":
                # transfer_checkpoint_path = pre_train_chpt_path + "/classfication/model.ckpt-0"
                # transfer_checkpoint_path = None
                # transfer_checkpoint_path = pre_train_chpt_path + "/classification/alexnet.pth"
                transfer_checkpoint_path = pre_train_chpt_path + "/classification/resnet50.pth"
            elif task == "detection":
                transfer_checkpoint_path = (
                    pre_train_chpt_path + "/detection/yolov3_coco_demo.ckpt"
                )
        else:
            transfer_checkpoint_path = None
            # if task == "segmentation":
            # # 重复训练的因为右前后背景loss,训练loss会上升,所以关闭early_stop功能
            #     is_early_stop = False

    accuracy_weight = np.ones(class_num).tolist()
    print("accuracy_weight ", accuracy_weight)
    eval_epoch_step = 5
    epoch_num = eval_epoch_step * epoch_num

    if task.lower() != "autoencoder":
        if not os.path.exists(
            os.path.abspath(data_path + "/train/")
        ) or not os.path.exists(os.path.abspath(data_path + "/val/")):
            raise ValueError("No split data...please check!")

        # 如果balance_num等配置和源数据发生变化则删除outpath，注意只判断train数据
        img_list, label_list = bz_path.get_all_subfolder_img_label_path_list(
            data_path + "/train/", ret_full_path=False
        )
        img_list, label_list = sorted(img_list), sorted(label_list)

        # 判断数据复用，如果源数据发生信息变化，则删除out_path文件夹
        if os.path.exists(out_path + "/data_infos.txt"):
            data_info = open(out_path + "/data_infos_new.txt", "w+")
            data_info.write("task=" + task + "\n")
            data_info.write("data_path=" + data_path + "\n")
            data_info.write("balance_num=" + str(balance_num) + "\n")
            data_info.write("img_list=" + str(img_list) + "\n")
            data_info.write("label_list=" + str(label_list) + "\n")
            data_info.close()
            is_same_data = filecmp.cmp(
                out_path + "/data_infos.txt", out_path + "/data_infos_new.txt"
            )
            if not is_same_data:
                shutil.rmtree(out_path)
        else:
            if os.path.exists(out_path):
                shutil.rmtree(out_path)

    if task.lower() == "segmentation":
        batch_size = 6
        bz_log.info(out_path)
        class_num += 1
        if memory >= 12:
            batch_size = 6
        elif memory < 12 and memory >= 10:
            batch_size = 4
        elif memory < 10 and memory >= 8:
            batch_size = 3
        elif memory < 8:
            batch_size = 2
        bz_log.info("class_num %d", class_num)

        if not os.path.exists(out_path):
            bz_log.info("进行数据balance")
            train_img_list, train_label_list, val_img_list, val_label_list = \
                balance_train_val_data(data_path,
                                       out_path,
                                       balance_num,
                                       out_file_extension_list,
                                       task)
            bz_log.info("数据balance完成")
        else:
            bz_log.info("源数据未发生变化，进行数据复用")
            train_img_list, train_label_list = \
                bz_path.get_all_aug_subfolder_img_label_path_list(
                            out_path + "/train_balance/", ret_full_path=True)
            val_img_list, val_label_list = \
                bz_path.get_all_aug_subfolder_img_label_path_list(
                                out_path + "/val_balance/", ret_full_path=True)

        if os.path.exists(out_path + "/data_infos_new.txt"):
            os.remove(out_path + "/data_infos.txt")
            os.rename(out_path + "/data_infos_new.txt", out_path + "/data_infos.txt")
        else:
            # 记录训练信息
            if not os.path.exists(out_path + "/data_infos.txt"):
                data_info = open(out_path + "/data_infos.txt", "w+")
                data_info.write("task=" + task + "\n")
                data_info.write("data_path=" + data_path + "\n")
                data_info.write("balance_num=" + str(balance_num) + "\n")
                data_info.write("img_list=" + str(img_list) + "\n")
                data_info.write("label_list=" + str(label_list) + "\n")
                data_info.close()

        bz_log.info("完成训练集和验证集的加载")
        # 按类别依次排序list
        val_balance_num = len(val_label_list) // (class_num - 1)
        train_img_list, train_label_list, val_img_list, val_label_list = (
                        train_img_list.reshape(-1, balance_num).T.flatten(),
                        train_label_list.reshape(-1, balance_num).T.flatten(),
                        val_img_list.reshape(-1, val_balance_num).T.flatten(),
                        val_label_list.reshape(-1, val_balance_num).T.flatten())

        segmenter_obj = segmenter.Segmenter(
            class_num=class_num,
            model_dir=all_models_dir,
            height_width=(500, 500),
            channels_list=(3, 1),
            file_extension_list=("jpg", "png"),
            accuracy_weight=accuracy_weight,
            batch_size=batch_size,
            epoch_num=epoch_num,
            eval_epoch_step=eval_epoch_step,
            # eval_epoch_step=1,
            is_socket=is_socket,
            is_early_stop=is_early_stop,
            transfer_checkpoint_path=transfer_checkpoint_path,
            learning_rate=0.001,
            class_loss_weight=class_weight,
            regularizer_scale=(0.0000001, 0.0000001),
            background_and_foreground_loss_weight=(0.4, 0.6),
        )

        bz_log.info("网络搭建成功，开始进行训练")
        segmenter_obj.fit(
            train_img_list, train_label_list, val_img_list, val_label_list
        )

    elif task.lower() == "detection":
        batch_size = 6
        if memory >= 12:
            batch_size = 16
        elif memory < 12 and memory >= 10:
            batch_size = 14
        elif memory < 10 and memory >= 8:
            batch_size = 10
        elif memory < 8:
            batch_size = 5

        out_file_extension_list = ["jpg", "npy"]

        if not os.path.exists(out_path):
            bz_log.info("进行数据balance")
            balance_train_val_data(
                data_path, out_path, balance_num, out_file_extension_list, task
            )
            bz_log.info("数据balance完成")

            bz_log.info("开始封装tfrecord")
            # 进行图片归一化,标签的计算
            processed_train_folder, processed_eval_folder = (
                out_path + os.sep + "processed_train",
                out_path + os.sep + "processed_val",
            )
            make_label_func(
                out_path + "/train_balance/",
                out_path + "/val_balance/",
                processed_train_folder,
                processed_eval_folder,
                class_num,
            )
            bz_log.info("封装tfrecord完成")

        if os.path.exists(out_path + "/data_infos_new.txt"):
            os.remove(out_path + "/data_infos.txt")
            os.rename(out_path + "/data_infos_new.txt", out_path + "/data_infos.txt")
        else:
            if not os.path.exists(out_path + "/data_infos.txt"):
                data_info = open(out_path + "/data_infos.txt", "w+")
                data_info.write("task=" + task + "\n")
                data_info.write("data_path=" + data_path + "\n")
                data_info.write("balance_num=" + str(balance_num) + "\n")
                data_info.write("img_list=" + str(img_list) + "\n")
                data_info.write("label_list=" + str(label_list) + "\n")
                data_info.close()

        bz_log.info("开始搭建模型")
        darknet_feature_extractor = darknet.DarkNet(activation=tf.nn.leaky_relu)

        yolo_estimator = yolov3_estimator.YOLOEstimator(
                        model_dir=all_models_dir,
                        feature_extractor=darknet_feature_extractor,
                        combined_feature_map_inds=[-1, -2, -3],
                        grids=config.grids,
                        prior_num_per_cell=config.prior_num_per_cell,
                        class_num=class_num,
                        activation=tf.nn.leaky_relu,
                        transfer_checkpoint_path=transfer_checkpoint_path,
                        assessment_list=config.assessment_list,
                        every_class_weight=class_weight,
                        params={
                            "coord_weight": config.loss_weight[0],
                            "obj_weight": config.loss_weight[1],
                            "cls_weight": config.loss_weight[2],
                            "learning_rate": config.learning_rate
                        })

        detect_instance = detector.detector(estimator_obj=yolo_estimator,
                                            model_dir=all_models_dir,
                                            epoch_num=epoch_num,
                                            batch_size=batch_size,
                                            eval_epoch_step=eval_epoch_step,
                                            class_num=class_num,
                                            is_socket=is_socket,
                                            is_early_stop=is_early_stop)

        bz_log.info("模型初始化成功，开始训练")
        train_records = glob.glob(
            out_path + os.sep + "processed_train" + os.sep + "train*.tfrecords"
        )
        eval_records = glob.glob(
            out_path + os.sep + "processed_val" + os.sep + "eval*.tfrecords"
        )

        detect_instance.fit(train_records, eval_records)

    elif task.lower() == "autoencoder":
        batch_size = 6
        if memory >= 12:
            batch_size = 10
        elif memory >= 10 and memory < 12:
            batch_size = 8
        elif memory < 10 and memory >= 8:
            batch_size = 6
        elif memory < 8:
            batch_size = 4
        autoencoder = Autoencoder(data_dir=data_path,
                                  model_dir=all_models_dir,
                                  epoch_num=int(epoch_num / 5),
                                  batch_size=batch_size,
                                  channels_list=(1,1))
        autoencoder.train()
    # 轻量级网络
    elif task.lower() == "lw_classification":
        batch_size = 6
        if memory >= 12:
            batch_size = 24
        elif memory >= 10 and memory < 12:
            batch_size = 18
        elif memory < 10 and memory >= 8:
            batch_size = 10
        elif memory < 8:
            batch_size = 4
        # bz_log.info(out_path)
        print("batch_size", batch_size)
        # out_file_extension_list = ['png', 'txt']
        # cnn_classify = LW_Classification(model_dir_base=all_models_dir,
        #                          epoch_num= epoch_num,
        #                          channel_num=3,
        #                          batch_size=batch_size,
        #                          class_num=class_num,
        #                          ext=out_file_extension_list)
        # # train
        # cnn_classify.train_cnn_classifier(data_path+'/train/', data_path+'/val/')
        model = LW_Classification(
            data_path,
            class_num,
            int(epoch_num / 5),
            batch_size,
            model_dir=all_models_dir,
        )
        model.fit()

    # 轻量级网络
    elif task.lower() == "lw_segmentation":
        batch_size = 15
        if memory >= 12:
            batch_size = 32
        elif memory >= 10 and memory < 12:
            batch_size = 25
        elif memory < 10 and memory >= 8:
            batch_size = 20
        elif memory < 8:
            batch_size = 4

        args = parse_args()
        socket = 0
        args.input_size = "256,256"
        args.ignore_label = args.classes
        args.dataset = data_path
        args.batch_size = batch_size
        args.max_epochs = epoch_num
        args.classes = class_num
        args.savedir = all_models_dir
        args.is_socket = socket
        train_model(args)

    elif task.lower() == "classification":
        bz_log.info(out_path)
        if memory >= 12:
            batch_size = 24
        elif memory == 10:
            batch_size = 24
        elif memory == 9:
            batch_size = 16
        elif memory == 8:
            batch_size = 10
        elif memory < 8:
            batch_size = 4
        out_file_extension_list = ["jpg", "txt"]

        if not os.path.exists(out_path):
            bz_log.info("进行数据balance")
            train_img_list, train_label_file_list, val_img_list, val_label_file_list \
                = balance_train_val_data(data_path,
                                         out_path,
                                         balance_num,
                                         out_file_extension_list,
                                         task=task)
            bz_log.info("数据balance完成")
        else:
            bz_log.info("源数据未发生变化，进行数据复用")
            train_img_list, train_label_file_list = \
                bz_path.get_all_aug_subfolder_img_label_path_list(
                            out_path + "/train_balance/", ret_full_path=True)
            val_img_list, val_label_file_list = \
                bz_path.get_all_aug_subfolder_img_label_path_list(
                            out_path + "/val_balance/", ret_full_path=True)

        # 记录数据信息
        if os.path.exists(out_path + "/data_infos_new.txt"):
            os.remove(out_path + "/data_infos.txt")
            os.rename(out_path + "/data_infos_new.txt", out_path + "/data_infos.txt")
        else:
            if not os.path.exists(out_path + "/data_infos.txt"):
                data_info = open(out_path + "/data_infos.txt", "w+")
                data_info.write("task=" + task + "\n")
                data_info.write("data_path=" + data_path + "\n")
                data_info.write("balance_num=" + str(balance_num) + "\n")
                data_info.write("img_list=" + str(img_list) + "\n")
                data_info.write("label_list=" + str(label_list) + "\n")
                data_info.close()

        # 按类别依次排序list
        val_balance_num = len(val_label_file_list) // class_num
        train_img_list, train_label_list, val_img_list, val_label_list = (
            train_img_list.reshape(-1, balance_num).T.flatten(),
            train_label_file_list.reshape(-1, balance_num).T.flatten(),
            val_img_list.reshape(-1, val_balance_num).T.flatten(),
            val_label_file_list.reshape(-1, val_balance_num).T.flatten(),
        )

        classifier_obj = classifier.Classifier(class_num=class_num,
                                                       model_dir=all_models_dir,
                                                       height_width=(256, 256),
                                                       channels_list=[3],
                                                       file_extension_list=("jpg",),
                                                       epoch_num=epoch_num,
                                                       eval_epoch_step=eval_epoch_step,
                                                       batch_size=batch_size,
                                                       accuracy_weight=accuracy_weight,
                                                       is_socket=is_socket,
                                                       is_early_stop=is_early_stop,
                                                       transfer_checkpoint_path=transfer_checkpoint_path,
                                                       learning_rate=0.001,
                                                       label_weight=class_weight)

        bz_log.info("网络搭建成功，开始进行训练")
        classifier_obj.fit(
            train_img_list, train_label_list, val_img_list, val_label_list)

    bz_log.info("生成训练模型")
    frozen_model_dir = all_models_dir + "/export_model_dir/frozen_model/"
    if task.lower() == "lw_segmentation" or task.lower() == 'lw_classification' :
        lw_model_dir = save_model_path + "lw_model"
        if not os.path.exists(lw_model_dir):
            os.makedirs(lw_model_dir)
        shutil.copy(frozen_model_dir + "/frozen_model.pt", lw_model_dir)
        shutil.copy(frozen_model_dir + "/model_config.txt", lw_model_dir)
    else:
        checkpoint_files = bz_path.get_file_path(frozen_model_dir, ret_full_path=True)
        for path in checkpoint_files:
            shutil.copy(path, save_model_path)
    bz_log.info("网络训练结束！")


if __name__ == "__main__":
    import time

    start_time = time.time()
    f = open("train_config.txt")
    isweight = False
    class_weight = []
    # task = None
    task_type = None
    for line in f.readlines():
        line = line.strip()
        values = line.split("-")
        if not isweight:
            if values[0] == "data_path":
                data_path = values[1]
            elif values[0] == "model_dir":
                model_dir = values[1]
            elif values[0] == "class_num":
                class_num = int(values[1])
            elif values[0] == "balance_num":
                balance_num = int(values[1])
            elif values[0] == "epoch_num":
                epoch_num = int(values[1])
            elif values[0] == "task":
                task = values[1]
            elif values[0] == "class_weight":
                isweight = True
            elif values[0] == "early_stop":
                is_early_stop = values[1]
        else:
            class_weight = np.append(class_weight, int(values[0]))
    f.close()

    class_weight = class_weight.tolist()

    train(
        model_dir,
        epoch_num,
        class_weight,
        class_num,
        balance_num,
        data_path,
        task,
        is_socket=0,
        is_early_stop=is_early_stop,
    )

    print("训练时间为:", time.time() - start_time)
