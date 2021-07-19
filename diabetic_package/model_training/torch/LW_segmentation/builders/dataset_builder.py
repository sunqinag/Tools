import os
import pickle
import shutil
import numpy as np
from diabetic_package.file_operator import bz_path
from torch.utils import data
from diabetic_package.model_training.torch.LW_segmentation.datasets \
    import DataSet, ValDataSet, TrainInform, TestDataSet, create_data




def build_dataset_train(data_dir,
                        classes_num,
                        input_size,
                        batch_size,
                        ignore_label,
                        train_type,
                        random_scale,
                        random_mirror,
                        num_workers):
    inform_data_file = './dataset_inform.pkl'
    if os.path.exists(inform_data_file):
        os.remove(inform_data_file)
    #划分数据集train, val, test
    train_dir = data_dir + "/train/"
    val_dir = data_dir + "/val/"

    train_img_list, train_label_list = create_data(train_dir)
    val_img_list, val_label_list =  create_data(val_dir)

    data_list = np.append(train_img_list, val_img_list)
    label_list = np.append(train_label_list, val_label_list)

    # data_list, label_list
    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))

        # 注意:data_list 和label_list应该给数据划分之后的train + val的
        dataCollect = TrainInform(data_dir, classes_num,
                                        train_set_file=data_list,
                                        label_set_file=label_list,
                                        inform_data_file=inform_data_file)


        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))


    trainLoader = data.DataLoader(
                    DataSet(train_dir,
                            crop_size=input_size,
                            scale=random_scale,
                            mirror=random_mirror,
                            mean=datas['mean'],
                            ignore_label=ignore_label),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)

    valLoader = data.DataLoader(
        ValDataSet(val_dir, f_scale=1, crop_size=input_size,
                         mean=datas['mean'],ignore_label=ignore_label),
                         batch_size=1, shuffle=True, num_workers=num_workers,
                         pin_memory=True)
    return datas, trainLoader, valLoader


def build_dataset_test(data_dir, class_num, num_workers,batch_size,
                       none_gt=False,mode="train"):
    inform_data_file =  './dataset_inform.pkl'
    data_list, label_list = create_data(data_dir)

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))

        dataCollect = TrainInform(data_dir, class_num, train_set_file=data_list,
                                                label_set_file=label_list,
                                                inform_data_file=inform_data_file)
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))


    if mode=="predict":
        testLoader = data.DataLoader(
            TestDataSet(data_dir , mean=datas['mean']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True)
    else:
        testLoader = data.DataLoader(
            ValDataSet(data_dir , mean=datas['mean']),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True)
    return datas, testLoader
