import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data
import pickle
from diabetic_package.file_operator import bz_path

def create_data(data_dir):
    folders = np.sort(bz_path.get_subfolder_path(data_dir, ret_full_path=True, is_recursion=False))
    feature_set = []
    labels_set = []
    for i, file in enumerate(folders):
        print(file)
        img_paths = np.sort(bz_path.get_file_path(file + "/img", ret_full_path=True))
        label_paths =  np.sort(bz_path.get_file_path(file + "/label", ret_full_path=True))
        feature_set = np.append(feature_set, img_paths)
        labels_set = np.append(labels_set,label_paths)
    return feature_set, labels_set

class DataSet(data.Dataset):
    """ 
       DataSet is employed to load train set
       Args:
        root: the dataset path,
        list_path: train data dir path
    """
    def __init__(self, data_dir='', max_iters=None, crop_size=(512, 512),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=0):
        # self.root = root
        self.data_dir = data_dir
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.files = []
        # img_list = np.sort(bz_path.get_file_path(data_dir + "/img/", ret_full_path=True))
        # label_list = np.sort(bz_path.get_file_path(data_dir + "/label/", ret_full_path=True))
        img_list, label_list = create_data(data_dir)

        for img, label in zip(img_list, label_list):
            name = img + " " + label
            self.files.append({
                "img": img,
                "label": label,
                "name": name
            })
        print("length of train set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        # if self.scale:
        #     scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # random resize between 0.5 and 2
        #     f_scale = scale[random.randint(0, 5)]
        #     # f_scale = 0.5 + random.randint(0, 15) / 10.0  #random resize between 0.5 and 2
        #     image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        #     label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)

        # image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        # image = cv2.resize(image,(self.crop_h, self.crop_w),cv2.INTER_NEAREST)
        # label = cv2.resize(label,(self.crop_h, self.crop_w),cv2.INTER_NEAREST)
        image = image.transpose((2, 0, 1))  # NHWC -> NCHW

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class ValDataSet(data.Dataset):
    """ 
       ValDataSet is employed to load val set
       Args:
        root: the dataset path,
        list_path: val path

    """
    def __init__(self, data_dir='',
                 f_scale=1, mean=(128, 128, 128),crop_size=(512,512),ignore_label=2):
        # self.root = root
        self.list_path = data_dir
        self.ignore_label = ignore_label
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.f_scale = f_scale
        self.files = []

        self.files = []
        # img_list = np.sort(bz_path.get_file_path(data_dir + "img/", ret_full_path=True))
        # label_list = np.sort(bz_path.get_file_path(data_dir + "label/",
        #                                    ret_full_path=True))

        img_list, label_list = create_data(data_dir)

        for img, label in zip(img_list, label_list):
            name = img + " " + label
            self.files.append({
                "img": img,
                "label": label,
                "name": name
            })
        print("length of Validation set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        # if self.f_scale != 1:
        #     image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation=cv2.INTER_LINEAR)
            # label = cv2.resize(label, None, fx=self.f_scale, fy=self.f_scale, interpolation = cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)

        # image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # revert to RGB

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        # image = cv2.resize(image,(self.crop_h, self.crop_w),cv2.INTER_NEAREST)
        # label = cv2.resize(label,(self.crop_h, self.crop_w),cv2.INTER_NEAREST)
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        # print('image.shape:',image.shape)
        return image.copy(), label.copy(), np.array(size), name


class TestDataSet(data.Dataset):
    """ 
       TestDataSet is employed to load test set
       Args:
        root: the dataset path,
        list_path: test path

    """

    def __init__(self, data_dir='',
                 mean=(128, 128, 128), ignore_label=2):
        # self.root = root
        self.list_path = data_dir
        self.ignore_label = ignore_label
        self.mean = mean
        # self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        self.files = []
        # img_list = np.sort(bz_path.get_file_path(data_dir + "/img/", ret_full_path=True))
        # label_list = np.sort(bz_path.get_file_path(data_dir + "/label/",
        #                                    ret_full_path=True))
        img_list, label_list = create_data(data_dir)

        for img, label in zip(img_list, label_list):
            name = img.strip().split('/', 1)[1].split("/")[-1].split(".")[0]

            self.files.append({
                "img": img,
                "label": label,
                "name": name
            })
        print("lenth of test set ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"],cv2.IMREAD_GRAYSCALE)

        image_org = image
        name = datafiles["name"]

        image = np.asarray(image, np.float32)

        size = image.shape

        # image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB
        mean_size = int((size[0] + size[1]) / 2)
        if mean_size < 512:
            image = cv2.resize(image,(512,512),interpolation=cv2.INTER_NEAREST)
        elif mean_size > 512 and mean_size < 1024 and (mean_size - 512) < 200:
            image = cv2.resize(image, (512, 512),
                               interpolation=cv2.INTER_NEAREST)
        elif mean_size > 512 and mean_size < 1024 and (mean_size - 512) > 200:
            image = cv2.resize(image, (1024, 1024),
                               interpolation=cv2.INTER_NEAREST)
        elif mean_size > 1024 and mean_size < 2048 and (mean_size - 1024) < 500:
            image = cv2.resize(image, (1024, 1024),
                               interpolation=cv2.INTER_NEAREST)
        else:
            image = cv2.resize(image, (2048, 2048),
                               interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2, 0, 1))  # HWC -> CHW
        # label = cv2.resize(label, (512, 512),interpolation=cv2.INTER_LINEAR)

        return image_org, image.copy(), label.copy(), np.array(size), name


class TrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """
    def __init__(self, data_dir='', classes=2, train_set_file="",label_set_file="",
                 inform_data_file="", normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.train_set_file = train_set_file
        self.label_set_file = label_set_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        # self.classWeights = [1,3]
        # weights = [1.0,3.0]
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))
            # self.classWeights[i] = weights[i]

    def readWholeTrainSet(self, fileName, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data
        
        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        for img_file, label_file in  zip(self.train_set_file, self.label_set_file):
            # we expect the text file to contain the data in following format
            # <RGB Image> <Label Image>
            label_img = cv2.imread(label_file, 0)
            unique_values = np.unique(label_img)
            max_val = max(unique_values)
            min_val = min(unique_values)

            max_val_al = max(max_val, max_val_al)
            min_val_al = min(min_val, min_val_al)

            if train_flag == True:
                hist = np.histogram(label_img, self.classes, [0, self.classes - 1])
                global_hist += hist[0]

                rgb_img = cv2.imread(img_file)
                self.mean[0] += np.mean(rgb_img[:, :, 0])
                self.mean[1] += np.mean(rgb_img[:, :, 1])
                self.mean[2] += np.mean(rgb_img[:, :, 2])

                self.std[0] += np.std(rgb_img[:, :, 0])
                self.std[1] += np.std(rgb_img[:, :, 1])
                self.std[2] += np.std(rgb_img[:, :, 2])

            else:
                print("we can only collect statistical information of train set, please check")
            if max_val > (self.classes - 1) or min_val < 0:
                print('Labels can take value between 0 and number of classes.')
                print('Some problem with labels. Please check. label_set:', unique_values)
                print('Label Image ID: ' + label_file)
            no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(fileName=self.train_set_file)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None
