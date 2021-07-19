import tensorflow as tf
import os,cv2,random
import numpy as np
from diabetic_package.file_operator import bz_path
from diabetic_package.image_processing_operator.python_data_augmentation import python_base_data_augmentation

class Dataset:
    def __init__(self,image_dir,image_size,batch_size,num_epoch):
        self.image_dir = image_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch

    def get_next(self):

        #dropped by enfu
        # image_list = bz_path.get_file_path(self.image_dir, exts=['.jpg'], ret_full_path=True)
        # choice_ind = np.random.choice(
        #     np.arange(len(image_list)),
        #     len(image_list),
        #     replace=False)
        # image_list = np.array(image_list)
        # image_list = image_list[choice_ind]
        # train_num = int(len(image_list) * 0.8)
        # train_data_list = image_list[:train_num]
        # val_data_list = image_list[train_num:]
        # train_dataset = self.creat_dataset(train_data_list)
        # val_dataset = self.creat_dataset(val_data_list)

        #revised by enfu
        train_data_list=bz_path.get_file_path(self.image_dir+'/train',exts=['.jpg'], ret_full_path=True)
        val_data_list = bz_path.get_file_path(self.image_dir+'/val',exts=['.jpg'], ret_full_path=True)
        train_dataset = self.creat_dataset(train_data_list)
        val_dataset = self.creat_dataset(val_data_list)

        return train_dataset, val_dataset

    def creat_dataset(self,data_list):
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.read_image, num_parallel_calls=4)

        dataset = dataset.batch(self.batch_size, drop_remainder=True).shuffle(
            buffer_size=len(data_list)).repeat()
        dataset = dataset.prefetch(buffer_size=10 * self.batch_size)
        iterator = dataset.make_one_shot_iterator()

        batch_data = iterator.get_next()

        return batch_data

    def pyfunc(self,image,size):
        # image = cv2.imread(image.decode('ascii'),0)
        image = cv2.imdecode(np.fromfile(image,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, tuple(size))
        # if random.randint(0,1) > 0.5:
        #     image = python_base_data_augmentation.random_rotate_image_and_label(image,min_angle=-20,max_angle=20)
        # if random.randint(0, 1) > 0.5:
        #     image = python_base_data_augmentation.random_translation_image_and_label(image,max_dist=0.3)
        if random.randint(0, 1) > 0.5:
            image = np.uint8(python_base_data_augmentation.add_noise(image, ['gaussian'], 0.0001, 0.0001)[0] * 255)
        return image

    def read_image(self,image):
        image = tf.py_func(self.pyfunc,inp=[image,self.image_size],Tout=[tf.uint8])
        image = tf.squeeze(image,axis=0)
        image = tf.expand_dims(image,axis=-1)
        return {'img':image}

