import os
import shutil
import cv2
import numpy as np
from private_tools.file_opter import get_file_path


class Img_processing:
    '''涉及图片的操作'''

    def ChangeImageFormat(self, image_dir, format):
        images = get_file_path(image_dir, ret_full_path=True)

        for i, image in enumerate(images):
            print('第', i, '张')
            name = os.path.split(image)[-1]
            name = name.split('.')[0]
            img = cv2.imread(image)
            print(img.shape)
            cv2.imwrite(image_dir + os.sep + name + format, img)
            os.remove(image)

    def moveImage(self, image_dir, label_dir, target_img_dir, target_label_dir, label_format='.txt'):
        imgs = sorted(get_file_path(image_dir, ret_full_path=True))
        labels = sorted(get_file_path(label_dir, ret_full_path=True))

        for i in range(len(imgs)):
            img_name = os.path.split(imgs[i])[-1]
            label_name = img_name.split('.')[0] + label_format
            print(img_name)
            print(label_name)
            shutil.copy(imgs[i], target_img_dir + os.sep + img_name)
            shutil.copy(labels[i], target_label_dir + os.sep + label_name)

    def deleteMoreImageOrLabel(self, img_dir, label_dir, format='.bmp'):
        '''对图片数量和类别数量不相等的情况，删除多余的文件'''
        images = get_file_path(img_dir, ret_full_path=False)
        labels = get_file_path(label_dir, ret_full_path=False)

        img_names = []
        label_names = []
        for i in range(len(labels)):
            label_name = labels[i].split('.')[0]
            label_names.append(label_name)
        for i in range(len(images)):
            img_name = images[i].split('.')[0]
            img_names.append(img_name)
        if len(images) > len(labels):
            for img in img_names:
                if img not in label_names:
                    os.remove(img_dir + os.sep + img + format)
        for label in label_names:
            if label not in img_names:
                os.remove(label_dir + os.sep + label + '.txt')

    def parseBoxAndLabel(self, label_file):
        '''解析目标检测的label，返回一张图额box和label'''
        with open(label_file, 'r') as f:
            labels = f.readlines()
            boxes = []
            lab = []
            for label in labels:
                label = label.strip().split('_')
                box = [int(float(x)) for x in label[:4]]
                boxes.append(box)
                lab.append(int(float(label[-1])))
        return boxes, lab

    def split_date(self, img_list, label_list, train_folder, test_folder, thresh=0.7):
        if not os.path.exists(train_folder + os.sep + 'img'):
            os.mkdir(train_folder + os.sep + 'img')
        if not os.path.exists(train_folder + os.sep + 'label'):
            os.mkdir(train_folder + os.sep + 'label')
        if not os.path.exists(test_folder + os.sep + 'img'):
            os.mkdir(test_folder + os.sep + 'img')
        if not os.path.exists(test_folder + os.sep + 'label'):
            os.mkdir(test_folder + os.sep + 'label')

        shuffle_index = np.arange(len(img_list))
        np.random.shuffle(shuffle_index)
        train_index = shuffle_index[:int(len(shuffle_index) * thresh)]
        test_index = shuffle_index[int(len(shuffle_index) * thresh):]
        for i in train_index:
            shutil.copy(img_list[i], train_folder + os.sep + 'img')
            shutil.copy(label_list[i], train_folder + os.sep + 'label')
        for i in test_index:
            shutil.copy(img_list[i], test_folder + os.sep + 'img')
            shutil.copy(label_list[i], test_folder + os.sep + 'label')

    def viewBoxOnImage(self, img_dir, label_dir):
        '''将所有目标检测的图片的label全画在图片上'''
        imgs = sorted(get_file_path(img_dir, ret_full_path=True))
        labels = sorted(get_file_path(label_dir, ret_full_path=True))

        for i in range(len(imgs)):
            box, la = self.parseBoxAndLabel(labels[i])
            origimg = cv2.imread(imgs[i], -1)
            for i in range(len(box)):
                p1 = (box[i][0], box[i][1])
                p2 = (box[i][2], box[i][3])
                cv2.rectangle(origimg, p1, p2, (0, 255, 0))
                p3 = (max(p1[0], 15), max(p1[1], 15))
                title = "%s:%d" % ('label::', la[i])
                cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
            cv2.imshow("view", origimg)
            cv2.waitKey(3000)

    def splitClassToSingleFolder(self, img_dir, label_dir, class_num):
        '''讲一个混合数据集的img和label分开成各个文件夹下的img和label'''
        imgs = sorted(get_file_path(img_dir, ret_full_path=True))
        labels = sorted(get_file_path(label_dir, ret_full_path=True))

        for i in range(class_num):
            os.makedirs(str(i) + os.sep + 'img')
            os.makedirs(str(i) + os.sep + 'label')

        for i in range(len(imgs)):
            _, label = Img_processing().parseBoxAndLabel(labels[i])

            for j in [0, 1, 2, 3]:
                if j in label:
                    name = os.path.basename(imgs[i])
                    name = name.split('_')[0]
                    src_img_name = img_dir + '/' + name + '.bmp'
                    src_label_name = label_dir + '/' + name + '.txt'
                    if os.path.exists(src_img_name):
                        print(src_img_name, '>>>>>>>>>>>>>>>>', str(j) + '/img/' + name + '.bmp')
                        shutil.copy(src_img_name, str(j) + '/img/' + name + '.bmp')
                        print(src_label_name, '>>>>>>>>>>>>>>>', str(j) + '/label/' + name + '.txt')
                        shutil.copy(src_label_name, str(j) + '/label/' + name + '.txt')

    def NpyToTxt(self, src_dir, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        npys = get_file_path(src_dir, ret_full_path=True)

        for npy in npys:
            npy_name = os.path.split(npy)[-1]
            npy_name = npy_name.split('.')[0]
            label = np.load(npy)
            with open(dst_dir + os.sep + npy_name + '.txt', 'w') as f:
                for la in label:
                    la = [str(i) for i in la]
                    la = '_'.join(la) + '\n'
                    f.write(la)
            print(dst_dir + os.sep + npy_name + '.txt')

if __name__ == '__main__':
    image_dir = '/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/caffe-jacinto-models/scripts/test/images'
    label_dir = '/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/caffe-jacinto/caffe-jacinto-models/scripts/test/labels'
    tool = Img_processing()
    tool.deleteMoreImageOrLabel(image_dir,label_dir)