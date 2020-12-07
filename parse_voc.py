# -*- coding: utf-8 -*-
# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：刘恩甫
#   完成日期：2019-x-x
# -----------------------------

import cv2
import os
import xml.dom.minidom
import numpy as np

char2id={'0':0,'1':1,'2':2,'3':3,'4':4
         }

if __name__ == '__main__':
    from private_tools.imgFileOpterator import Img_processing
    img_path= '/media/bzjgsq/68C63D57C63D26AA/dm_dpm_data/dpmImg/'
    anno_path= '/media/bzjgsq/68C63D57C63D26AA/dm_dpm_data/dpm_label/'
    imagelist=os.listdir(img_path)
    label_save_path= 'dpm_label_npy/'
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)

    for image in imagelist:
        # image='000050.jpg'
        print(image)
        image_pre, ext = os.path.splitext(image)
        img_file = img_path + image
        img = cv2.imread(img_file)
        xml_file = anno_path + image_pre + '.xml'
        DOMTree = xml.dom.minidom.parse(xml_file)
        collection = DOMTree.documentElement
        objects = collection.getElementsByTagName("object")

        all_labels=[]
        for object in objects:
            bndbox = object.getElementsByTagName('bndbox')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin_data = xmin.childNodes[0].data
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin_data = ymin.childNodes[0].data
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax_data = xmax.childNodes[0].data
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax_data = ymax.childNodes[0].data
            xmin = int(xmin_data)
            xmax = int(xmax_data)
            ymin = int(ymin_data)
            ymax = int(ymax_data)
            name_obj = object.getElementsByTagName('name')[0]
            name_char = name_obj.childNodes[0].data
            name_id = char2id[name_char]
            all_labels.append([xmin,ymin,xmax,ymax,name_id])

        print(all_labels)
        np.save(label_save_path+image.split('.')[0]+'.npy',all_labels)
    Img_processing().NpyToTxt(label_save_path, 'dpm_label_txt')


