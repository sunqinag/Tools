import os
import numpy as np
import codecs
import cv2
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

saved_path='./VOC2007/'

image_dir='../VOCdevkit/VOC2007/JPEGImages'
image_save_path=saved_path+'JPEGImages/'
#图片存储路径
image_raw_path='../VOCdevkit/VOC2007/JPEGImages/'
# txt格式label存储路径
label_dir='./dpm_label_txt/'

#创建要求文件夹
if not os.path.exists(saved_path+'Annotations'):
    os.makedirs(saved_path+'Annotations')
if not os.path.exists(saved_path+'JPEGImages'):
    os.makedirs(saved_path+'JPEGImages')
if not os.path.exists(saved_path+'ImageSets/Main/'):
    os.makedirs(saved_path+'ImageSets/Main/')



def get_file_path(folder, exts=[], ret_full_path=False):
    '''
        作用:
            获取指定文件夹下所有指定扩展名的文件路径
        参数：
            folder       : 指定文件夹路径
            ret_full_path: 是否返回全路径，默认只返回符合条件的扩展名的文件名
            exts         : 扩展名列表
    '''
    if not (ret_full_path == True or ret_full_path == False):
        raise ValueError('输入参数只能是True或者False')
    if not (os.path.isdir(folder)):
        raise ValueError('输入参数必须是目录或者文件夹')
    if isinstance(exts, str):
        exts = [exts]
    result = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            (file_name, file_ext) = os.path.splitext(f)
            if (file_ext in exts) or (file_ext[1:] in exts) or (len(exts) == 0):
                if ret_full_path:
                    result.append(os.path.join(root, f))
                else:
                    result.append(f)
    return result

def parseBoxAndLabel(label_file):
    '''解析目标检测的label，返回一张图额box和label'''
    print(label_file)
    with open(label_file, 'r') as f:
        labels = f.readlines()
        dst_label = []
        for label in labels:
            label = label.strip().split('_')
            la = [int(float(x)) for x in label]
            dst_label.append(la)
    return dst_label

#获取待处理文件
filename_list = sorted(get_file_path(image_dir,ret_full_path=False))
labelname_list =  sorted(get_file_path(label_dir,ret_full_path=False))
# for i in range(len(filename_list)):
#     print(filename_list[i])
#     print(label_list[i])

# 读取标注信息并写入xml
for filename, labelname in zip(filename_list,labelname_list):
    # embed()
    image = cv2.imread(image_raw_path + filename)
    height, width, channels = cv2.imread(image_raw_path + filename).shape
    # embed()
    label =parseBoxAndLabel(label_dir+labelname)

    # for i in range(len(label)):
    #     p1 = (label[i][0], label[i][1])
    #     p2 = (label[i][2], label[i][3])
    #     cv2.rectangle(image, p1, p2, (0, 255, 0))
    #     p3 = (max(p1[0], 15), max(p1[1], 15))
    #     title = "%s:%d" % ('label::', label[i][-1])
    #     cv2.putText(image, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    # cv2.imshow("view", image)
    # cv2.waitKey(0)

    with codecs.open(saved_path + "Annotations/" + filename.replace(".jpg", ".xml"), "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
        xml.write('\t<filename>' + filename + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The UAV autolanding</database>\n')
        xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>ChaojieZhu</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        if isinstance(label, float):
            ## 空白
            xml.write('</annotation>')
            continue
        for label_detail in label:
            labels = label_detail
            # embed()
            xmin = int(labels[0])
            ymin = int(labels[1])
            xmax = int(labels[2])
            ymax = int(labels[3])
            label_ = labels[-1]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + str(label_) + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(filename, xmin, ymin, xmax, ymax, labels)
        xml.write('</annotation>')

#6.split files for txt
txt_save_path=saved_path+"ImageSets/Main/"
ftrainval = open(txt_save_path+'/trainval.txt', 'w')
ftest = open(txt_save_path+'/test.txt', 'w')
ftrain = open(txt_save_path+'/train.txt', 'w')
fval = open(txt_save_path+'/val.txt', 'w')
total_files = glob(saved_path+"Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]

#test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")

# move images to voc JPEGImages folder
for image in glob(image_raw_path+"/*.jpg"):
    image_name = os.path.split(image)[-1]
    print('copy:',image,'to:',image_save_path+image_name)
    shutil.copy(image,image_save_path+image_name)

train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)

print('开始写入train.txt文件')
for file in train_files:
    ftrain.write(file + "\n")
#val
print('写入完成')
print('开始写入val.txt文件')
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()
#ftest.close()

