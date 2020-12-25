import tensorflow as tf
import numpy as np

from ..utils import config


def __bytes_feature(value):
    '''要进行tostring(),value两边需要[]'''
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def __float_feature(value):
    '''注意没有value两边没有[], 且数据一定要flatten()'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_yolo_example(img, label):
    '''
        用于生成写入tfrecord文件的example字符串
    '''
    img_raw = img.tobytes()

    label_sbbox = label[0].astype(np.float32)
    label_mbbox = label[1].astype(np.float32)
    label_lbbox = label[2].astype(np.float32)
    sbboxes = label[3].astype(np.float32)
    mbboxes = label[4].astype(np.float32)
    lbboxes = label[5].astype(np.float32)

    feature = {
        'img': __bytes_feature(img_raw),
        'label_sbbox': __float_feature(label_sbbox.reshape(-1)),
        'label_mbbox': __float_feature(label_mbbox.reshape(-1)),
        'label_lbbox': __float_feature(label_lbbox.reshape(-1)),
        'sbboxes': __float_feature(sbboxes.reshape(-1)),
        'mbboxes': __float_feature(mbboxes.reshape(-1)),
        'lbboxes': __float_feature(lbboxes.reshape(-1))
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_yolo_example(example_proto,class_num):
    '''
        将tfrecord文件中存储的example解析成img和yolo模型使用的label
    '''
    feature_description = {
        'img': tf.FixedLenFeature([],tf.string),
        # 'img': tf.FixedLenFeature([config.img_shape[0], config.img_shape[1], config.img_shape[2]], tf.float32),
        'label_sbbox': tf.FixedLenFeature([config.grids[0], config.grids[0], config.prior_num_per_cell,5 + class_num], tf.float32),
        'label_mbbox': tf.FixedLenFeature([config.grids[1], config.grids[1], config.prior_num_per_cell,5 + class_num], tf.float32),
        'label_lbbox': tf.FixedLenFeature([config.grids[2], config.grids[2], config.prior_num_per_cell,5 + class_num], tf.float32),
        'sbboxes': tf.FixedLenFeature([config.max_bbox_per_scale, 4], tf.float32),
        'mbboxes': tf.FixedLenFeature([config.max_bbox_per_scale, 4], tf.float32),
        'lbboxes': tf.FixedLenFeature([config.max_bbox_per_scale, 4], tf.float32)}

    features = tf.parse_single_example(example_proto, feature_description)

    img_string = features['img']
    img_tensor = tf.decode_raw(img_string, out_type=tf.uint8)
    img_tensor = tf.cast(tf.reshape(img_tensor, [config.img_shape[0], config.img_shape[1], config.img_shape[2]]),tf.float32)

    label_sbbox_tensor = features['label_sbbox']
    label_mbbox_tensor = features['label_mbbox']
    label_lbbox_tensor = features['label_lbbox']
    sbboxes_tensor = features['sbboxes']
    mbboxes_tensor = features['mbboxes']
    lbboxes_tensor = features['lbboxes']

    return img_tensor,label_sbbox_tensor,label_mbbox_tensor,label_lbbox_tensor,\
           sbboxes_tensor,mbboxes_tensor,lbboxes_tensor

def serialize_yolo_example_bytesbak(img, label):
    '''
        用于生成写入tfrecord文件的example字符串
    '''
    img = img.astype(np.float32)

    label_sbbox = label[0].astype(np.float32)
    label_mbbox = label[1].astype(np.float32)
    label_lbbox = label[2].astype(np.float32)
    sbboxes = label[3].astype(np.float32)
    mbboxes = label[4].astype(np.float32)
    lbboxes = label[5].astype(np.float32)

    feature = {
        'img': __bytes_feature(img.tobytes()),
        'label_sbbox': __bytes_feature(label_sbbox.tostring()),
        'label_mbbox': __bytes_feature(label_mbbox.tostring()),
        'label_lbbox': __bytes_feature(label_lbbox.tostring()),
        'sbboxes': __bytes_feature(sbboxes.tostring()),
        'mbboxes': __bytes_feature(mbboxes.tostring()),
        'lbboxes': __bytes_feature(lbboxes.tostring()),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_yolo_example_bytesbak(example_proto,class_num):
    '''
        将tfrecord文件中存储的example解析成img和yolo模型使用的label
    '''
    feature_description = {
        'img': tf.FixedLenFeature([], tf.string),
        'label_sbbox': tf.FixedLenFeature([], tf.string),
        'label_mbbox': tf.FixedLenFeature([], tf.string),
        'label_lbbox': tf.FixedLenFeature([], tf.string),
        'sbboxes': tf.FixedLenFeature([], tf.string),
        'mbboxes': tf.FixedLenFeature([], tf.string),
        'lbboxes': tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(example_proto, feature_description)

    img_string = features['img']
    label_sbbox_string = features['label_sbbox']
    label_mbbox_string = features['label_mbbox']
    label_lbbox_string = features['label_lbbox']
    sbboxes_string = features['sbboxes']
    mbboxes_string = features['mbboxes']
    lbboxes_string = features['lbboxes']


    img_tensor=tf.decode_raw(img_string,out_type=tf.float32)
    img_tensor=tf.reshape(img_tensor,[config.img_shape[0], config.img_shape[1], config.img_shape[2]])

    label_sbbox_tensor=tf.decode_raw(label_sbbox_string,out_type=tf.float32)
    label_sbbox_tensor=tf.reshape(label_sbbox_tensor,[config.grids[0], config.grids[0], config.prior_num_per_cell,5 + class_num])

    label_mbbox_tensor = tf.decode_raw(label_mbbox_string, out_type=tf.float32)
    label_mbbox_tensor = tf.reshape(label_mbbox_tensor,[config.grids[1], config.grids[1], config.prior_num_per_cell,5 + class_num])

    label_lbbox_tensor = tf.decode_raw(label_lbbox_string, out_type=tf.float32)
    label_lbbox_tensor = tf.reshape(label_lbbox_tensor,[config.grids[2], config.grids[2], config.prior_num_per_cell,5 + class_num])

    sbboxes_tensor = tf.decode_raw(sbboxes_string, out_type=tf.float32)
    sbboxes_tensor = tf.reshape(sbboxes_tensor,[config.max_bbox_per_scale, 4])

    mbboxes_tensor = tf.decode_raw(mbboxes_string, out_type=tf.float32)
    mbboxes_tensor = tf.reshape(mbboxes_tensor,[config.max_bbox_per_scale, 4])

    lbboxes_tensor = tf.decode_raw(lbboxes_string, out_type=tf.float32)
    lbboxes_tensor = tf.reshape(lbboxes_tensor,[config.max_bbox_per_scale, 4])

    return img_tensor,label_sbbox_tensor,label_mbbox_tensor,label_lbbox_tensor,\
           sbboxes_tensor,mbboxes_tensor,lbboxes_tensor
