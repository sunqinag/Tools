import tensorflow as tf
import numpy as np

from diabetic_package.model_training.estimator.yolov3_detection.utils import config

class YoloLoss:
    def __init__(
        self, grids,class_num,every_class_weight,coord_weight, obj_weight,cls_weight,
        conv_sbbox,pred_sbbox,conv_mbbox,pred_mbbox,conv_lbbox,pred_lbbox
    ):
        self.grids = grids
        self.num_class=class_num
        self.every_class_loss_weight=every_class_weight
        self.coord_weight = coord_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.conv_sbbox=conv_sbbox
        self.pred_sbbox=pred_sbbox
        self.conv_mbbox=conv_mbbox
        self.pred_mbbox=pred_mbbox
        self.conv_lbbox=conv_lbbox
        self.pred_lbbox=pred_lbbox
        self.anchors = config.anchors
        self.strides = config.strides
        self.iou_loss_thresh = config.iou_loss_thresh

    def __call__(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        with tf.name_scope('smaller_box_loss'):#[52,52]
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])
        with tf.name_scope('medium_box_loss'):#[26,26]
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])
        with tf.name_scope('bigger_box_loss'):#[13,13]
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]
        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]
        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        tf.summary.scalar('loss_coord', giou_loss)
        tf.summary.scalar('loss_conf', conf_loss)
        tf.summary.scalar('loss_cls', prob_loss)

        total_loss=self.coord_weight * giou_loss + self.obj_weight * conf_loss + self.cls_weight * prob_loss
        return {'loss': total_loss,
                'loss_coord': self.coord_weight * giou_loss,
                'loss_conf': self.obj_weight * conf_loss,
                'loss_cls': self.cls_weight * prob_loss}

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv,(batch_size, output_size, output_size,config.prior_num_per_cell, 5+self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]#[?,52,52,3,4]
        respond_bbox = label[:, :, :, :, 4:5]#[?,52,52,3,1],控制只要前景
        label_prob = label[:, :, :, :, 5:]#[?,52,52,3,20]

        #coord_loss
        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)
        #bbox_loss_scale:对于一个grid来说,w*h/(52**2),有归一化三个不同尺度的效果
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4]\
                          / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        # confidence loss
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], \
                            bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        #计算当前grid的单个box与标签中的150个box的交并比
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        #选择参与计算的背景框
        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )
        #计算置信度加权系数
        conf_focal = self.focal(respond_bbox, pred_conf)
        conf_loss = conf_focal *(
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)+
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        #分类 loss,带每类加权,respond_bbox用来控制只要前景的
        prob_loss = respond_bbox * self._get_class_weighted_loss(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
        return giou_loss, conf_loss, prob_loss

    def focal(self, target, actual, alpha=1, gamma=2):
        '''alpha*(target - actual)^gamma'''
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):
        '''计算每个pred grid和label grid 之间的iou'''

        #[cx,cy,w,h]->[xmin,ymin,xmax,ymax]
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        #为了加坐标保证,保证xmin>xmax,ymin<ymax
        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
        #计算每个grid交并比
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        #enclose:计算并的信息
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        #这句是giou的精华所在,但感觉跟bbox_iou作用是一样的
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
        return giou

    def bbox_iou(self, boxes1, boxes2):
        '''计算当前grid的单个box与标签中的150个box的交并比'''
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area
        return iou

    def _get_class_weighted_loss(self,labels,logits):
        """
        对每个类别进行loss加权
        :param logits: 预测结果
        :param labels: 标签
        :param loss_weight: loss权重
        :return:
        """
        weights = self.__cal_loss_weight(labels)
        class_weight_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=logits)*weights[:, :, :, :, tf.newaxis]
        return class_weight_loss

    def __cal_loss_weight(self, labels):
        """
        :param labels: 标签,one_hot
        :param weight: 每一个类别的权重
        :return:
        """
        label_shape = tf.shape(tf.argmax(labels,axis=-1))
        weight = tf.zeros(label_shape)
        loss_weight = tf.zeros(label_shape)
        for c in range(self.num_class): #对于每类标签
            classes_loss_weight = tf.where(
                tf.equal(tf.argmax(labels,axis=-1),c),
                tf.multiply(tf.ones(label_shape), self.every_class_loss_weight[c]),
                weight)
            loss_weight = tf.add_n([loss_weight, classes_loss_weight])
        return loss_weight