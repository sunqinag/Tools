import tensorflow as tf

from .model import yolo_network
from .model.feature_extractor import IFeatureExtractor
from .utils.vis_utils import vis_func
from diabetic_package.log.log import bz_log
import diabetic_package.model_training.estimator.yolov3_detection.utils.config as yolo_config
from diabetic_package.model_training.estimator.yolov3_detection.loss.losses import YoloLoss
from diabetic_package.machine_learning_common.accuracy import tensorflow_accuracy

class YOLOEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir,
                 feature_extractor,
                 grids,
                 prior_num_per_cell,
                 combined_feature_map_inds,
                 class_num,
                 activation,
                 assessment_list,
                 every_class_weight,
                 transfer_checkpoint_path=None,
                 config=None,
                 params=None,
                 warm_start_from=None):
        '''
        model_dir:
            checkpoint存储路径
        feature_extractor:
            用于提取特征的IFeatureExtractor实例
        filters_for_combined_features:
            对多尺度融合的feature map进行卷积的卷积层输出通道数
        grids:
            分块数目
        prior_num_per_cell:
            每个grid cell prior的数量
        combined_feature_map_inds:
            feature map extractor返回的所有尺度的feature map中用于预测的
            feature map的索引，combined feature map都会被resize到
            combined_feature_map_inds指定的最后一维feature map的大小
        class_num:
            objectness的分类个数
        activation:
            激活函数
        output_layer_type:
            输出层的处理方式，包括:
                global_average_pooling, global_average_pooling_conn,conv
        config:
            estimator运行的配置
        every_class_weight:
            对各类别的分类loss进行加权
        params:
            模型训练需要的超参，包括：
            coord_weight——bbox回归loss的权重
            obj_weight——objectness loss的权重
            cls_weight——分类loss的权重
            learning_rate——学习率
        warm_start_from:
            模型加载的checkpoint的路径
        '''
        super().__init__(self.__model_fn,
                         model_dir,
                         config,
                         params,
                         warm_start_from)

        if not isinstance(feature_extractor, IFeatureExtractor):
            bz_log.error('feature_extractor不是IFeatureExtractor的实例')
            raise ValueError('feature_extractor不是IFeatureExtractor的实例')

        self.feature_extractor = feature_extractor
        self.grids = grids
        self.prior_num_per_cell = prior_num_per_cell
        self.combined_feature_map_inds = combined_feature_map_inds
        self.class_num = class_num
        self.activation = activation
        self.transfer_checkpoint_path=transfer_checkpoint_path
        self.assessment_list=assessment_list
        self.every_class_weight=every_class_weight
        self.strides=yolo_config.strides
        self.anchors=yolo_config.anchors
        self.compute_loss=YoloLoss

    def __model_fn(self, features, labels, mode, params, config):
        training = mode == tf.estimator.ModeKeys.TRAIN
        imgs = features['img']

        #经过网络，得到最后的预测值，self.pred_sbbox，self.pred_mbbox，self.pred_lbbox
        self.network(imgs,training)

        if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
            #获取标签
            self.label_sbbox, self.label_mbbox, self.label_lbbox,\
                                    self.true_sbboxes, self.true_mbboxes,self.true_lbboxes=\
            labels['label_sbbox'],labels['label_mbbox'],labels['label_lbbox'],\
                                    labels['sbboxes'],labels['mbboxes'],labels['lbboxes']

            #计算loss
            loss_dict = self.compute_loss(\
                grids=self.grids,
                class_num=self.class_num,
                every_class_weight=self.every_class_weight,
                coord_weight=params['coord_weight'],
                obj_weight=params['obj_weight'],
                cls_weight=params['cls_weight'],
                conv_sbbox=self.conv_sbbox,
                pred_sbbox=self.pred_sbbox,
                conv_mbbox=self.conv_mbbox,
                pred_mbbox=self.pred_mbbox,
                conv_lbbox=self.conv_lbbox,
                pred_lbbox=self.pred_lbbox)(self.label_sbbox,#labels
                                            self.label_mbbox,
                                            self.label_lbbox,
                                            self.true_sbboxes,
                                            self.true_mbboxes,
                                            self.true_lbboxes)
            loss = loss_dict['loss']
        else:
            loss = None
            loss_dict = {}

        #[cx,cy,w,h,conf,one_hot]
        pred_sbbox,pred_mbbox,pred_lbbox=tf.reshape(self.pred_sbbox,[-1,yolo_config.grids[0]**2*yolo_config.prior_num_per_cell,5+self.class_num]),\
                                         tf.reshape(self.pred_mbbox,[-1,yolo_config.grids[1]**2*yolo_config.prior_num_per_cell,5+self.class_num]),\
                                         tf.reshape(self.pred_lbbox,[-1,yolo_config.grids[2]**2*yolo_config.prior_num_per_cell,5+self.class_num])

        concat_node = tf.concat([pred_sbbox, pred_mbbox, pred_lbbox], axis=-2) #[?,10647,5+cls_num]
        pred_xywh,pred_conf,pred_prob=tf.split(concat_node,[4,1,self.class_num],axis=-1)

        #(cx,cy,w,h)->(xmin, ymin, xmax, ymax)
        pred_coor = tf.concat([pred_xywh[..., :2] - pred_xywh[..., 2:] * 0.5,
                               pred_xywh[..., :2] + pred_xywh[..., 2:] * 0.5], axis=-1)

        class_id = tf.expand_dims(tf.cast(tf.argmax(pred_prob, axis=-1),dtype=tf.float32),axis=-1)
        class_prob=tf.expand_dims(tf.reduce_max(pred_conf*pred_prob, axis=-1),axis=-1)
        output_node=tf.concat([pred_conf,class_prob,class_id,pred_coor],axis=-1,name='outputs')
        predictions={'predict_results':output_node} #输出节点

        if mode == tf.estimator.ModeKeys.TRAIN:
            # # 训练过程的metrics可视化
            # assessment_dict = tensorflow_accuracy.get_assessment_result(
            #     tf.cast(labels_selected[:, 5], dtype=tf.int32),  # label类别
            #     tf.cast(logits_selected[:, 2], dtype=tf.int32),  # logits类别
            #     self.class_num, 'train', self.assessment_list)
            # for key, value in assessment_dict.items():
            #     tf.summary.scalar(key, value[1])

            # 训练过程可视化
            draw_box_result=tf.py_func(vis_func,[features['img'],
                                                 self.pred_sbbox,#[b,52,52,3,5+cls_num]
                                                 self.pred_mbbox,
                                                 self.pred_lbbox,
                                                 self.class_num
                                                 ],[tf.uint8],name='train_draw_box')
            tf.summary.image('show_train_process_image',draw_box_result[0],max_outputs=3)

            #模型迁移
            if self.transfer_checkpoint_path:
                print('Loading transfer parameters...')
                variables_to_restore = tf.contrib.framework.get_variables_to_restore(\
                    exclude=yolo_config.exclude)
                tf.train.init_from_checkpoint(
                    self.transfer_checkpoint_path,
                    {v.name.split(':')[0]:v for v in variables_to_restore})

            #fine-tune
            train_vars = list(self.var_filter(tf.trainable_variables()))
            train_op = tf.train.AdamOptimizer(
                params['learning_rate']).minimize(loss, tf.train.get_global_step(),var_list=train_vars)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op, update_ops])
        else:
            train_op = None

        if mode == tf.estimator.ModeKeys.EVAL:
            logits_ = tf.reshape(output_node, [-1, 7])  # [?,7]
            labels_raw = tf.reshape(tf.concat([tf.reshape(self.label_sbbox, [-1, yolo_config.grids[0] ** 2 * yolo_config.prior_num_per_cell, 5 + self.class_num]), \
                                               tf.reshape(self.label_mbbox, [-1, yolo_config.grids[1] ** 2 * yolo_config.prior_num_per_cell, 5 + self.class_num]), \
                                               tf.reshape(self.label_lbbox, [-1, yolo_config.grids[2] ** 2 * yolo_config.prior_num_per_cell, 5 + self.class_num])], \
                                              axis=-2), [-1, 5 + self.class_num])  # [?,25]
            labels_split = tf.split(labels_raw, [5, self.class_num], axis=-1)
            labels_clsid = tf.reshape(tf.cast(tf.argmax(labels_split[1], axis=-1), dtype=tf.float32), [-1, 1])
            labels_ = tf.concat([labels_split[0], labels_clsid], axis=-1)  # [?,6]
            # 选择参与计算的前景的cls值
            conf_idx = tf.reshape(tf.where(tf.equal(labels_[:, 4:5], 1)), [-1])
            logits_selected, labels_selected = tf.gather(logits_, conf_idx), tf.gather(labels_, conf_idx)
            #针对分类的评估方法
            assessment_result = tensorflow_accuracy.get_assessment_result\
                (tf.cast(labels_selected[:,5],dtype=tf.int32),#label类别
                 tf.cast(logits_selected[:,2], dtype=tf.int32),#logits类别
                 self.class_num, mode=mode,assessment_list=self.assessment_list)

            # 针对回归的metrics, mean_squared_error
            #logits_selected coor:[xmin,ymin,xmax,ymax]
            #labels_selected coor:[cx,cy,w,h]->[xmin,ymin,xmax,ymax]
            label_calc_boxes=tf.concat([labels_selected[..., :2] - labels_selected[..., 2:4] * 0.5,
                       labels_selected[..., :2] + labels_selected[..., 2:4] * 0.5], axis=-1)
            box_error= {'bbox_error': tf.metrics.mean_squared_error(\
                                        label_calc_boxes, logits_selected[:,3:])}
            assessment_result.update(box_error)

            eval_metric_ops = assessment_result
        else:
            eval_metric_ops = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                (tf.saved_model.signature_constants.
                 DEFAULT_SERVING_SIGNATURE_DEF_KEY):
                tf.estimator.export.PredictOutput(predictions)
            }
        else:
            export_outputs = None

        logging_hook = tf.train.LoggingTensorHook(
            tensors=loss_dict,
            every_n_iter=10)

        return tf.estimator.EstimatorSpec(
            mode,
            predictions,
            loss,
            train_op,
            eval_metric_ops,
            export_outputs,
            training_hooks=[logging_hook]
        )

    def network(self,imgs,training):
        '''yolo_v3的网络结构'''
        self.conv_lbbox,self.conv_mbbox,self.conv_sbbox=yolo_network.YoloNetwork(self.feature_extractor)(
            imgs,
            self.grids,
            self.prior_num_per_cell,
            self.combined_feature_map_inds,
            self.class_num,
            self.activation,
            training)

        with tf.variable_scope('pred_sbbox'): #52
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[1], self.strides[0])
        with tf.variable_scope('pred_mbbox'): #26
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])
        with tf.variable_scope('pred_lbbox'): #13
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def decode(self, conv_output, anchors, stride):
        """
        return [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               包含(x, y, w, h, score, probability)
        """
        conv_shape=tf.shape(conv_output)
        batch_size=conv_shape[0]
        output_size=conv_shape[1]
        anchor_per_scale=len(anchors)

        conv_output=tf.reshape(conv_output,\
                               (batch_size,output_size,output_size,anchor_per_scale,5+self.class_num))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh,pred_conf,pred_prob], axis=-1)

    def var_filter(self,var_list):
        '''实现做微调,选择进行训练的参数'''
        filter_keywords = ['darknet']
        for var in var_list:
            kw = filter_keywords[0]
            if kw not in var.name:
                yield var
            else:
                continue