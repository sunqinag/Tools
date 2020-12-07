YOLO V3 检测主干部分：

* 由于生成anchor和比对过程由dataset完成，dataset返回为【(3, 416, 416, 3)，(3, 52, 52, 3, 10)，(3, 26, 26, 3, 10)，(3, 13, 13, 3, 10)，(3, 150, 4)，(3, 150, 4)，(3, 150, 4)）因此，yolo主干便可以非常简洁

* 首先通过直通的卷积得到三种尺度的feature 

  ```Python
      def __build_nework(self, input_data):
          #(1, 52, 52, 256),(1, 26, 26, 512),(1, 13, 13, 512)
          route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)
  
          input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
          input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
          input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
          input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
          input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')
  		
          #得到定位分支和分类分支
          conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')#(1, 13, 13, 1024)
          conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                            trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)#(1, 13, 13, 30)
  
          input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
          input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)
  		#这一步通过上采样13-->26得到FPN结构，再加上与route_2做特征融合更有效果
          with tf.variable_scope('route_1'):
              input_data = tf.concat([input_data, route_2], axis=-1)
  
          input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
          input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
          input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
          input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
          input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')
  
          conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')#(1, 26, 26, 512)
          conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                            trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)#(1, 26, 26, 30)
  
          input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
          input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)
  
          with tf.variable_scope('route_2'):
              input_data = tf.concat([input_data, route_1], axis=-1)
  
          input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
          input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
          input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
          input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
          input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')
  
          conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')#(1, 52, 52, 256)
          conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                            trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)#(1, 52, 52, 30)
  
          return conv_lbbox, conv_mbbox, conv_sbbox
  ```

* 拿到直筒的结果之后要进行decode，

```python
    def decode(self, conv_output, anchors, stride):
        """
        conv_output:(1, 52, 52, 30)
        anchors:(3, 2)
        stride:8
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)
		
        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))
        
        #以上就是做了一个reshape，tensor并无实质上的变化
		#使用dxdy是因为取值范围还是有限的，没有到达图片尺寸的大小
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]
        
		#画一个与conv尺寸相等的矩阵，x画为（52,52），y画为（52,52）
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        
		#然后通过扩维将xy拼接起来达到52,52,2，注意这些值是从0到51
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        
        #  然后继续复制扩维到与conv_output同维度（1,52,52,3,2）
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        
		#通过公式计算算出原图尺寸下的坐标位置，注意，这里xy_gtid就相当于是偏移量了
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
```



* 下面要计算loss，计算loss很简单，就是在三中尺度上用网络给出的结果和label比对，计算出三中尺度的loss

  ```python
  计算loss的核心是loss_layer函数：
      def loss_layer(self, conv, pred, label, bboxes, anchor, stride):
          '''
          :param conv: 分类分支得到的conv_sbbox，(1, 52, 52, 30（3*（5+num_class））)
          :param pred: 定位分支得到的pred_sbbox,(150,4)
          :param label: gt给出的分类分支的label_sbbox，(1, 52, 52, 30)
          :param bboxes: gt给出的定位分支的true_sbbox,(150,4)
          :param anchor:
          :param stride:
          :return:
          '''
          conv_shape = tf.shape(conv)
          batch_size = conv_shape[0]
          output_size = conv_shape[1]
          input_size = stride * output_size
          conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                   self.anchor_per_scale, 5 + self.num_class))
  
          #经过network输出的分类的置信度和概率
          conv_raw_conf = conv[:,:,:,:,4:5]
          conv_raw_prob=conv[:,:,:,:,5:]
  
          #network输出的定位的坐标和置信度
          pred_xywh = pred[:, :, :, :, 0:4]
          pred_conf = pred[:, :, :, :, 4:5]
  
          #label标签中给的坐标，置信度和概率
          label_xywh = label[:, :, :, :, 0:4]
          respond_bbox = label[:, :, :, :, 4:5]
          label_prob = label[:, :, :, :, 5:]
  
  
          giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
          input_size = tf.cast(input_size, tf.float32)
  
          #这是对gt的box置信度做了压缩（压缩到0-1之间）
          bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
  
          giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
  
          iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
          max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
  
          respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)
  
          #focal loss参照知乎博客https://zhuanlan.zhihu.com/p/80594704，这里得到一个惩罚权重系数，fical loss用来解决one stage网络的正负样本问题
          conf_focal = self.focal(respond_bbox, pred_conf)
  
          #？？？？？？？？？？？？？？这是怎么计算置信度loss的？？？？
          conf_loss = conf_focal * (
                  respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                  +
                  respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
          )
  
          #类别概率的loss
          prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
  
          giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
          conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
          prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
  
          return giou_loss, conf_loss, prob_loss
  ```

  