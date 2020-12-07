* yolov3 dataset解析

  yolov3 dataset部分主要在将原图做出三种尺度上的gt部分，即_processing函数

  ```python
  #首先构建一个label为三种尺度分别为[[52,52,3,5+num_cls],[26,26,3,5+num_cls],[13,13,3,5+num_cls]]
  label = [np.zeros((self.output_size[i], self.output_size[i], self.anchor_per_scale,
                             5 + self.num_classes)) for i in range(3)]
  #然后构建xywh形式的坐标，形式为(self.max_bbox_per_scale, 4)
  bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
  bbox_count = np.zeros((3,))
  
          for bbox in bboxes:
              box_coor = bbox[:4]  # 取x1,y1,x2,y2
              bbox_class_id = bbox[4]  # 取类别序号
  
              # 将类别做onehot编码
              one_hot = np.zeros(self.num_classes, dtype=np.float32)
              one_hot[bbox_class_id] = 1.0  # 将类别索引出置为1
              uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)  # num_classes维，每维以类别总数的倒数填充
              # 这是嫌弃onehot不够平滑吗？？？？
              deta = 0.01
              smooth_onehot = one_hot * (1 - deta) + deta + uniform_distribution
  
              # 根据x1,y1,x2,y2计算得出x,y,w,h，（x,y）为矩形框中心点坐标，来自于原图的尺寸坐标
              bbox_xywh = np.concatenate(((box_coor[:2] + box_coor[2:]) * 0.5, box_coor[2:] - box_coor[:2]), axis=-1)
              # 除以下采样率，对应到特征图上的坐标，包含小中大三个尺寸信息
              bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.stride[:, np.newaxis]
  
              iou = []
              exist_positive = False  # 存在标记框
              for i in range(3):
                  anchors_xywh = np.zeros((self.anchor_per_scale, 4))  # anchor_per_scale每个框产生几个anchor
                  #anchor的中心点也是下采样后feature上的点，可以直接用做xy
                  anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # np.floor()向下取整
                  # anhcor的宽高就用anchor变量的参数来表示，所以会非常小[ 99.5  118.5    1.25   1.25
                  #这种尺度下anchor的框太小了与gt比对iou根本不可能大于0.3
                  anchors_xywh[:, 2:4] = self.anchors[i]  # 获取基准anchor的宽高.这是一个混合体，由中心点坐标和anchor尺寸拼接而成
                  iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  # 计算缩放后的GT与anchor框的IOU
                  iou.append(iou_scale)
                  iou_mask = iou_scale > 0.3
                  if np.any(iou_mask):  # 有1个非0
                      xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)  # 标记框中心坐标
  
                      # # 减少数据敏感性， 9月19号添加，可以打开，不影响结果。
                      xind, yind = abs(xind), abs(yind)
                      if yind >= label[i].shape[1]:  # shape[1] 13，26,52
                          yind = label[i].shape[1] - 1
                      if xind >= label[i].shape[0]:  # shape[0] 13，26,52
                          xind = label[i].shape[0] - 1
  
                      label[i][yind, xind, iou_mask, :] = 0  # 先初始化为0
                      label[i][yind, xind, iou_mask, 0:4] = bbox_xywh  # 标记框的坐标信息
                      label[i][yind, xind, iou_mask, 4:5] = 1.0  # 置信度
                      label[i][yind, xind, iou_mask, 5:] = smooth_onehot  # 分类概率
  
                      bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                      bboxes_xywh[i][bbox_ind, :4] = bbox_xywh  # 第i个box的标记框
                      bbox_count[i] += 1
  
                      exist_positive = True
  			#由于anchor 的客观因素会直接来到这个分支
              if not exist_positive:  # 没有标记框，找iou最大值
                  best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)  # 找iou最大值
                  #label为一个list，三个元素分别为[[52,52,3,5+num_cls],[26,26,3,5+num_cls],[13,13,3,5+num_cls]],
                  #由于尺度为三个三个一组，直接除以尺度数量得到best_detect，这是得到某个尺寸13,26,52上的150个框----->label[best_detect]-->shape:[13,13,3,5+num_cls]
                  best_detect = int(best_anchor_ind / self.anchor_per_scale)
                  #同理得到best_anchor，通过上面得到了某一种尺度下的label-
                  #而同理得到best_anchor,从而得到某种尺度下的某个anchor_scale下的一个框-->label[best_detect][yind, xind, best_anchor, :]
                  best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                  xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
  
                  # 减少数据敏感性 9月19号添加
                  xind, yind = abs(xind), abs(yind)
                  if yind >= label[best_detect].shape[1]:
                      yind = label[best_detect].shape[1] - 1
                  if xind >= label[best_detect].shape[0]:
                      xind = label[best_detect].shape[0] - 1
  
                  label[best_detect][yind, xind, best_anchor, :] = 0
                  label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh  # 标记框坐标
                  label[best_detect][yind, xind, best_anchor, 4:5] = 1.0  # 置信度
                  label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot  # 分类概率
  
                  bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                  bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                  bbox_count[best_detect] += 1
          label_sbbox, label_mbbox, label_lbbox = label  # 获取小中大标记框的标签
          sbboxes, mbboxes, lbboxes = bboxes_xywh  # 获取小中大标记框的坐标值
          return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
  
  
  
  
  
  
  ```

  

