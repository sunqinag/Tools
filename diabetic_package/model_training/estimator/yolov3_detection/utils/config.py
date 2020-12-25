img_shape = (416, 416, 3)
grids = [52,26,13]
strides=[8,16,32]
max_bbox_per_scale=150
iou_loss_thresh=0.5
prior_num_per_cell=3
learning_rate=1e-4
loss_weight=[1,1,1] #coord,obj,cls
# assessment_list=['recall','precision','iou',]
assessment_list=['accuracy', 'iou', 'recall', 'precision', 'auc']

#(416,416)时的锚点设计
prior_anchors=[
    [[10,13],[16,30],[33,23]],#小物体
    [[30,61],[62,45],[59,119]],
    [[116,90],[156,198],[373,326]]] #大物体
#按strides归一化的anchors
anchors=[
    [[1.25,1.625],[2.,3.75],[4.125,2.875]],
    [[1.875,3.8125],[3.875,2.8125],[3.6875,7.4375]],
    [[3.625,2.8125],[4.875,6.1875],[11.65625,10.1875]]]

#模型迁移
exclude=['global_step','conv_lbbox/weight','conv_lbbox/bias','conv_mbbox/weight',
         'conv_mbbox/bias','conv_sbbox/weight','conv_sbbox/bias','beta1_power','beta2_power',]

id2char={0:'person', 1: 'bird', 2: 'cat', 3: 'cow', 4: 'dog', 5: 'horse', 6: 'sheep',
         7: 'aeroplane', 8: 'bicycle', 9: 'boat', 10: 'bus', 11: 'car', 12: 'motorbike',
         13: 'train', 14: 'bottle', 15: 'chair', 16: 'diningtable', 17: 'pottedplant',
         18: 'sofa', 19: 'tvmonitor'}