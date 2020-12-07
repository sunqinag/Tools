'''
requirement:
    torchvision:0.4.2
    torch:1.3.1
    numpy:1.15
    cv2:4.1.2.30
'''
import torchvision
import torch
import numpy as np
import cv2
import torch.nn as nn

'''生成一个dummy image'''
dummy_image = torch.zeros((1, 3, 800, 800)).float()
# print('dummy_image:', dummy_image)
labels = torch.LongTensor([6, 8, 9])

'''列出vgg16所有的层'''
model = torchvision.models.vgg16(pretrained=True)
# pre=torch.load('/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/pycharm_project/Faster_rcnn/vgg16-397923af.pth')
# model = model.load_state_dict(pre)
fe = list(model.features)
# print(fe)

# 将图像传输通过所有层，确定得到相应的尺寸
req_features = []
k = dummy_image.clone()
for i in fe:
    k = i(k)
    if k.size()[2] < 800 // 16:
        break
    req_features.append(i)
    out_channels = k.size()[1]
# print(len(req_features))  # 30
# print(out_channels)  # 512

# 将list转换为Sequential module
faster_rcnn_fe_extractor = torch.nn.Sequential(*req_features)
# 现在faster_rcnn_fe_extractor可以作为后端，计算特征：
out_map = faster_rcnn_fe_extractor(dummy_image)
print('out_map size:', out_map.size())

'''
    Anchor Boxes
    过程分解：
    1.在一个feature map的坐标上生成Anchor
    2.在所有feature map的坐标上生成Anchor
    3.对每个目标分配标签及坐标（相当于anchor）
    4.在feature mao坐标生成Anchor
    *将采用anchor_scales=8,16,32 ratios=0.5,1,2 sub_sampling=16(因为我们将图像从800像素池化至50像素，输出的feature map的每个像素对应原像素中16*16像素)
    *首先需要在这16*16像素上生成锚框，然后沿着x轴和y轴进行类似的操作以获得所有的anchor boxes，这个将在步骤2中完成。
    *feature map的每个像素位置生成9个anchor boxes（anchor_scales的数量和ratios的数量），每个anchor box具有’y1,x1,y2,x2‘因此每个位置anchor会有形状（9,4），
    开始为一个空的全0的数组
'''
ratios = [0.5, 1, 2]
anchor_sacles = [8, 16, 32]
sub_sampling = 16
anchor_base = np.zeros((len(ratios) * len(anchor_sacles), 4), dtype=np.float32)

print('anchor base:', anchor_base)
'''
    我们用相应的y1,x1,y2,x2填充这些值，这个基础anchor的中心将在
'''
ctr_y = sub_sampling / 2
ctr_x = sub_sampling / 2

print('ctr_y , ctr_y', ctr_x, ctr_y)

for i in range(len(ratios)):
    for j in range(len(anchor_sacles)):
        h = sub_sampling * anchor_sacles[j] * np.sqrt(ratios[i])
        w = sub_sampling * anchor_sacles[j] * np.sqrt(1. / ratios[i])

        index = i * len(anchor_sacles) + j

        anchor_base[index, 0] = ctr_y - h / 2
        anchor_base[index, 1] = ctr_x - w / 2
        anchor_base[index, 2] = ctr_y + h / 2
        anchor_base[index, 3] = ctr_x + w / 2
print('anchor_base：', anchor_base)
'''
    这些是第一个feature mao像素的Anchor位置，现在必须在feature mao的所有位置生成这些anchor，还要注意，negitive值表示anchor boxes在图像维度之外，
    在后面的部分中用-1来标记，斌该计算函数损失和声场anchor建议时删除它们。而且由于在每个位置都有9个anchor，并且在一个图像中有50*50个这样的位置，共得到17500（50*50*9）个anchor
    现在在所有特征图位置生成anchor
    为实现这一目标首先要为每个feature map生成像素中心（复原到原始图像的位置点）
'''
fe_size = (800 // 16)
ctr_x = np.arange(16, (fe_size + 1) * 16, 16)
ctr_y = np.arange(16, (fe_size + 1) * 16, 16)
ctr = np.zeros((len(ctr_y) * len(ctr_x), 2), dtype=np.int32)

# 找到中心并进行中心可视化
image = np.zeros((801, 801), dtype=np.float32)
cut = 0
for x in ctr_x:
    for y in ctr_y:
        image[x, y] = 255
        cut += 1
print('共：', cut)
cv2.imwrite('anchor中心点.png', image)

# 采用Python生成中心
index = 0
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index, 1] = ctr_x[x] - 8
        ctr[index, 0] = ctr_y[y] - 8
        index += 1
'''
    输出将是每个未知的(x,y)值，如anchor中心店可视化所示，共2500个锚点，现在需要在每个中心生成anchor boxes，可以使用在一个而位置生成anchor的代码来完成，
    为提供每个anchor中心的代码添加一个提取for循环就可以了
'''
# image = np.zeros((801, 801), dtype=np.float32)
# for c in ctr:
#     ctr_y, ctr_x = c
#     image[ctr_x, ctr_y] = 255
# cv2.imwrite('检查提取中心点.png', image)
# 检查完成，anchor匹配正常
anchors = np.zeros((fe_size * fe_size * 9, 4), dtype=np.float32)
index = 0
for c in ctr:
    ctr_y, ctr_x = c
    for i in range(len(ratios)):
        for j in range(len(anchor_sacles)):
            h = sub_sampling * anchor_sacles[j] * np.sqrt(ratios[i])
            w = sub_sampling * anchor_sacles[j] * np.sqrt(1. / ratios[i])

            anchors[index, 0] = ctr_y - h / 2
            anchors[index, 1] = ctr_x - w / 2
            anchors[index, 2] = ctr_y + h / 2
            anchors[index, 3] = ctr_x + w / 2
            index += 1
print('anchors.shape:', anchors.shape)

'''
    下一步是分配目标标签和位置给每个anchor，在算法中有些分配标签的指导原则：
    a.与gt重叠度最高的iou的anchor
    b.与gt的iou大于0.7的anchor
    注意：单个gt对象可以为多个anchor对象分配标签
    c.对所有与gt的iou小于0.3的anchor标记为副标签
    d.anchor既不是正标签也不是副标签，对样本没有帮助，舍弃
    通过如下方式对anchor boxes分配表前和位置
        1.找到有效的anchor boxes的索引，并升恒索引数组，生成标签数组其形状索引数组填充-1（无效anchor boxes。对应在边框外的anchor boxes）
        2.检查是否满足以上三条中的一条，并填写标签。如果是正anchor boxes标签为1，注意那个gt目标可以得到这个结果
        3.计算与anchor box相关的gt位置loc
        3.通过为所有无效的anchor box填充-1和为所有有效锚箱计算的值重新组织所有锚箱
        4.输出应该是（N,1）数组的标签和带有(N,4)数组的loc
'''

# 找到所有有效anchor boxes的索引
inside_index = np.where(
    (anchors[:, 0] >= 0) &
    (anchors[:, 1] >= 0) &
    (anchors[:, 2] <= 800) &
    (anchors[:, 3] <= 800)
)[0]
# 生成空的标签数组，大小为inside_index,填充-1，默认值设置为0
label = np.empty(len(inside_index, ), dtype=np.int32)
label.fill(-1)
print('label shape:', label.shape)

# 生成有效anchor boxes数组
valid_anchor_boxes = anchors[inside_index]
print('valid_anchor_boxes_shape：', valid_anchor_boxes.shape)

# 对每个有效anchor box计算与每个gt目标的iou，因为有8940个anchor boxes,和2个gt目标，应该计算得到（8940,2）的数组作为输出，
'''
    两个框之间计算iou的代码逻辑如下
    -find the max of x1 and y1 in both the boxes (xn1,yn1)
    -find the min of x2 and y2 in both the boxes (xn2,yn2
    -now both the boxes ate intersecting only 
    if (xn1 < xn2) and (yn1<yn2)
        -iou_areas will be (xn2-xn1)*(yn2-yn1)
    else
        -iou_areas will be 0
    -similary calculate areas for anchor box and ground truth object
    -iou = iou_aea/(anchor_box_areas+ground_truth_areas - iou_areas)
'''
bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32)
ious = np.empty((len(valid_anchor_boxes), 2), dtype=np.float32)  # 这里2代表的是有多少不同类别的框。
ious.fill(0)
print('bbox：', bbox)
for num1, i in enumerate(valid_anchor_boxes):
    ya1, xa1, ya2, xa2 = i
    anchor_areas = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_areas = (yb2 - yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_areas = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)  # 这里了经常写错，注意注意～～
            iou = iter_areas / (anchor_areas + box_areas - iter_areas)
        else:
            iou = 0
        ious[num1, num2] = iou
print('ious.shape：', ious.shape)

'''
    考虑到1和2的情况，需要找到两件事：
    1.每个gt_box及对应的anchor box的最高iou
    2.每个anchor box及对应的gt box的最高iou
'''
# case-1
gt_argmax_ious = ious.argmax(axis=0)
print('gt_argmax_ious：', gt_argmax_ious)
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
print('gt_max_ious：', gt_max_ious)
# case-2
argmax_ious = ious.argmax(axis=1)
print('argmax_ious shape：', argmax_ious.shape)
print('argmax_ious：', argmax_ious)
max_ious = ious[np.arange(ious.shape[0]), argmax_ious]
print('max_ious：', max_ious)

# 找到具有max_ious的anchor_boxes(gt_max_ious)
gt_argmax_ious = np.where(ious == gt_max_ious)[0]
print('通过where找到gt_argmax_ious ：', gt_argmax_ious)

'''
    至此得到三个数组：
        -argmax_ious----确定哪个gt目标与每个anchor都有最大的iou
        -max_ios--------确定gt目标与每个anchor的max iou
        -gt_argmax_ious-确定与gt box有最大的iou重叠的anchor
    使用argmax_ious和max_ios可以班组bc的anchorbox分配标签和位置，
    使用gt_argmax_ious可以满足a为anchor box分配标签和位置
'''
pos_iou_threshold = 0.7
neg_iou_threshold = 0.3

# 分配负标签0给max iou小于负阈值 条件c 的所有anchor boxes
label[max_ious < neg_iou_threshold] = 0

# 分配正标签1给gt box 条件a 的iou重叠最大的anchor box
label[gt_argmax_ious] = 1

# 分配正标签1给max iou大于正阈值 条件b 的anchor boxes
label[max_ious >= pos_iou_threshold] = 1

'''
    下面进入训练 RPN，Faster R_CNN文章描述如下：
        每个batch都来自包含许多正样本和负样本anchor的单个图像，但这偏向于负样本，因为他们占据主导地位。相反，随机采样图像中的256个anchor来计算batch的损失函数，
        其中被采样的正锚点和负锚点的比例达到1:1。如果一幅图像中有少于128个正样本我们就用负样本填充batch图像，便有下面两个变量
'''
pos_ratio = 0.5
n_sample = 256
# 所有正样本
n_pos = pos_ratio * n_sample

'''现在要从正标签中随机采样n_pos个样本，忽略-1剩余的样本，得到少于n_pos个样本，此时随机采样n_sample-n_pos个负样本 0 忽略剩余的anchor box
    实现如下：
'''
# positive sample
pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
    label[disable_index] = -1

# negative sample
n_neg = n_sample - np.sum(label == 1)
neg_index = np.where(label == 0)[0]
if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
    label[disable_index] = -1

'''
    下面进入anchor box定位阶段
    现在用具有最大iou的gt对象为每个anchor box分配位置，注意，将为所有有效的anchor box分配anchor locs，而不考虑其标签，在后面计算损失时可以使用简单的过滤器筛除
    
    已知与每个anchor box具有高iou的gt目标，现在我们需要找到gt相对于anchor box的坐标，文章中按照如下参数化：
    t_{x}=(x - x_{a})/w_{a}
    t_{y}=(y - y_{a})/h_{a}
    t_{w}=log(w/w_a)
    t_{h}=log(h/h_a)
    x,y,w,h是gt的中心坐标，宽，高。x_a,y_a,h_a,w_a为anchor boxes的中心坐标，宽，高
'''
# 对于每个anchor box找到具有max iou的gt目标
max_iou_bbox = bbox[argmax_ious]
print('max_iou_bbox：', max_iou_bbox)

'''为找到t_{x}，t_{y},t_{w},t_{h},需要祖安环有效的anchor boxes的y1,x1,y2,x2格式与gt box相关的ctr_x,ctr_y,w,h格式'''

# 先找到有效box的数据
height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
ctr_y = valid_anchor_boxes[:, 0] + 0.5 * height
ctr_x = valid_anchor_boxes[:, 1] + 0.5 * width

# 再找到gt的数据
base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

# 根据上述公式找到位置
'''下面前三行是因为后面的操作中出现了除法为了不出现负值和0值要找到一个最小值'''
eps = np.finfo(height.dtype).eps  # finfo函数是根据括号中的类型来获得信息，获得符合这个类型的数型，会返回很多值，可以根据实际情况选取
height = np.maximum(height, eps)
width = np.maximum(width, eps)

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dw = np.log(base_width / width)
dh = np.log(base_height / height)
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
print('anchor_locs：', anchor_locs)

'''
    现在得到了anchor_locs和每个有效的anchor boxes相关的标签
    用inside_index变量将他们映射到原始的anchors，无效的anchor box标签填充为-1，位置填充为0
'''
# 最终标签
anchor_label = np.empty(len(anchors), dtype=label.dtype)
anchor_label.fill(-1)
anchor_label[inside_index] = label

# 最终的坐标
anchor_locations = np.empty((len(anchors), 4), dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[inside_index, :] = anchor_locs
'''
    最终得到两个矩阵：
     *anchor_label{N,}----[22500]
     *anchor_locations{N,4}--[22500,4]
     这两个矩阵将用于RPN网络的输入，下面来看RPN网络的设计
'''

'''
    Region Proposal Network(RPN网络)设计：
        网络包括一个卷积模块，在卷积模块下一层包括一个回归层，预测anchor的box位置
        为生成region proposal，在特征提取模块得到的卷积层输出上滑动一个小的网络，这个小网络将输入卷积层特征的n*n空间窗口作为输入，每个滑动窗口映射到更低维的特征【512维度】
        这个特征将输入到两个并列的全连接层：
        *框回归层
        *框分类层
        自文章中提到，采用n=3采用n*n的卷积层和两个并列的1*1卷积层实现这一框架
'''
mid_channels = 512
in_channels = 512  # 这取决于feature map的输出维度，在vgg16中，我选取的特征提取层的输出维度为512
n_anchor = 9  # 每个锚点生成多少个框
conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
reg_layer = nn.Conv2d(in_channels=mid_channels, out_channels=n_anchor * 4, kernel_size=1, stride=1, padding=0)
cls_layer = nn.Conv2d(in_channels=mid_channels, out_channels=n_anchor * 2, kernel_size=1, stride=1,
                      padding=0)  # 这里可以加上激活函数做分类

# 下面来初始化权重，文章中使用了均值为0，标准差为0.1的初始权重，偏差初始化为0
conv1.weight.data.normal_(0, 0.1)
conv1.bias.data.zero_()

reg_layer.weight.data.normal_(0, 0.1)
reg_layer.bias.data.zero_()

cls_layer.weight.data.normal_(0, 0.1)
cls_layer.bias.data.zero_()

'''特征提取过程的输出可以输入到网络中用于预测目标相对于anchor的位置与值相关的目标分值'''
x = conv1(out_map)
pred_anchor_locs = reg_layer(x)
pred_cls_scoress = cls_layer(x)

print('pred_anchor_locs shape:', pred_anchor_locs.shape)
print('pred_cls_scoress shape', pred_cls_scoress.shape)

'''
    将他们重新格式化，让它与之前设计的锚点目标对齐，我们还将找到每个anchor box的目标得分用于proposal层
'''
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(
    (1, -1, 4))  # view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个
print('pred_anchor_locs shape:', pred_anchor_locs.shape)
pred_cls_scoress = pred_cls_scoress.permute(0, 2, 3, 1)
print('pred_cls_scoress shape', pred_cls_scoress.shape)

objectness_scores = pred_cls_scoress.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
print('objectness_scores shape:', objectness_scores.shape)
pred_cls_scoress = pred_cls_scoress.contiguous().view(1, -1, 2)
print('pred_cls_scoress shape', pred_cls_scoress.shape)

'''
    进入下一节：
        *pred_cls_scoress和pred_anchor_locs是RPN网络的输出，损失更新权重
        *pred_cls_scoress和objectness_scores作为proposal层的输入，proposal层用于后续roi网络的一系列proposal
    Generating proposal to feed faster　RCNN network
        proposal层需要用到如下参数：
            *模式：training_mode　or testing_mode
            *nms_thresh
            *n_train_pre_nms ------------训练时ｎｍｓ之前的ｂｂｏｘ数目
            *n_train_post_nms------------训练时ｎｍｓ之后的ｂｂｏｘ数目
            *n_test_pre_nms--------------测试时ｎｍｓ之前的ｂｂｏｘ数目
            *n_test_post_nms-------------测试时ｎｍｓ之后的ｂｂｏｘ数目
            *min_size--------------------生成一个ｐｒｏｐｏｓａｌ的所需要目标的最小高度
    Faster RCNN算法中RPN中proposal彼此之间重叠度过高，为了减少冗余，根据proposal区域的cls分数进行ｎｍｓ，将ｎｍｓ的阈值设置为０．７，
        这样每个图像大约会有２０００个建议区，作者表明。ｎｍｓ不会损伤算法的准确性且大大减少了建议的数量，在ｎｍｓ之后使用top-N的建议区进行检测。后续使用２０００个RPN proposal训练。
        在测试期间则只保留３００个proposal。
'''

nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

'''
    下面来长生网络感兴趣的proposal  region
    １.转换RPN网络的loc预测为bbox[y1,x1,y2,x2]格式
    ２．将预测框剪辑到图像上
    ３．去除高度或宽度
    ４．通过分数高低排序所有的(proposal scores)对
    ５．起初前几个预测框pre_nms_topN(训练时１２０００，测试时３００)
    ６．采用nms_thresh>0.7
    7.去除前几个预测框pos_nms_topN(训练师２０００，测试时３００)
'''

# １.转换RPN网络的loc预测为bbox[y1,x1,y2,x2]格式
'''这是为anchor boxe设置gt时的逆操作，该操作通过无参数化以及相对图像的偏差来解码预测，公式如下：
    x = (w_{a}*ctr_x_{p})+ctr_x_{a}
    y=(h_{a}*ctr_y_{p})+ctr_y_{a}
    h = np.exp(h_{p}) *h_{a}
    w = np.exp(w_{p})*w_{a}   
'''
# 转化anchor 格式从y1,x1,y2,x2到ctr_x,ctr_y,h,w
anc_height = anchors[:, 2] - anchors[:, 0]
anc_width = anchors[:, 3] - anchors[:, 1]
anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

# 通过上述公式转换预测locs，再抓换之前先将pred_anchor_loc和objectness_scores为numpy数组
pred_anchor_locs_numpuy = pred_anchor_locs[0].data.numpy()
objectness_scores_numpy = objectness_scores[0].data.numpy()
dy = pred_anchor_locs_numpuy[:, 0]
dx = pred_anchor_locs_numpuy[:, 1]
dw = pred_anchor_locs_numpuy[:, 2]
dh = pred_anchor_locs_numpuy[:, 3]

ctr_y = dy * anc_height + anc_ctr_y
ctr_x = dx * anc_width + anc_ctr_x
h = np.exp(dh) * anc_height
w = np.exp(dw) * anc_width

# 转换ctr_y,ctr_x,h,w到ｙ1,x1,y2,x2格式
roi = np.zeros(pred_anchor_locs_numpuy.shape, dtype=np.float32)
roi[:, 0] = ctr_y - 0.5 * h
roi[:, 1] = ctr_x - 0.5 * w
roi[:, 2] = ctr_y + 0.5 * h
roi[:, 3] = ctr_x + 0.5 * w
print('roi:', roi)  # 这里的值偏差有点大，有待商榷

# 剪辑预测框到图像上
image_size = (800, 800)
roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, image_size[0])

roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, image_size[0])
print('roi:', roi)

# 除去高度或者宽度<threshold的预测框
hs = roi[:, 2] - roi[:, 0]
ws = roi[:, 3] - roi[:, 1]
keep = np.where((hs > min_size) & (ws > min_size))[0]
roi = roi[keep, :]
scores = objectness_scores_numpy[keep]
print('scores shape:', scores.shape)  # 少了这么多，怀疑前面有地方写错了

# 按照分数从高到低排列所有的(proposal scores)对
order = scores.argsort()[::-1]
print('order:', order)

# 去除前几个预测框pre nms topN

order = order[:n_train_pre_nms]
roi = roi[order, :]
scores = scores[order]
print('roi shape:', roi.shape)

'''
    NMS
        - Take all the roi boxes [roi_array]
        - Find the areass of all the boxes [roi_areas]
        - Take the indexes of order the probability scores in descending order [order_array]
        keep = []
        while order_array.size > 0:
         - take the first element in order_array and append that to keep  
         - Find the areas with all other boxes
         - Find the index of all the boxes which have high overlap with this box
         - Remove them from order array
         - Iterate this till we get the order_size to zero (while loop)
        - Ouput the keep variable which tells what indexes to consider.
'''
y1 = roi[:, 0]
x1 = roi[:, 1]
y2 = roi[:, 2]
x2 = roi[:, 3]

areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 这个误差还是不能小视的
order = scores.argsort()[::-1]

keep = []

while order.size > 0:
    i = order[0]
    keep.append(i)  # 保留改图片的值
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, yy2 - yy1 + 1)
    h = np.maximum(0.0, xx2 - xx1 + 1)
    inter = w * h
    overleap = inter / (areas[i] + areas[order[1:]] - inter)

    inds = np.where(overleap <= nms_thresh)[0]
    order = order[inds + 1]

keep = keep[:n_train_post_nms]
roi = roi[keep]
print('nms后roi shape:', roi.shape)

'''
    现在最后得到了region proposal 这将被用作faster RCNN的输入，最终用于预测目标的位置（相对于建议的框）和目标的类别（每个proposal的分类）
    首先研究为何训练网络的proposal制定目标，之后将研究faster rcnn网络是如何实现的，并将这些proposal传给网络一伙的预测的输出，然后确定损失，
    算法将计算rpn损失和faster rcnn损失
'''

'''
    Peroposal  Target
        faster RCNN网络将region proposal（通过上一部分获取），gt边界框，以及对应的标签作为输入：
            *n_sample: roi重采样的样本数目，默认128
            *pos_ratio: n_sample中的正样本的比例，默认设置0.25
            *pos_iou_thresh: 设置为正样本regionproposal与gt目标之间最小重叠阈值
            *[neg_iou_threshold_lo,neg_iou_threshold_hi]=[0.0,0.5]设置为负样本的重叠值阈值
'''
n_sample = 128
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_threshold_lo = 0.0
neg_iou_threshold_hi = 0.5
'''
    写一下伪代码：
    *for 每个roi寻找他们与每个gt之间的iou[N,n]
        N为proposal box的数量
        n为gt box的数量
    *找出iou最大的值在N维度上对应哪一个gt值，这就是proposal的标签
    *如果这个iou值大于0.5就设置为正样本
    *正样本：
        我们随机抽取n_sample*pos_ratio个样本作为正标签
    *负样本：
        如果iou值在0.1-0.5之间，就设置为负标签，
    *随机抽取128个负样本proposal 并给标签0
    *收集了政府样本之后就删除其他proposal，
    *将每个gt区域的位置信息转换成所西药的格式，借助上面提到的转换公式，转到0-1之间
    *输出选中roi的label和locations
'''
# 找到每个gt目标与region proposal的iou，采用anchor box中相同代码
ious = np.empty((len(roi), 2), dtype=np.float32)  # 注意这里的2是因为bbox只有两个obj，封装活用
ious.fill(0)
for num1, i in enumerate(roi):
    ya1, xa1, ya2, xa2 = i
    roi_area = (ya2 + ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_areas = (yb2 - yb1) * (xb2 - xb1)

        inter_y1 = max([ya1, yb1])
        inter_x1 = max([xa1, xb1])
        inter_y2 = min([ya2, yb2])
        inter_x2 = min([xa2, xb2])

        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            iou = inter_area / (roi_area + box_areas - inter_area)
        else:
            iou = 0
        ious[num1, num2] = iou
print('ious shape:', ious.shape)

# 找到与每个region proposal具有较高iou的gt，并找到最大iou
gt_assignment = ious.argmax(axis=1)
max_iou = ious.max(axis=1)
print('gt_assignment:', gt_assignment)
print('max_iou:', max_iou)

# 为每个proposal分配标签
gt_roi_label = labels[gt_assignment].cpu().numpy()
'''
    根据每个pos_iou_thresh选择前景rois，希望只保留n-sample * pos_raio 128*0.25=32个前景样本，
    因此如果只得到少于32个正样本，保持原状，如果得到多余32个前景目标，从中采样32个样本
'''
pos_index = np.where(max_iou >= pos_iou_thresh)[0]
pos_roi_per_this_image = int(min((n_sample * pos_ratio, pos_index.size)))
if pos_index.size > pos_roi_per_this_image:
    pos_index = np.random.choice(pos_index, pos_roi_per_this_image, replace=False)

print('pos_roi_per_this_image:', pos_roi_per_this_image)
print('pos_index:', pos_index)

'''
    针对负背景样本proposal进行相似处理，如果对于之前分配的gt目标，proposal的iou在neg_iou_threshold_lo和neg_iou_threshold_hi之间，对该proposal分配0标签，
    从这些负样本中采取n_sample-pos_sample个proposal  
'''
neg_index = np.where((max_iou < neg_iou_threshold_hi) & (max_iou > neg_iou_threshold_lo))[0]
neg_roi_per_image = int(n_sample - pos_roi_per_this_image)
if neg_index.size > neg_roi_per_image:
    neg_index = np.random.choice(neg_index, neg_roi_per_image, replace=False)

print('neg_roi_per_image:', neg_roi_per_image)
print('neg_index:', neg_index)

# 下面整合正样本索引和负样本索引以及他们格子标签和proposal
keep_index = np.append(pos_index, neg_index)
gt_roi_label = gt_roi_label[keep_index]
gt_roi_label[pos_roi_per_this_image:] = 0  # 负样本label置0
sample_roi = roi[keep_index]
print('sample_roi shape:', sample_roi.shape)

# 对这些选出来的gt按照anchor box的分配位置的方式进行参数化，同样用到上面的公式,注意这种取值方式，很有意思
bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]
print('bbox_for_sampled_roi shape:', bbox_for_sampled_roi.shape)

height = sample_roi[:, 2] - sample_roi[:, 0]
width = sample_roi[:, 3] - sample_roi[:, 1]
ctr_y = sample_roi[:, 0] + 0.5 * height
ctr_x = sample_roi[:, 1] + 0.5 * width

base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
base_ctr_y = bbox_for_sampled_roi[:, 0] + 0.5 * base_height
base_ctr_x = bbox_for_sampled_roi[:, 1] + 0.5 * base_width

# 套公式
eps = np.finfo(height.dtype).eps
height = np.maximum(eps, height)
width = np.maximum(eps, width)

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()

# print('gt_roi_locs:', gt_roi_locs)

'''
    通过上一节得到了roi的gt_roi_locs和gt_roi_labels,下面设计Faster rcnn网络来预测locs和标签
    Faster RCNN
        使用ROI Polling来提取特征，每个proposal有选择搜索来建议得出
        roi polling目的是执行从不均匀大小到固定大小的feature map，这一层有两个输入：
        *一个从有几个卷积和最大池化层的深度卷积网络获得的固定大小的feature map
        *一个N*5矩阵，代表一列roi polling N表示roi的个数，第一列表示影像索引，剩下的为左上右下点坐标
    Roi Polling：
        *划分proposal到等大小的部分
        *找到每个部分的最大值
        *复制这些最大值到输出缓冲区
    从前面的部分可以得到gt_roi_locs,gt_roi_labels和sample_rois我们将会使用sample_rois作为roi polling的输入，
    注意：sample_rois的规格是[N,4]每行的格式是yxhw，需要做两件事：
    1.添加图像索引：【目前这里只有一个图像】
    2.格式改为xywh
'''
# 因为sample_rois是一个numpy数组，我们将会转换为pytorch张量，创建roi_idices张量
rois = torch.from_numpy(sample_roi).float()
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()
print('rois shape:', rois.shape)
print('roi_indices shape:', roi_indices.shape)

# 合并roi和roi_indices_这样就能得到维度为【N,5]的(x,y,h,w)张量
indice_and_rois = torch.cat([torch.reshape(roi_indices, [len(rois), 1]), rois],
                            dim=1)  # 这里两个向量都是一维的，然而了None之后相当于变成了一个列向量
xy_indices_and_rois = indice_and_rois[:, [0, 2, 1, 4, 3]]
indices_and_rois = xy_indices_and_rois.contiguous()
print('xy_indices_and_rois shape:', xy_indices_and_rois.shape)

'''
    下面将数据传输到rooi polling层，伪代码如下：
        *将tensor等分成4*4的16个格子
        *定义输出tensor
        *对每个roi：
            找到同roi同维度的下采样映射子集adaptIveMaxPool2d来降维
            输出结果
'''
size = (7, 7)
adaptive_max_pool = nn.AdaptiveMaxPool2d(size)
output = []
rois = indice_and_rois.data.float()
rois[:, 1:].mul_(1 / 16.0)  # 这里为什么要将数值做下采样弄不明白
rois = rois.long()
num_rois = rois.size(0)
''''这种单图的操作在写算法的时候可以写一个batch的函数扩展'''
for i in range(num_rois):
    roi = rois[i]
    im_idx = roi[0]
    # narrow为out_map第0个维度从im_idx到1的值，这是拿了一张图的out map出来啊
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1),
         roi[1]:(roi[3] + 1)]  # 这里roi是根据一张图得到的，out map的batch为1也需要抽取出来一张图，然后使用roi坐标剪辑得到，感兴趣区域，进行Polling
    output.append(adaptive_max_pool(im))
output = torch.cat(output, 0)
print('output shape:', output.size())
k = output.view(output.size(0), -1)
print('k shape:', k.size())

'''
    这将会是一个classifier层的输入明进一步将会分出classification 和regression
'''
# 定义网络
roi_head_classifier = nn.Sequential(nn.Linear(25088, 4096),
                                    nn.Linear(4096, 4096))
cls_loc = nn.Linear(4096, 21 * 4)  # 假设使用voc数据集的话
cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()
score = nn.Linear(4096, 21)

# 将roi pooling的输出传到上面定义的网络
# 其中，roi_cls_loc和roi_cls_score是从事极边界区域的道德两个输出张量
k = roi_head_classifier(k)  # 这些定义的都是模型，将tensor放进去
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)
print('roi_cls_loc shape:', roi_cls_loc.shape)
print('roi_cls_score shape:', roi_cls_score.shape)

'''下面来说计算损失函数，RPN损失函数，在之前，计算了anchor box的目标值，和RPN网络的输出值，两者的差值就是RPN的损失'''
print('pred_anchor_locs.shape:', pred_anchor_locs.shape)
print('pred_cls_scoress.shape:', pred_cls_scoress.shape)
print('anchor_locations.shape:', anchor_locations.shape)
print('anchor_label.shape:', anchor_label.shape)

# 重新排列，将输入和输出排成一行
rpn_loc = pred_anchor_locs[0]  # 因为它的bath为1，多batch操作同样可以用自定义的batch扩展函数
rpn_score = pred_cls_scoress[0]
gt_rpn_loc = torch.from_numpy(anchor_locations)
gt_rpn_score = torch.from_numpy(anchor_label)
print('rpn_loc shape:', rpn_loc.shape)
print('rpn_score shape:', rpn_score.shape)
print('gt_rpn_loc shape:', gt_rpn_loc.shape)
print('gt_rpn_score shape:', gt_rpn_score.shape)

# pred_cls_scoress和anchor_label是rpn玩过的预测对象之和实际对象值，我们将会用如下的额分别对regression and classification损失函数
# 对classification使用交叉上损失
rpn_cls_loss = nn.functional.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)
print('rpn_cls_loss:', rpn_cls_loss)

# 对regression使用smooth L1损失，不用L2是因为RPN的预测回归头的值不是有线的，regression损失也被应用在有正标签的边界区域中
pos = gt_rpn_score > 0
mask = pos.unsqueeze(dim=1).expand_as(rpn_loc)  # 从向量扩充到（？，1）再扩充到（？，4）
print('mask shape:', mask.shape)

# 取得有正数标签的便捷区域,pytorch中获取指定tensor的方法？？？？转成相同维度的mask？？
# 不能用索引表前找到对应的loc吗？
mask_loc_preds = rpn_loc[mask].view(-1, 4)
mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)
print('mask_loc_preds shape:', mask_loc_preds.shape)
print('mask_loc_targets shape:', mask_loc_targets.shape)

# regression对应的损失应为
x = torch.abs(mask_loc_preds - mask_loc_targets)
rpn_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * 0.5 * (x - 0.5) ** 2)
print('rpn_loc_loss:', rpn_loc_loss.sum())

'''合并rpn_cls_loss and rpn_loc_loss 因为rpn_cls_loss应用在全部的边界区域，regression loss应用在正数标签区域'''
rpn_lambda = 10
N_reg = (gt_rpn_score > 0).float().sum()  # 求正样本个数
rpn_loc_loss = rpn_loc_loss.sum() / N_reg  # 累加求均值
rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
print('rpn loss', rpn_loss)

'''Faster RCNN Loss'''
# 预测
print('roi_cls_loc shape:', roi_cls_loc.shape)
print('roi_cls_score shape', roi_cls_score.shape)
# 真实
print('gt_roi_locs shape:', gt_roi_locs.shape)
print('gt_roi_label shape:', gt_roi_label.shape)

# 转化为torch变量
gt_roi_locs = torch.from_numpy(gt_roi_locs)
gt_roi_label = torch.from_numpy(gt_roi_label).long()
print('gt_roi_locs shape:', gt_roi_locs.shape)
print('gt_roi_label shape:', gt_roi_label.shape)

# 分类损失
roi_cls_loss = nn.functional.cross_entropy(roi_cls_score, gt_roi_label, ignore_index=-1)
print('roi_cls_loss:', roi_cls_loss)

'''回归损失，每个roi有21个预测框便捷，为了计算损失，也只采用带有正标签的预测框'''
n_sample = roi_cls_loc.shape[0]
roi_loc = roi_cls_loc.view(n_sample, -1, 4)
print('roi_loc shape:', roi_loc.shape)

roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]  # 这种通过传入向量的方式来指定第二个维度（21）上用哪个通道的值，可以做到通道之间相互不会干扰
print('roi_loc shape:', roi_loc.shape)

# 用计算rpn的方式计算回归损失
# 对regression使用smooth L1损失，不用L2是因为RPN的预测回归头的值不是有线的，regression损失也被应用在有正标签的边界区域中
pos = gt_roi_label > 0
mask = pos.unsqueeze(dim=1).expand_as(roi_loc)  # 从向量扩充到（？，1）再扩充到（？，4）
print('mask shape:', mask.shape)

# 取得有正数标签的便捷区域,pytorch中获取指定tensor的方法？？？？转成相同维度的mask？？
# 不能用索引表前找到对应的loc吗？
mask_loc_preds = roi_loc[mask].view(-1, 4)
mask_loc_targets = gt_roi_locs[mask].view(-1, 4)
print('mask_loc_preds shape:', mask_loc_preds.shape)
print('mask_loc_targets shape:', mask_loc_targets.shape)

# regression对应的损失应为
x = torch.abs(mask_loc_preds - mask_loc_targets)
roi_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * 0.5 * (x - 0.5) ** 2)
print('roi_loc_loss:', roi_loc_loss.sum())

# roi损失总和
roi_lambda = 10
roi_loss = roi_cls_loss + roi_lambda * roi_loc_loss
print('roi_loss :', roi_loss)

# final loss
total_loss = rpn_loss + roi_loss
d = 0
