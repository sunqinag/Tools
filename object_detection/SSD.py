import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from itertools import product
import math
import torch.nn.init as init

'''
    vgg使用了vgg16版本作为基础网络,做出了更改，丢弃了全连接层改为1024*3*3和1024*1*1的卷基层，
    其中conv4-1卷基层前的maxpooling层的ceil_model=True，这使得特征图长宽为38*38，
    还有conv5-3后面的maxpooling参数为（kernel_size=3,stride=1,padding=1）,不进行下采样，
    然后在fc7后面街上多尺度提取的另外四个卷积层就构成了完整的ssd网络
    
    参考博客：https://zhuanlan.zhihu.com/p/95032060
    参照github：ssd.pytorch
'''


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


'''
    在代码中最后获得的conv7就是上面说的fc7，特征维度为[None,1024,19,19]在它后面构建多尺度提取网络，
    也就是网络结构图中的Extra Feaure Layers
'''


def add_extras(cfg, i, bath_norm=False):
    layers = []
    in_channels = i
    flag = False  # 用来控制kernel_size=1 or 3
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


'''
    看网络结构图可以看到，改造后的vgg16和Extra Layers还有6个横着的线，
    这代表的对6个尺度的特征图进行卷积获得预测框的回归(loc)和分类(cls)信息， 
'''


# vgg = vgg(base['300'], 3)
# add_extras = add_extras(extras['300'], 1024)
# for k, v in enumerate(list(vgg)):
#     print('第：', k, '层：：：', v)
# print('===============================================')
# for k, v in enumerate(list(add_extras)):
#     print('第：', k, '层：：：', v)
# print('===============================================')
# for k, v in enumerate(list(add_extras[1::2])):
#     print('第：', k, '层：：：', v)
#

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []  # 多尺度分支的回归网络
    conf_layers = []  # 多尺度分支的分类网络
    # 第一部分：vgg网络的conv2d-4_3(21层)，conv2d_7_1（-2层）
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        # 回归box*4坐标
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # 置信度box*(num_classes)
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    # 第二部分，cfg从第三个开始作为box的个数，而且用于多尺度提取的网络分别为1,3,5,7
    for k, v in enumerate(extra_layers[1::2], 2):  # 从第二个开始，
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


'''
    Anchor生成（prior box层）
    在之前原理篇https://mp.weixin.qq.com/s/lXqobT45S1wz-evc7KO5DA中有所介绍，
    SSD从魔改后的VGG16的conv4-3开始一共使用6个不同大小的特征图，尺寸分别为（38,38），（19,19），（10，10），（5,5），（3,3），（1,1）
    但对每个特征图设置的anchor的数量不同，anchor的设置包括尺寸和长宽比两个方面。
    对于anchor的公式有Sk= Smin+[(Smax-Smin)/(m-1)]*(k-1)其中k属于【1，M】,
    其中M指的是特征图个数，这里为5因为第一层con4-3的anchor是单独设置的。Sk代表anchor大小相对于特征图的比例
    这最后Smin和Smax表示比例的最小值和最大值，论文中给出的分别为0.2和0.9.对于第一个特征图，anchor尺度比例设置为Smin、2=0.1
    则它的尺度为300*0.1=30，后面的特征图代入公式计算，并将其映射回原图300的大小可以得到。剩下的5个特征图的尺度为：
    60,111,162,213,264，综合起来，6个特征图的尺度Sk为30,60,111,162,213,264.有了Anchor的尺度，接下来设置anchor的长宽。
    论文中长宽一般设置为ar=1,2,3,1/2,1/3,根据面积和长宽可以得到anchor的宽度和高度：
    Wk=Sk*sqr（ar）,Hk = Sk/sqr(ar),注意点如下：
    *上面的Sk是相对于原图大小
    *，默认情况下，每个特征图除了上面5个比例的anchor还会设置一个尺度为sqr（SK*S(k+1)）且ar=1的anchor，
        这样每个特征图都设置了两个长宽比为1但大小不同的正方形anchor，最后一个特征图需要参考一下Sm+1=315来计算Sm
    *在实现conv4-3，conv10-2，conv11-2层时仅适用4个anchor，不适用长宽比为3,1/3的anchor
    *每个单元的anchor的中心点分布在每个单元的中心，即【i+0.5/fk，j+0.5/fk】其中fk为特征图的大小
    *从anchor的值来看，越前面的特征图的anchor尺寸越小，也就是说对小目标的效果越好，anchor的总数为38*38*4+19*19*6+10*10*6+5*5*6+3*3*45+1*1*4=8732
    
'''


class PriorBox:
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_size = cfg['min_sizes']
        self.max_size = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('variance的值必须大于0')

    def forward(self):
        mean = []
        # 遍历多尺度的 特征图: [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            print('=============')
            # 遍历每个像素
            for i, j in product(range(f), repeat=2):  # 活用featuer map长宽相同
                print(i, j)
                f_k = self.image_size / self.steps[k]  # 第k层feature map的大小
                # 每个框中心坐标,没明白为什么还向下做映射
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio:1当ratio==1的是偶，会产生两个box
                # r==1, size = s_k， 正方形
                s_k = self.min_size[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # r==1, size = sqrt(s_k * s_(k+1)), 正方形
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = math.sqrt(s_k * (self.max_size[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 当 ratio != 1 的时候，产生的box为矩形
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                    mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
        # 转化为torch的tensor
        output = torch.Tensor(mean).view(-1, 4)
        # 归一化，把输出设置在【0,1】
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


'''
    网络结构，前面有魔改后的VGG16和Extra Layers还有生成anchor的Priobox策略，可以总结出SSD的整体结构如下
'''


class SSD(nn.Module):
    """Single Shot Multibox Architecture
        The network is composed of a base VGG network followed by the
        added multibox conv layers.  Each multibox layer branches into
            1) conv2d for class conf scores
            2) conv2d for localization predictions
            3) associated priorbox layer to produce default bounding
               boxes specific to the layer's feature map size.
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
        Args:
            phase: (string) Can be "test" or "train"
            size: input image size
            base: VGG16 layers for input, size of either 300 or 500
            extras: extra layers that feed to multibox loc and conf layers
            head: "multibox head" consists of loc and conf conv layers
        """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # 配置config
        self.cfg = voc_cfg
        # 初始化anchor
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(
                self.priorbox.forward())  # variable的volatile属性默认为False，如果某一个variable的volatile属性被设为True，那么所有依赖它的节点volatile属性都为True。volatile属性为True的节点不会求导，volatile的优先级比requires_grad高。
        self.size = size

        # SSD Network
        # backbone网络
        self.vgg = nn.ModuleList(base)
        # conv4-3后面的网络，L2正则化
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        # 回归和分类网络
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()  # 定位类

    def forward(self, x):
        source = list()
        loc = list()
        conf = list()

        # vgg网络到conv4-3,此时尺寸应该为19
        for k in range(23):  # 我勒个擦，还能这样一层层下来
            x = self.vgg[k](x)
        # L2正则化
        s = self.L2Norm(x)
        source.append(s)
        with open('extra.log', 'a') as f:
            info = '抽取vgg的第：' + str(k) + '个卷基层，tensor尺寸为:' + str(s.shape[-1]) + '\n'
            f.write(info)
        # conv4-3到fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        source.append(x)
        with open('extra.log', 'a') as f:
            info = '抽取vgg的第：' + str(k) + '个卷基层，tensor尺寸为:' + str(x.shape[-1]) + '\n'
            f.write(info)

        # extras网络
        for k, v in enumerate(self.extras):
            x = nn.functional.relu(v(x), inplace=True)
            if k % 2 == 1:
                source.append(x)  # 把需要多尺度的网络输出存入source
                with open('extra.log', 'a') as f:
                    info = '抽取extras的第：' + str(k) + '个卷基层，tensor尺寸为:' + str(x.shape[-1]) + '\n'
                    f.write(info)
        with open('extra.log', 'a') as f:
            for sour in source:
                f.write('收集一下source采集的多尺度样本=============' + '\n')
                f.write('source的尺寸大小为:' + str(sour.shape[-1]) + '\n')
        # 多尺度回归和分类网络
        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            output = self.detect()
            pass
        else:
            output = (
                # loc的输出size：batch，8732,4
                loc.view(loc.size(0), -1, 4),
                # conf的输出size：batch，732,21
                conf.view(conf.size(0), -1, self.num_classes),
                # 生成所有的后选框size：【8732,4】
                self.priors
            )
        return output  # 这种维度的输出是不是太大了


# 为了增加可读性，对上面SSD增加一次封装

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != 'test' and phase != 'train':
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # 调用multibox生成cgg，extras，head
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)


'''
    loss解析：
        SSD的损失包含定位损失Lloc和分类损失Lconf，真个损失函数表达式为：
        L(x,c,l,g) = (1./N))Lconf(x,c)+Alpha*Lloc(x,l,g)
        其中N是先验框的正样本数量，c是类别置信度预测值，l是先验框对应的边界框预测值，
        g为gt的未知参数，x为网络预测值。对于位置损失，采用了smooth L1 loss，位置信息都是encode之后的数值，
        对于分类损失，首先需要使用 hard negtive mining价格正负样本按照1:3的比例吧负样本抽取下来：
        抽取方法为：
            针对所有batch的confidence按照置信度误差进行降序排列，取出前top_k个负样本。
    实验步骤：
        *reshapebatch中的conf，方便后续排序
        *置信度无差越大实际上就是预测背景的置信度越小，则logsoftmax越小，取绝对值，则|logsoftmax|越大，
        降序排列-logsoftmax，取前top_k的负样本     
'''

'''损失函数完整代码,将上面得到的output接入'''


class MutiBoxLoss(nn.Module):
    '''
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    '''

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        # super(MutiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_trget = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = voc_cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
            Args:
                predictions (tuple): A tuple containing loc preds, conf preds,
                and prior boxes from SSD net.
                    conf shape: torch.size(batch_size,num_priors,num_classes)
                    loc shape: torch.size(batch_size,num_priors,4)
                    priors shape: torch.size(num_priors,4)
                targets (tensor): Ground truth boxes and labels for a batch,
                    shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)  # batch_size
        priors = priors[:loc_data.size(1), :]  # 这里写不写都行，没有区别
        num_priors = priors.size(0)  # 先验框个数
        num_classes = self.num_classes  # 类别数

        # 匹配每个prior box的gt box
        # 创建loc-t和conf-t来保存真实box的位置和类别
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data  # gt box信息
            labels = targets[idx][:, -1].data  # gt conf信息
            defaults = priors.data  # priors的box信息
            print('11111111111')
            # 匹配gt
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t =conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t,requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)
        #匹配中所有的正样本
        pos = conf_t>0
        num_pos = pos.sum(dim=1,keepdim=True)#keepdim:求和之后这个dim的元素个数为１，所以要被去掉，如果要保留这个维度，则应当keepdim=True
        #定位loss,使用smooth L1
        pos_idx =pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)  # 预测的正样本box信息
        loc_t = loc_t[pos_idx].view(-1, 4)  # 真实的正样本box信息
        loss_l = nn.functional.smooth_l1_loss(loc_p, loc_t, size_average=False)  # Smooth L1 损失

        '''
        Target；
            下面进行hard negative mining
        过程:
            1、 针对所有batch的conf，按照置信度误差(预测背景的置信度越小，误差越大)进行降序排列;
            2、 负样本的label全是背景，那么利用log softmax 计算出logP,
               logP越大，则背景概率越低,误差越大;
            3、 选取误差交大的top_k作为负样本，保证正负样本比例接近1:3;
        '''
        # Compute max conf across batch for hard negative mining
        # shape[b*M,num_classes]
        batch_conf = conf_data.view(-1, self.num_classes)
        # 使用logsoftmax，计算置信度,shape[b*M, 1]
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # 把正样本排除，剩下的就全是负样本，可以进行抽样
        loss_c = loss_c.view(num, -1)  # shape[b, M]
        # 两次sort排序，能够得到每个元素在降序排列中的位置idx_rank
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # 抽取负样本
        # 每个batch中正样本的数目，shape[b,1]
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        # 抽取前top_k个负样本，shape[b, M]
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        # shape[b,M] --> shape[b,M,num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 提取出所有筛选好的正负样本(预测的和真实的)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # 计算conf交叉熵
        loss_c = nn.functional.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # 正样本个数
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
''' 工具函数和类'''


class Detect:
    pass


# 匹配gt函数
'''
    先验框的匹配函数，在训练时首先要确定训练图片中gt是由哪一个先验框来匹配，与之匹配的先验框所对应的边界框将负责预测它
    SSD的先验框和gt匹配原则主要有两点：
        1.对于图片中每个gt找到和他iou最大的先验框，该先验框预期匹配，这样可以拨癌症每个gt一定能与某个prior匹配
        2.对于剩余的未匹配的先验框，若某个gt和它的iou大于某个阈值（一般为0.5）那么该prior和这个gt，
            剩下的没有配配上的先验框都是负样本（如果过个gt与某一个先验框的iou均大于阈值，那么prior只与iou最大的那个进行匹配）
'''


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    '''
        把和每个prior box 有最大的IOU的ground truth box进行匹配，
    同时，编码包围框，返回匹配的索引，对应的置信度和位置
     Args:
        threshold: IOU阈值，小于阈值设为背景
        truths: ground truth boxes, shape[N,4]
        priors: 先验框， shape[M,4]
        variances: prior的方差, list(float)
        labels: 图片的所有类别，shape[num_obj]
        loc_t: 用于填充encoded loc 目标张量
        conf_t: 用于填充encoded conf 目标张量
        idx: 现在的batch index
        The matched indices corresponding to 1)location and 2)confidence preds.
    '''
    # 计算iou
    overlaps = overlap(truths.long(), priors.long())
    # [1,num_objects] 和每个ground truth box 交集最大的 prior box
    best_priot_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # 这个keepdim控制着是否返回索引
    # [1,num_priors] 和每个prior box 交集最大的 ground truth box
    best_true_overlap, best_true_idx = overlaps.max(0, keepdim=True)
    # 扩展一下维度
    best_true_idx.squeeze_(0)
    best_true_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_priot_overlap.squeeze_(1)

    # 为保证每个gt box与某个prior box匹配，固定值为2>threshold,来确定best prior
    best_true_overlap.index_fill_(0, best_prior_idx,
                                  2)  # index_fill(dim,index,val)按照指定的维度轴dim 根据index去对应位置，将原tensor用参数val值填充，这里强调一下，index必须是1D tensor，index去指定轴上索引数据时候会广播，与上面gather的index去索引不同(一一对应查)
    # 确保每个gt匹配它的都是具有最大的iou prior，根据best_prior_idx锁定best_true_idx里面的最大iou prior
    for j in range(best_prior_idx.size(0)):
        best_true_idx[best_prior_idx[j]] = j
    mathes = truths[best_true_idx]  # 提取出所有匹配的ground truth box, Shape: [M,4]
    conf = labels[best_true_idx] + 1  # 提取出所有GT框的类别， Shape:[M],但这里为什么要加一呢？？？？
    # 把iou<threshold的框类别设置为0
    conf[best_true_overlap < threshold] = 0

    # 编码包围框：
    loc = encode(mathes, priors, variances)
    # 保存匹配好的loc和conf到loc_t和conf_t中
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    d = 0


'''
    位置信息编解码:
    之前在提到计算坐标损失的时候，坐标是encoding之后的，根据论文的描述，预测框和gt box之间存在一个转换关系：
        *先验框位置：d=(Dx,Dy,Dw,Dh)
        *gt框位置： g(Gx,Gy,Gw,Gh)
        *variance是先验框坐标方差，然后编码的过程可以表示为：
            剩余公式参照博客：https://zhuanlan.zhihu.com/p/95032060
'''


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
        we have matched (based on jaccard overlap) with the prior boxes.
        Args:
            matched: (tensor) Coords of ground truth for each prior in point-form
                Shape: [num_priors, 4].
            priors: (tensor) Prior boxes in center-offset form
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def overlap(box_a, box_b):
    '''
        Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes，

    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    '''
    inter = intersect(box_a, box_b)
    # box_a和box_b的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]#(N,)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]#(M,)  # 这里两个面积为什么要在不同维度上扩展呢？不明白～～～～
    union = area_a + area_b - inter
    return inter / union


def intersect(box_a, box_b):
    '''计算A ∩ B'''
    A = box_a.size(0)
    B = box_b.size(0)
    # 右下角，选出最小值
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    # 左上角，选出最大值
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    # 负数用0截断，为0代表交集为0
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


'''
    坐标转换：
        默认的，网络得出的先验框的坐标形式为（x,y,w,h）但我们比对的label为（Xmin,Ymin,Xmax,Ymax）
        这里要将prior的坐标进行转换
'''


def point_form(boxes):
    '''
    把 prior_box (cx, cy, w, h)转化为(xmin, ymin, xmax, ymax)
    '''
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


# 将网络层进行L2正则化
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(
            self.n_channels))  # rameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)  # 使用值val填充输入Tensor或Variable.

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x/norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def log_sum_exp(x):
    x_max = x.detach().max()  # 返回一个新的从当前图中分离的 Variable。
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


if __name__ == '__main__':
    import cv2

    voc_cfg = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
    }

    base = {
        '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '512': [],
    }
    extras = {
        '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        '512': [],
    }
    mbox = {
        '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
        '512': [],
    }

    image = 'demo.jpg'
    image = cv2.imread(image)
    image = cv2.resize(image, (300, 300))
    image = torch.from_numpy(image).transpose(0, 2).float()
    image = torch.unsqueeze(image, 0)
    label = np.array([[0, 0, 200, 200, 1], [50, 50, 250, 250, 2]])
    label = torch.from_numpy(label).long().unsqueeze(0)

    # # 查看下vgg网络，共35层
    # vgg_net = vgg(cfg=base['300'], i=3)
    # for k, v in enumerate(vgg_net):
    #     image = v(image)
    #     print('第' + str(k) + '层', 'vgg特征提取到尺寸为：', image.shape[-1])  # 300的图得到了300,150,75,38,19的尺寸
    #     cv2.imwrite('vgg_pic/' + str(k) + '.jpg', image[0, 0, :, :].detach().numpy())
    #
    # extra_net = add_extras(extras['300'], 1024)  # 接上方尺寸为19的图得到10,5,3,1的尺寸
    # for k, v in enumerate(extra_net):
    #     image = v(image)
    #     print('第' + str(len(vgg_net) + k + 1) + '层', 'extra网络提取到尺寸为：', image.shape[-1])
    #     cv2.imwrite('extra_pic/' + str(k) + '.jpg', image[0, 0, :, :].detach().numpy())
    # '''综上：经过魔改vgg和extra网络共有不同尺寸为300,150,75,38,19,10,5,3,1'''

    # 运行完整的SSD看看全局保存的尺寸
    ssd = build_ssd(phase='train', size=300, num_classes=21)

    output = ssd.forward(image)

    loss = MutiBoxLoss(num_classes=21,
                       overlap_thresh=0.5,
                       prior_for_matching=None,
                       bkg_label=None,
                       neg_mining=None,
                       neg_pos=None,
                       neg_overlap=None,
                       encode_target=None,
                       use_gpu=True)
    loss.forward(predictions=output, targets=label)

    # output = PriorBox(cfg=voc_cfg).forward()
    # print(output)
    # vgg, extra_layers, (l, c) = multibox(vgg(base['300'], 3),
    #                                      add_extras(extras['300'], 1024),
    #                                      [4, 6, 6, 6, 4, 4], 21)
    # print(nn.Sequential(*l))
    # print('---------------------------')
    # print(nn.Sequential(*c))
