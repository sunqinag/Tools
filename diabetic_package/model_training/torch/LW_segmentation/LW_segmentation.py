# ----------------------------
#!  Copyright(C) 2021
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：陈瑞侠,刘胜
#   完成日期：2021-3-4
# -----------------------------
import os,sys
import time
import torch
import math
import socket
import numpy as np
import matplotlib
from torch import optim
import torch.nn as nn
matplotlib.use('Agg')
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# from torch.utils.tensorboard import SummaryWriter

from diabetic_package.model_training.estimator.LW_segmentation.model.DABNet\
    import DABNet
from diabetic_package.model_training.estimator.LW_segmentation.builders.\
    dataset_builder import build_dataset_train
from diabetic_package.model_training.estimator.LW_segmentation.utils.utils \
    import setup_seed, init_weight, netParams
from diabetic_package.model_training.estimator.LW_segmentation.utils.metric.metric \
    import get_iou
from diabetic_package.model_training.estimator.LW_segmentation.utils.losses.loss \
    import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth
from diabetic_package.model_training.estimator.LW_segmentation.utils.optim \
    import RAdam, Ranger, AdamW
from diabetic_package.model_training.estimator.LW_segmentation.utils.scheduler.\
    lr_scheduler import WarmupPolyLR

from diabetic_package.model_training.estimator.LW_segmentation.transfer_pth2pt\
    import transfer_pth2pt
from diabetic_package.log.log import bz_log

sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'
GLOBAL_SEED = 1234

# writer = SummaryWriter()
def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default="DABNet",
                        help="model name: (default ENet)")
    parser.add_argument('--dataset', type=str, default=" ",
                        help="dataset: cityscapes or camvid")
    parser.add_argument('--input_size', type=str, default="512,512",
                        help="input size of model")
    parser.add_argument('--ignore_label', type=str, default="0",
                        help="background label")
    parser.add_argument('--num_workers', type=int, default=0,
                        help=" the number of parallel threads")
    parser.add_argument('--classes', type=int, default=2,
                        help="the number of classes in the dataset. "
                             "19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--train_type', type=str, default="trainval",
                        help="ontrain for training on train set,"
                             " ontrainval for training on train+val set")
    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=100,
                        help="the number of epochs: 300 for train set, "
                             "350 for train+val set")
    parser.add_argument('--random_mirror', type=bool, default=True,
                        help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=True,
                        help="input image resize 0.5 to 2")
    parser.add_argument('--lr', type=float, default=5e-3,
                        help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--optim',type=str.lower,default='adam',
                        choices=['sgd','adam','radam','ranger'],
                        help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='warmpoly',
                        help='name of lr schedule: poly')
    parser.add_argument('--num_cycles', type=int, default=1,
                        help='Cosine Annealing Cyclic LR')
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=500,
                        help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0 / 3,
                        help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--use_label_smoothing', action='store_true',
                        default=False,
                        help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--use_ohem', action='store_true', default=False,
                        help='OhemCrossEntropy2d Loss for cityscapes dataset')
    parser.add_argument('--use_lovaszsoftmax', action='store_true',
                        default=False,
                        help='LovaszSoftmax Loss for cityscapes dataset')
    parser.add_argument('--use_focal', action='store_true', default=False,
                        help=' FocalLoss2d for cityscapes dataset')
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True,
                        help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0",
                        help="default GPU devices (0,1)")
    # checkpoint and log
    parser.add_argument('--resume', type=str, default="",
                        help="use this file to load last checkpoint for "
                             "continuing training")
    parser.add_argument('--savedir', default="./checkpoint/",
                        help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt",
                        help="storing the training and validation logs")
    args = parser.parse_args()
    return args

def train_model(args):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> 输入尺寸:{}".format(input_size))

    print(args)

    if args.cuda:
        print("=====> 使用的GPU id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("没有找到GPU或者GPU id错误，请使用CPU")

    # socket
    if args.is_socket:
        socketed = build_socket_connect()

    # set the seed
    setup_seed(GLOBAL_SEED)

    cudnn.enabled = True
    print("=====> 构建网络")
    bz_log.info("构建网络")
    # build the model and initialization
    model =  DABNet(classes=args.classes)
    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')

    print("=====>计算网络参数量和每秒浮点运算次数")
    total_paramters = netParams(model)
    print("参数数目: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    bz_log.info("开始加载数据...")
    # load data and data augmentation
    datas, trainLoader, valLoader = build_dataset_train(args.dataset,args.classes,
                input_size, args.batch_size, args.ignore_label, args.train_type,
                args.random_scale, args.random_mirror, args.num_workers)
    bz_log.info("数据加载完成！")

    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter


    print('=====> 数据统计结果：')
    print("data['classWeights']: ", datas['classWeights'])
    print('mean and std: ', datas['mean'], datas['std'])

    # define loss function, respectively
    weight = torch.from_numpy(datas['classWeights'])

    if args.use_label_smoothing:
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight,
                                                 ignore_label=args.ignore_label)
    else:
        criteria = CrossEntropyLoss2d(weight=weight,
                                      ignore_label=args.ignore_label)
    if args.cuda:
        criteria = criteria.cuda()

        args.gpu_nums = 1
            # print("single GPU for training")
        model = model.cuda()  # 1-card data parallel

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0

    # continue training
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("=====> 加载checkpoint文件 '{}' (epoch {})".format(args.resume,
                                                                    checkpoint[
                                                                        'epoch']))
        else:
            print("=====> 没有找到checkpoint文件 '{}'".format(args.resume))

    model.train()
    cudnn.benchmark = True
    # cudnn.deterministic = True ## my add

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write("\n%s\t\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)', 'lr'))

    logger.flush()

    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    elif args.optim == 'radam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            betas=(0.90, 0.999), eps=1e-08, weight_decay=1e-4)
    elif args.optim == 'ranger':
        optimizer = Ranger(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            betas=(0.95, 0.999), eps=1e-08, weight_decay=1e-4)
    elif args.optim == 'adamw':
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    lossTr_list = []
    epoches = []
    mIOU_val_list = []
    global_step = 0


    with open(args.savedir + '/model_config.txt', 'w+') as f:
        f.write('model_name:frozen_model.pt' + '\n')
        f.write('input_height:' + str(args.input_size[0]) + '\n')
        f.write('input_width:' + str(args.input_size[1]) + '\n')
        f.write('input_channel:' + str(3) + '\n')
        f.write('batch_size:' + str(args.batch_size) + '\n')
        f.write('class_num:' + str(args.classes))

    print('=====> beginning training')
    bz_log.info("开始进行网络训练...")
    loss_not_decrease_epoch_num = 0
    value_judgment = np.inf
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr, step = train(
                        args, trainLoader, model, criteria, optimizer, epoch)
        global_step += epoch * step
        print("global ----", global_step)
        lossTr_list.append(lossTr)

        saved_model_value = lossTr

        if args.is_socket:
            data_dict = [epoch + 1, lossTr]
            data_dict = str(data_dict).encode('utf-8')
            socketed.send(data_dict)
        # --------------------
        # # validation
        # if epoch % 50 == 0 or epoch == (args.max_epochs - 1):
        # # if epoch == (args.max_epochs - 1):
        #     epoches.append(epoch)
        #     mIOU_val, per_class_iu = val(args, valLoader, model, global_step)
        #     mIOU_val_list.append(mIOU_val)
        #     # record train information
        #     logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, mIOU_val, lr))
        #     logger.flush()
        #     print("Epoch : " + str(epoch) + ' Details')
        #     print("Epoch No.: %d\tTrain Loss = %.4f\t mIOU(val) = %.4f\t
        #           lr= %.6f\n" % (epoch,lossTr,mIOU_val,lr))
        #
        #     # writer.add_scalar('Loss/Train', lossTr, epoch)
        #     # writer.add_scalar('mIOU/Train', mIOU_val, epoch)
        # else:
        #     # record train information
        #     logger.write("\n%d\t\t%.4f\t\t\t\t%.7f" % (epoch, lossTr, lr))
        #     logger.flush()
        #     print("Epoch : " + str(epoch) + ' Details')
        #     print("Epoch No.: %d\tTrain Loss = %.4f\t lr= %.6f\n" % (epoch, lossTr, lr))
        #     # writer.add_scalar('Loss/Train', lossTr, global_step)
        # --------------------
        # save the model
        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        state = {"epoch": epoch + 1, "model": model.state_dict()}

        # # Individual Setting for save model !!!
        # if epoch >= args.max_epochs - 10:
        #     torch.save(state, model_file_name)
        #     transfer_pth2pt(model, args.savedir, args.classes)
        #
        # elif not epoch % 50:
        #     print('eopch-------------------', epoch)
        #     torch.save(state, model_file_name)
        #     transfer_pth2pt(model, args.savedir, args.classes)

        # 模型保存的条件
        if saved_model_value < value_judgment:
            value_judgment = saved_model_value
            torch.save(state, model_file_name)
            transfer_pth2pt(model, args.savedir, args.classes)

        # early stopping
        if (args.is_early_stop):
            loss_tolerance = 0.0005
            if saved_model_value - value_judgment >= loss_tolerance:
                loss_not_decrease_epoch_num += 1
            else:
                loss_not_decrease_epoch_num = 0
            if loss_not_decrease_epoch_num > 10:
                bz_log.info("early stopping 共训练%d个epoch", epoch)
                bz_log.info("is early stop %s", args.is_early_stop)
                print("early stopping 共训练%d个epoch" % epoch)
                break

    bz_log.info("网络训练结束！")

    if args.is_socket:
        # self.socket.send(b'exit')
        socketed.close()

    logger.close()

def build_socket_connect():
    socketed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("初始化socket")
    # 建立连接:
    try:
        socketed.connect(('127.0.0.1', 9990))
        print("尝试连接9990")
    except socket.error as e:
        raise ValueError('服务连接失败: %s \r\n' % e)
    return socketed

def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """
    model.train()
    epoch_loss = []

    total_batches = len(train_loader)
    print("=====> 每个epoch中迭代次数: ", total_batches)
    st = time.time()

    global_step = 0
    # -----------------------------------
    for iteration, batch in enumerate(train_loader, 0):
        args.per_iter = total_batches
        args.max_iter = args.max_epochs * args.per_iter
        args.cur_iter = epoch * args.per_iter + iteration
        # learming scheduling
        if args.lr_schedule == 'poly':
            lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)),
                                             args.poly_exp)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif args.lr_schedule == 'warmpoly':
            scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter,
                                 cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=args.warmup_iters, power=0.9)

        global_step += 1

        lr = optimizer.param_groups[0]['lr']

        start_time = time.time()
        images, labels, _, _ = batch

        images = images.cuda()
        labels = labels.long().cuda()

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # In pytorch 1.1.0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time
        # socket传参
        print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f '
              'loss: %.3f time:%.2f' % (epoch + 1, args.max_epochs,
                                         iteration + 1, total_batches,
                                         lr, loss.item(), time_taken))

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("训练剩余时间 = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr, global_step


def val(args, val_loader, model, global_step):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    # total_batches = len(val_loader)

    data_list = []
    with torch.no_grad():
        for i, (input, label, size, name) in enumerate(val_loader):
            input_var = input.cuda()
            output = model(input_var)
            output = output.cpu().data[0].numpy()
            gt = np.asarray(label[0].numpy(), dtype=np.uint8)
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            data_list.append([gt.flatten(), output.flatten()])

        # writer.add_images('images', input, global_step)
        # writer.add_image("label", label, global_step)
        # output = output.reshape(-1, output.shape[0], output.shape[1])
        # writer.add_image("predicted", output * 255, global_step)

    meanIoU, per_class_iu = get_iou(data_list, args.classes)
    return meanIoU, per_class_iu



# if __name__ == '__main__':
#     start = timeit.default_timer()
#     args = parse_args()
#     socket = 0
#     args.classes = 4
#     args.input_size = '350,350'
#     args.ignore_label = args.classes
#     args.dataset = "./dataset/new_energy"
#     args.batch_size = 20
#     args.max_epochs = 100
#
#     args.savedir = "./all_model_dir/"
#     args.is_socket = socket
#     if os.path.exists("./runs"):
#         shutil.rmtree("./runs")
#     train_model(args)
#     end = timeit.default_timer()
#     hour = 1.0 * (end - start) / 3600
#     minute = (hour - int(hour)) * 60
#     print("training time: %d hour %d minutes" % (int(hour), int(minute)))
