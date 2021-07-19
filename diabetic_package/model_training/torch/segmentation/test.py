import numpy as np
from glob import glob
import torch
import cv2
from torch.utils.data import DataLoader
from time import time
from diabetic_package.model_training.torch.segmentation.SegmenterNet import bz_unet
from diabetic_package.model_training.torch.segmentation.SegmenterDataset import SegmentDataset
from diabetic_package.file_operator.bz_path import get_all_subfolder_img_label_path_list
from diabetic_package.machine_learning_common.accuracy.python_accuracy import get_assessment_result

def create_dataloader(test_img_list, test_label_list,batch_size=1, num_workers=4,pin_memory=True):
    test_dataset= SegmentDataset(test_img_list, test_label_list, img_resize)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return test_loader

def create_network(class_num):
    net = bz_unet(3,class_num)
    net.load_state_dict(torch.load(model))
    return net.to(device).eval()

def test(net, test_loader, class_num, show_flag=False,assessment_list=['recall']):
    n,assessment_dict=0,{} #总体的评价指标
    with torch.no_grad():
        for batch in test_loader:
            batch_imgs, batch_true_labels = batch['image'].to(device=device, dtype=torch.float32), \
                                            batch['label'].to(device=device, dtype=torch.int64)
            logits = net(batch_imgs)
            batch_pred=np.squeeze(torch.argmax(torch.softmax(logits,dtype=torch.float32,dim=1),dim=1).cpu().numpy())

            if show_flag:#显示前景
                if len(batch_pred.shape)==2: batch_pred=np.expand_dims(batch_pred,0)
                final_result=np.expand_dims(batch_pred,-1)
                final_result=np.where(final_result==0,0,255).astype(np.uint8)

                for i in range(len(final_result)):
                    cur_img=batch_imgs[i].detach().cpu().permute(1,2,0).numpy()
                    cur_img=np.expand_dims(cur_img[:,:,0]*.299+cur_img[:,:,1]*.587+cur_img[:,:,2]*.114,-1)
                    cur_img=cur_img.astype(np.uint8)
                    cur_label=np.expand_dims(batch_true_labels[i].detach().cpu().numpy(),-1)
                    cur_label=np.where(cur_label==0,0,255).astype(np.uint8)
                    show_img=np.hstack([cur_img,final_result[i],cur_label])
                    cv2.imshow('tmp',show_img)
                    cv2.waitKey(0)

            batch_assessment_dict=get_assessment_result(batch_true_labels.cpu().numpy().reshape(-1),
                                                  batch_pred.reshape(-1),class_num,'eval', assessment_list)
            n += 1
            if not assessment_dict:
                assessment_dict=batch_assessment_dict
            else:
                for k, v in assessment_dict.items():
                    assessment_dict[k]=assessment_dict[k]+batch_assessment_dict[k]
        for k, v in assessment_dict.items():
            assessment_dict[k] = assessment_dict[k]/(n+0.001)
        return assessment_dict

if __name__ == '__main__':
    img_resize = (500, 500)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    #two circle
    # model = glob('seg_circle_dir/all_models_dir/best_checkpoint_dir/*.pth')[0]
    # test_dir = 'twocircle_data/test'
    # class_num = 2

    # #new energy
    # model = glob('seg_model_dir/all_models_dir/best_checkpoint_dir/*.pth')[0]
    # test_dir = 'new_energy/test'
    # class_num = 2

    # mic
    model = glob('seg_model_dir/all_models_dir/best_checkpoint_dir/*.pth')[0]
    test_dir = 'mic/test'
    # test_dir = 'mic/val'
    class_num = 2

    test_img_list, test_label_list=get_all_subfolder_img_label_path_list(test_dir)
    test_loader=create_dataloader(sorted(test_img_list),sorted(test_label_list))
    net=create_network(class_num)

    start_time=time()
    # result = test(net,test_loader,class_num)
    result = test(net,test_loader,class_num,True)
    for k,v in result.items(): print(k,':' ,v)
    print(time()-start_time)