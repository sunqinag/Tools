cfg = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "./diabetic_package/pre_train_model/detection/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 3,
    },
    "lr": 0.001,
    "weight_decay": 4e-05,
    "train_path": "../data/coco/trainvalno5k.txt",
    "epochs": 500,
    "img_h": 640,
    "img_w": 640,
    "parallels": [0,1,2,3],         #  config GPU device        #  replace with your working dir
    "log_dir":'log',                #  load checkpoint
    "evaluate_type": "",
    "try": 0,

}
