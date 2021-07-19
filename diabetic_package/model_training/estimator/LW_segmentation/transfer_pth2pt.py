import torch
from diabetic_package.model_training.estimator.LW_segmentation.model.DABNet import DABNet

def transfer_pth2pt(model, result_dir, classes):
    net = DABNet(classes=classes)
    # state_dict = torch.load(model_dir, map_location='CPU')['model']

    net.load_state_dict(model.state_dict())
    net.eval()

    example = torch.rand(1, 3, 400, 400)
    ts = torch.jit.trace(net, example)

    # ts = torch.jit.script(net)
    ts.save(result_dir + "/frozen_model.pt")
