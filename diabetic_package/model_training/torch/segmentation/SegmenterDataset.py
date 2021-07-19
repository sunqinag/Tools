import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    def __init__(self, img_list,label_list,height_width):
        self.img_list=img_list
        self.label_list=label_list
        self.height_width=height_width

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,i):
        img,label= self.preprocess(Image.open(self.img_list[i])),\
                              self.preprocess(Image.open(self.label_list[i]))
        return {'image': img,'label': label}

    def preprocess(self, pil_img):
        pil_img = pil_img.resize(self.height_width)
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 3:
            return img_nd.transpose((2, 0, 1)) # HWC to CHW
        else:
            return img_nd #label