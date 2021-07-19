from PIL import Image
import torch
import numpy as np


palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 60, 40, 222, 128, 128, 0, 192, 128, 128, 64,
                  64,
                  128, 64, 0, 128, 64, 64, 0, 0, 128, 192]


# zero_pad = 256 * 3 - len(camvid_palette)
# for i in range(zero_pad):
#     camvid_palette.append(0)




def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

