from diabetic_package.file_operator import bz_path
import numpy as np
import os


def create_data(data_dir, ext):
    folders = np.sort(bz_path.get_subfolder_path(data_dir, ret_full_path=True, is_recursion=False))
    feature_set = []
    labels_set = []
    for i, file in enumerate(folders):
        print(file)
        img_paths = np.sort(bz_path.get_file_path(file + "/img", ret_full_path=True))
        feature_set = np.append(feature_set, img_paths)
        labels_set = np.append(labels_set, np.ones(len(img_paths)) * i)
    # feature_set, labels_set = bz_path.get_all_subfolder_img_label_path_list(data_dir, ret_full_path=True)
    return feature_set, labels_set
