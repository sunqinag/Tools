# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要: 获取指定文件夹下所有指定扩展名的文件路径
#           获取folder中所有子文件夹的路径
#   当前版本 : 2019120617
#   作   者 ：于川汇 陈瑞侠 崔宗会
#   完成日期 : 2019-12-06
# -------------------------------------------------------------------
import os
import numpy as np
import platform
from diabetic_package.log.log import bz_log

def get_file_path(folder, exts=[], ret_full_path=False):
    '''
        作用:
            获取指定文件夹下所有指定扩展名的文件路径
        参数：
            folder       : 指定文件夹路径
            ret_full_path: 是否返回全路径，默认只返回符合条件的扩展名的文件名
            exts         : 扩展名列表
    '''
    if not (ret_full_path == True or ret_full_path == False):
        bz_log.error('输入参数只能是True或者False')
        bz_log.error(ret_full_path)
        raise ValueError('输入参数只能是True或者False')
    if not (os.path.isdir(folder)):
        bz_log.error('输入参数必须是目录或者文件夹')
        bz_log.error(folder)
        raise ValueError('输入参数必须是目录或者文件夹')
    if isinstance(exts, str):
        exts = [exts]
    result = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            (file_name, file_ext) = os.path.splitext(f)
            if (file_ext in exts) or (file_ext[1:] in exts) or (len(exts) == 0):
                if ret_full_path:
                    result.append(os.path.join(root, f))
                else:
                    result.append(f)
    return result


def get_subfolder_path(folder, ret_full_path=True, is_recursion=True):
    """
    获取
    :param folder:父文件夹
    :param ret_full_path:是否返回全路径
    :param is_recursion:是否递归所有文件夹，如果是会将子文件下包含的子文件夹也遍历
    :return:
    """
    '''
        作用：
            获取folder中所有子文件夹的路径
        参数：
            ret_full_path: 是否返回全路径，默认返回子文件夹全路径
    '''

    if not (ret_full_path or ret_full_path == False):
        bz_log.error('输入参数只能是True或者False%s', ret_full_path)
        raise ValueError('输入参数只能是True或者False')
    if not (os.path.isdir(folder)):
        bz_log.error('输入参数必须是目录或者文件夹%s',folder)
        raise ValueError('输入参数必须是目录或者文件夹')
    default_separation_line = '/'
    if (platform.system() == 'Windows'):
        default_separation_line = '\\'
        if folder[-1] != '\\':
            folder = folder + '\\'
    elif (platform.system() == 'Linux'):
        if folder[-1] != '/':
            folder = folder + '/'
    else:
        bz_log.error('目前只支持Windows 系统和Linux系统！')
        raise ValueError('目前只支持Windows 系统和Linux系统！')
    result = []
    if is_recursion:
        for root, dirs, files in os.walk(folder):
            for d in dirs:
                if ret_full_path:
                    result.append(os.path.join(root, d) +
                                  default_separation_line)
                else:
                    result.append(d)
    else:
        bz_log.info("根据系统生成文件路径%s", folder)
        result = os.listdir(folder)
        if ret_full_path:
            result = [folder + folder_dir for folder_dir in result]
    return result


def get_img_label_path_list(img_path, label_path, ret_full_path=False, ext_list=([], [])):
    """
    获取经过排序后img和label的path_list
    :param img_path:图像路径
    :param label_path: label路径
    :param ret_full_path: 是否返回全路径
    :param ext_list:img 和label的文件扩展名,需要是一个包含两个list的tuple或者list,其中的第一个list与img对应,第二个list与label对应
    :return:获取经过排序后img和label的path_list
    """
    img_file_path_list = np.sort(get_file_path(img_path, ret_full_path=ret_full_path, exts=ext_list[0]))
    label_file_path_list = np.sort(get_file_path(label_path, ret_full_path=ret_full_path, exts=ext_list[1]))





    if len(img_file_path_list) == 0 or len(label_file_path_list) == 0:
        bz_log.error('img_path或者label_path为空!%d%d', len(img_file_path_list), len(label_file_path_list))
        raise ValueError('img_path或者label_path为空!')

    if len(img_file_path_list) != len(label_file_path_list):
        bz_log.error('img_path和label_path中文件个数不相等!%d%d', len(img_file_path_list), len(label_file_path_list))
        raise ValueError('img_path和label_path中文件个数不相等!')

    img_file_name_list = np.array(list(map(get_file_name, img_file_path_list)))
    label_file_name_list = np.array(list(map(get_file_name, label_file_path_list)))
    img_not_equal_file_list = img_file_path_list[img_file_name_list != label_file_name_list]
    label_not_equal_file_list = label_file_path_list[img_file_name_list != label_file_name_list]

    if len(img_not_equal_file_list) != 0:
        raise ValueError(img_not_equal_file_list, '和', label_not_equal_file_list, '文件名不一致!')

    return img_file_path_list, label_file_path_list


def get_file_name(file_path, return_ext=False):
    """
    获取文件的文件名和后缀
    :param file_path: 文件名
    :param return_ext: 是否返回文件扩展名
    :return:
    """
    img_full_name = os.path.basename(file_path)
    if '.' not in img_full_name:
        bz_log.error('file_path 不是文件名!%s', file_path)
        raise ValueError('file_path 不是文件名!')
    img_name_list = img_full_name.rsplit('.', 1)
    if return_ext:
        return img_name_list[0], img_name_list[1]
    return img_name_list[0]


def get_all_subfolder_img_label_path_list(folder, ret_full_path=True):
    '''
    获取folder文件夹下所有文件夹内的img和label path_list,如train文件夹下有1,2,3,4等各个类别文件夹，
    每个文件夹下是img和label文件夹，必须是img和label的命名。
    :param folder: 父文件夹
    :param ret_full_path:是否返回全路径
    :return:
    '''
    img_list = np.array([])
    label_list = np.array([])
    default_separation_line = '/'
    if (platform.system() == 'Windows'):
        default_separation_line = '\\'
        if folder[-1] != '\\':
            folder = folder + '\\'
    elif (platform.system() == 'Linux'):
        if folder[-1] != '/':
            folder = folder + '/'
    else:
        bz_log.error('目前只支持Windows 系统和Linux系统！')
        raise ValueError('目前只支持Windows 系统和Linux系统！')
    for subfolder in get_subfolder_path(
            folder, ret_full_path=True, is_recursion=False):
        sub_img_list, sub_label_list = get_img_label_path_list(
            img_path=subfolder + default_separation_line + 'img',
            label_path=subfolder + default_separation_line + 'label',
            ret_full_path=ret_full_path)
        img_list = np.append(img_list, sub_img_list)
        label_list = np.append(label_list, sub_label_list)
    return img_list, label_list

# if __name__ == '__main__':
#     # print(get_file_path('../../', 'jpg'))
#     # print(get_subfolder_path('../../'))
