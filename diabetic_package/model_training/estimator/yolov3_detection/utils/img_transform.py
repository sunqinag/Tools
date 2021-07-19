import numpy as np
import cv2


from . import bbox_op

def resize_img_and_bbox(img, bboxes, dst_img_size):
    '''
    resize后保证图像的长宽比不变，不足的地方补0

    img:
        原始图像
    bboxes:
        原始图像上的bbox，[bbox_ind, [center_y, center_x, height, width]]
    dst_img_size:
        resize后图像的尺寸, [height, width]

    返回值：
        resize后图像的大小和对应的bbox，[center_y, center_x, height, width]
    '''
    height_scale = dst_img_size[0] / img.shape[0]
    width_scale = dst_img_size[1] / img.shape[1]
    scale = min(height_scale, width_scale)
    resize_height = int(round(scale * img.shape[0]))
    resize_width = int(round(scale * img.shape[1]))
    resized_img = cv2.resize(img, (resize_width, resize_height))
    before_y = int((dst_img_size[0] - resize_height) / 2)
    after_y = dst_img_size[0] - resize_height - before_y
    before_x = int((dst_img_size[1] - resize_width) / 2)
    after_x = dst_img_size[1] - resize_width - before_x
    pad_width = ((before_y, after_y), (before_x, after_x), (0, 0))
    new_bboxes = np.array(bboxes) * scale
    new_bboxes[:, 0] += before_y
    new_bboxes[:, 1] += before_x
    return (np.pad(resized_img, pad_width, 'constant', constant_values=0),
            new_bboxes)


def padding_img_and_bbox(img, bboxes, dst_img_size):
    '''
        将图像用0补全到dst_img_size的大小，如果图像的size比dst_img_size大，在保
    证图像尺寸比例不变的情况下，先做resize，保证resize后，一条边和dst_img_size
    大小相同，另一条边需要补0

    img:
        原始图像
    bboxes:
        原始图像上的bbox，[bbox_ind, [center_y, center_x, height, width]]
    dst_img_size:
        resize后图像的尺寸, [height, width]

    返回值：
        padding后的图像的大小和其对应的bbox，[center_y, center_x, height,width]
    '''
    if dst_img_size[0] >= img.shape[0] and dst_img_size[1] >= img.shape[1]:
        before_y = int((dst_img_size[0] - img.shape[0]) / 2)
        after_y = dst_img_size[0] - img.shape[0] - before_y
        before_x = int((dst_img_size[1] - img.shape[1]) / 2)
        after_x = dst_img_size[1] - img.shape[1] - before_x
        pad_width = ((before_y, after_y), (before_x, after_x), (0, 0))
        padding_img = np.pad(img, pad_width, 'constant', constant_values=0)
        new_bboxes = bboxes.copy()
        new_bboxes[:, 0] += before_y
        new_bboxes[:, 1] += before_x
        return padding_img, new_bboxes
    else:
        return resize_img_and_bbox(img, bboxes, dst_img_size)


def block_img(img, grids):
    '''
        将输入的图像分块成指定大小，并返回分块儿结果
    img:
        输入图像，[height, width, channel]
    grids:
        要分成的块数，[grid_y, grid_x]

    返回值：
        分块后图像组成list，是按照grids以C-type拼接成的

    注意：为了保证子图像能够存储在一个ndarray中，函数会对输入图像做最小幅度的
          resize从而保证每个子图像的size相同
    '''
    return block_img_with_bbox(img, grids)


def block_img_with_bbox(img, grids, bboxes=None):
    '''
        将输入的图像分块成指定大小，病返回图像分块结果和每个子图像在自己坐标系
    对应的bbox
    img:
        输入图像，shape=[img_height, img_width, channel]
    bboxes:
        bbox列表，shape=[bbox_num, 4]，最后一维依次是[center_y, center_x,
        height, width]
    grids:
        要分成的块数，[grid_y, grid_x]

    返回值：
        分块后图像的list和每个分块图像对应的bbox的list，分块图像在list中是按照
        C-type排列的，如果对应的grid没有bbox，该grid对应list元素是一个shape=
        [0, 4]的ndarray

    注意：为了保证子图像能够存储在一个ndarray中，函数会对输入图像做最小幅度的
          resize从而保证每个子图像的size相同
    '''
    grid_height = np.round(img.shape[0] / grids[0]).astype(np.int)
    grid_width = np.round(img.shape[1] / grids[1]).astype(np.int)
    img_height = grid_height * grids[0]
    img_width = grid_width * grids[1]
    img_resized = cv2.resize(img, (img_width, img_height))
    sub_imgs = []
    sub_img_bboxes_list = []
    for i in range(grids[0]):
        for j in range(grids[1]):
            sub_imgs.append(
                img_resized[i * grid_height: (i + 1) * grid_height,
                            j * grid_width: (j + 1) * grid_width, :]
            )
            if bboxes is not None:
                bbox_inds = np.where(bbox_op.is_bboxes_center_in_grid(
                    bboxes,
                    [i * grid_height,
                     j * grid_width,
                     (i + 1) * grid_height,
                     (j + 1) * grid_width]
                ))[0]
                sub_img_bboxes = bboxes[bbox_inds].copy()
                sub_img_bboxes[:, 0] -= i * grid_height
                sub_img_bboxes[:, 1] -= j * grid_width
                sub_img_bboxes_list.append(sub_img_bboxes)
    if bboxes is None:
        return np.array(sub_imgs)
    return np.array(sub_imgs), sub_img_bboxes_list


def restore_block_img(sub_imgs, grids):
    '''
        将每个输入的sub_imgs拼接成一个整图
    sub_imgs:
        必须保证sub_imgs以图像为元素的一维数组是按照C-type生成的，[sub_img_num,
        sub_img_height, sub_img_width, channel_num]
    grids:
        分块数，[grid_y, grid_x]

    返回值：
        拼接好的图像
    '''
    return restore_block_img_with_bbox(sub_imgs, grids)


def restore_block_img_with_bbox(sub_imgs, grids, bboxes_list=None):
    '''
        将每个输入的sub_imgs拼接成一个整图，并将bboxes_list中对应与每个子图坐标
    系下的bbox还原成原始图像坐标系下bbox坐标
    sub_imgs:
        必须保证sub_imgs以图像为元素的一维数组是按照C-type生成的，[sub_img_num,
        sub_img_height, sub_img_width, channel_num]
    grids:
        分块数，[grid_y, grid_x]
    bboxes_list:
        每个元素对应sub_imgs中一张图中bbox

    返回值：
        拼接好的图像和bbox在拼接图像中坐标
    '''
    if len(sub_imgs) != grids[0] * grids[1]:
        raise ValueError('sub_imgs的个数与grids指定的维度不匹配')
    grid_height = sub_imgs.shape[1]
    grid_width = sub_imgs.shape[2]
    restore_img = np.zeros(
        [grids[0] * grid_height, grids[1] * grid_width, sub_imgs.shape[-1]],
        dtype=sub_imgs.dtype
    )
    restore_img_bboxes = np.array([], dtype=np.float).reshape([0, 4])
    for i in range(grids[0]):
        for j in range(grids[1]):
            restore_img[
                i * grid_height: (i + 1) * grid_height,
                j * grid_width: (j + 1) * grid_width
            ] = sub_imgs[i * grids[1] + j]
            if bboxes_list is not None:
                sub_img_bboxes = bboxes_list[i * grids[1] + j].copy()
                if len(sub_img_bboxes) > 0:
                    sub_img_bboxes[:, 0] += i * grid_height
                    sub_img_bboxes[:, 1] += j * grid_width
                    restore_img_bboxes = np.append(
                        restore_img_bboxes, sub_img_bboxes, axis=0)
    if bboxes_list is None:
        return restore_img
    return restore_img, restore_img_bboxes


def random_crop(img, bboxes, y_scale_range, x_scale_range):
    '''
        用于普通目标检测时的数据增强，截取图像值考虑是截取结果在y_scale_range，
    和x_scale_range指定的范围内，返回在截取范围内的bbox
    '''
    y_scale_range = np.array(y_scale_range)
    x_scale_range = np.array(x_scale_range)
    if y_scale_range.shape != (2,) or x_scale_range.shape != (2,):
        raise ValueError('y_scale_range和x_scale_range必须是2个标量的数组')
    if (
        y_scale_range[0] >= y_scale_range[1] or
        x_scale_range[0] >= x_scale_range[1]
    ):
        raise ValueError(
            'x_scale_range和y_scale_range的第一个元素必须小于第二个元素')
    u_scale, d_scale = __get_random_subrange(y_scale_range)
    l_scale, r_scale = __get_random_subrange(x_scale_range)
    return __crop_img_with_bboxes(
        img, bboxes, u_scale, l_scale, d_scale, r_scale)


def random_crop_accord_to_bbox(
    img, bboxes, y_scale_range, x_scale_range
):
    '''
        随机截取原图上的一个子图返回，这个子图和原图的大小比例在y_scale_range，
    和x_scale_range指定的范围之内，并且返回的子图至少包含一个bbox。
        为保证每个子图里至少含有一个bbox，因此函数会遍历所有的bbox，每遍历一个
    bbox就会截取一张图。
        这个函数主要在准备训练目标检测模型的训练数据时使用
    img:
        输入图像，shape=[height, width, channel_num]
    bboxes:
        原图中目标对应的bbox，shape=[bbox_num, bbox_height, bbox_width, 4]，最
        后一维依次是[center_y, center_x, height, width]
    y_scale_range:
        在y方向上截取的尺寸与原图尺寸比例范围, 第一个元素要小于第二个元素，且均
        在[0, 1]的范围内
    x_scale_range:
        在x方向上截取的尺寸与原图尺寸比例范围，第一个元素要小于第二个元素，且均
        在[0, 1]的范围内

    返回值：
        截取后的图像和对应bbox，输入几个bbox，就返回几个子图
    '''
    y_scale_range = np.array(y_scale_range)
    x_scale_range = np.array(x_scale_range)
    if y_scale_range.shape != (2,) or x_scale_range.shape != (2,):
        raise ValueError('y_scale_range和x_scale_range必须是2个标量的数组')
    if (
        y_scale_range[0] >= y_scale_range[1] or
        x_scale_range[0] >= x_scale_range[1]
    ):
        raise ValueError(
            'x_scale_range和y_scale_range的第一个元素必须小于第二个元素')
    corner_bboxes = bbox_op.center_2_corner(bboxes)
    crop_img_list = []
    crop_bboxes_list = []
    for corner_bbox in corner_bboxes:
        bbox_y_scale_range = (
            corner_bbox[0] / img.shape[0], corner_bbox[2] / img.shape[0])
        bbox_x_scale_range = (
            corner_bbox[1] / img.shape[1], corner_bbox[3] / img.shape[1])
        l_scale, r_scale = __get_random_subrange_include_subrange(
            x_scale_range, bbox_x_scale_range)
        u_scale, d_scale = __get_random_subrange_include_subrange(
            y_scale_range, bbox_y_scale_range)
        crop_img, crop_bboxes = __crop_img_with_bboxes(
            img, bboxes, u_scale, l_scale, d_scale, r_scale)
        crop_img_list.append(crop_img)
        crop_bboxes_list.append(crop_bboxes)
    return crop_img_list, crop_bboxes_list


def flip_img_and_bboxes(img, bboxes):
    '''
    img:
        输入图像，shape=[height, width, channel_num]
    bboxes:
        原图中目标对应的bbox，shape=[bbox_num, bbox_height, bbox_width, 4]，最
        后一维依次是[center_y, center_x, height, width]

    返回值：
        截取后的图像和对应bbox，[center_y, center_x, height, width]
    '''
    flip_code = np.random.randint(-1, 2)
    flip_img = cv2.flip(img, flip_code)
    flip_bboxes = bboxes.copy()
    if flip_code == 0:
        flip_bboxes[:, 0] = img.shape[0] - 1 - bboxes[:, 0]
    elif flip_code == 1:
        flip_bboxes[:, 1] = img.shape[1] - 1 - bboxes[:, 1]
    else:
        flip_bboxes[:, 0] = img.shape[0] - 1 - bboxes[:, 0]
        flip_bboxes[:, 1] = img.shape[1] - 1 - bboxes[:, 1]
    return flip_img, flip_bboxes


def scale_img_and_bboxes(img, bboxes, scale_range):
    '''
    img:
        输入图像，shape=[height, width, channel_num]
    bboxes:
        原图中目标对应的bbox，shape=[bbox_num, bbox_height, bbox_width, 4]，最
        后一维依次是[center_y, center_x, height, width]
    scale_range:
        图像放缩的比例的随机范围

    返回值：
        放缩后的图像和对应bbox，[center_y, center_x, height, width]
    '''
    scale_range = np.array(scale_range)
    if scale_range.shape != (2,):
        raise ValueError('scale_range必须是2个标量的数组')
    if (scale_range[0] >= scale_range[1]):
        raise ValueError(
            'x_scale_range和y_scale_range的第一个元素必须小于第二个元素')
    scale = np.random.rand() * (
        scale_range[1] - scale_range[0]) + scale_range[0]
    scale_height = np.int(np.round(scale * img.shape[0]))
    scale_width = np.int(np.round(scale * img.shape[1]))
    resized_img = cv2.resize(img, (scale_width, scale_height))
    scale_bboxes = bboxes.copy()
    scale_bboxes[:, 0] *= scale_height / img.shape[0]
    scale_bboxes[:, 1] *= scale_width / img.shape[1]
    scale_bboxes[:, 2] = scale_bboxes[:, 2] * scale_height / img.shape[0]
    scale_bboxes[:, 3] = scale_bboxes[:, 3] * scale_width / img.shape[1]
    return resized_img, scale_bboxes


def __get_random_subrange(scale_range):
    '''
        获取[0, 1]范围内的且一个范围[min, max]，且max - min必须落在scale_range
    指定的范围内
    '''
    sub_range = np.random.rand() * (
        scale_range[1] - scale_range[0]) + scale_range[0]
    low = np.random.rand() * (1 - sub_range)
    high = low + sub_range
    return low, high


def __get_random_subrange_include_subrange(scale_range, include_range):
    '''
        获取[0, 1]范围内的且一个范围[min, max]，且max - min必须落在scale_range
    指定的范围内，且返回的range必须包含bbox
    scale_range:
        截取后图像长宽与原图长宽的比例所处范围
    include_range:
        返回的range必须包含include_range

    返回值:
        截取的range
    '''
    if scale_range[1] < include_range[1] - include_range[0]:
        raise ValueError('根据scale_range中取得值包含不下include_range')
    scale_range = (max(include_range[1] - include_range[0], scale_range[0]),
                   scale_range[1])
    sub_range = np.random.rand() * (
        scale_range[1] - scale_range[0]) + scale_range[0]
    random_range_start = min(1 - sub_range, include_range[0])
    random_range_end = max(0, include_range[1] - sub_range)
    low = np.random.rand() * (
        random_range_end - random_range_start) + random_range_start
    high = low + sub_range
    return low, high


def __crop_img_with_bboxes(img, bboxes, u_scale, l_scale, d_scale, r_scale):
    up = u_scale * (img.shape[0] - 1)
    down = d_scale * (img.shape[0] - 1)
    left = l_scale * (img.shape[1] - 1)
    right = r_scale * (img.shape[1] - 1)
    new_bboxes = bboxes.copy()
    new_bboxes[:, 0] = bboxes[:, 0] - up
    new_bboxes[:, 1] = bboxes[:, 1] - left
    corner_new_bboxes = bbox_op.center_2_corner(new_bboxes)
    row_not_in_grid = np.logical_or(
        corner_new_bboxes[:, 0] < 0,
        corner_new_bboxes[:, 2] > down - up + 1
    )
    col_not_in_grid = np.logical_or(
        corner_new_bboxes[:, 1] < 0,
        corner_new_bboxes[:, 3] > right - left + 1
    )
    valid_bboxes_ind = np.where(np.logical_and(
        1 - row_not_in_grid,
        1 - col_not_in_grid
    ))[0]
    if len(valid_bboxes_ind) == 0:
        raise ValueError('bbox为空')
    new_bboxes = new_bboxes[valid_bboxes_ind]
    up = int(np.round(up))
    down = int(np.round(down))
    left = int(np.round(left))
    right = int(np.round(right))
    if down >= img.shape[0] - 1:
        down = img.shape[0] - 2
    if right >= img.shape[1] - 1:
        right = img.shape[1] - 2
    # 这里down和right加1是因为这里down和right是子图像的最后一个坐标
    return img[up: down + 1, left: right + 1, :], new_bboxes
