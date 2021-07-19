import numpy as np

class IoU:
    def __call__(self, bboxes, bbox):
        '''
            计算一组bbox与一个bbox的IoU
            bbox坐标排列顺序为[up, left, down, right]
            bboxes:
                输入的维度任意，但最后一维的维度必须是4
            bbox:
                shape=(4,)
        '''
        if not (isinstance(bbox, np.ndarray) and
                isinstance(bboxes, np.ndarray)):
            raise ValueError('输入参数的类型必须是ndarray')
        if bboxes.shape[-1] != 4:
            raise ValueError('bboxes的最后一个维度必须是4')
        if bbox.shape != (4,):
            raise ValueError('bbox的为shape必须为(4,)')
        self.bboxes = bboxes.reshape([-1, 4])
        self.bbox = bbox
        self.__validate_bbox()
        intersect_area = self.__intersect_area()
        bboxes_area = (self.bboxes[:, 2] - self.bboxes[:, 0]) * (
            self.bboxes[:, 3] - self.bboxes[:, 1])
        bbox_area = (self.bbox[2] - self.bbox[0]) * (
            self.bbox[3] - self.bbox[1])
        union_area = (bboxes_area + bbox_area - intersect_area)
        if (union_area == 0).any():
            if np.logical_and((union_area == 0), (intersect_area != 0)).any():
                raise ZeroDivisionError(
                    'iou计算出现交集面积为0，并集面积不为0的结果')
            else:
                union_area[np.where(union_area == 0)[0]] = 1
        if len(bboxes.shape) > 1:
            return (intersect_area / union_area).reshape(bboxes.shape[: -1])
        else:
            return (intersect_area / union_area)[0]

    def __validate_bbox(self):
        if self.bbox[0] > self.bbox[2]:
            raise ValueError(
                '请保证bbox的坐标顺序是[up, left, down, right]')
        if (np.less(self.bboxes[:, 2], self.bboxes[:, 0]).any() or
                np.less(self.bboxes[:, 3], self.bboxes[:, 1]).any()):
            raise ValueError(
                '请保证bboxes中每个bbox的坐标顺序都是[up, left, down, right]')

    def __intersect_area(self):
        inter_ups = np.maximum(self.bboxes[:, 0], self.bbox[0])
        inter_downs = np.minimum(self.bboxes[:, 2], self.bbox[2])
        inter_lefts = np.maximum(self.bboxes[:, 1], self.bbox[1])
        inter_rights = np.minimum(self.bboxes[:, 3], self.bbox[3])
        intersect_area = (inter_downs - inter_ups) * (
            inter_rights - inter_lefts)
        is_intersect = np.logical_and(inter_rights - inter_lefts > 0,
                                      inter_downs - inter_ups > 0)
        return intersect_area * is_intersect


def corner_2_center(corner_bbox):
    '''
    corner_bbox：
        以左上角和右下角坐标表示的bbox，最后一个维度是4，依次为[up, left, down,
        right]
    返回值：
        以中心和宽高描述的bbox，shape与corner_coord相同，最后一个维度依次是[
        center_y, center_x, height, width]
    '''
    if not isinstance(corner_bbox, np.ndarray):
        raise ValueError('corner_coord的类型必须是ndarray')
    if corner_bbox.shape[-1] != 4:
        raise ValueError('corner_coord的最后一维必须是4')
    corner_bbox_shape = corner_bbox.shape
    corner_bbox = corner_bbox.reshape([-1, 4])
    if np.logical_or(corner_bbox[:, 0] > corner_bbox[:, 2],
                     corner_bbox[:, 1] > corner_bbox[:, 3]).any():
        print(corner_bbox)
        raise ValueError('请保证输入的bbox的坐标顺序为[up, left, down, right]')

    center_y = ((corner_bbox[:, 0] + corner_bbox[:, 2]) / 2).reshape([-1, 1])
    center_x = ((corner_bbox[:, 1] + corner_bbox[:, 3]) / 2).reshape([-1, 1])
    height = (corner_bbox[:, 2] - corner_bbox[:, 0]).reshape([-1, 1])
    width = (corner_bbox[:, 3] - corner_bbox[:, 1]).reshape([-1, 1])
    center_bbox = np.concatenate([center_y, center_x, height, width], axis=1)
    return center_bbox.reshape(corner_bbox_shape)

def b2t_yolo(bbox,
             c_y,
             c_x,
             prior_height,
             prior_width,
             grid_height,
             grid_width):
    '''
        按照Yolo v2和v3的方式将bbox转换成训练时用的描述方式
    bbox:
        bbox在原始图像尺寸下的坐标，最后一维的维度是4，[center_y, center_x,
        height, width]
    c_y:
        grid_cell左上角的行标
    c_x:
        grid_cell左上角的列标
    prior_height:
        预设的prior的高度
    prior_width:
        预设的prior的宽度
    grid_height:
        grid cell的高度
    grid_width:
        grid_cell的宽度
    返回值:
        返回yolo v3训练使用的描述空间下的bbox的坐标[ty, tx, th, tw]，其中ty, tx
        是相对于bbox所在grid cell左上角的相对距离
    '''
    if not isinstance(bbox, np.ndarray):
        raise ValueError('bbox的类型必须是ndarray')
    if bbox.shape[-1] != 4:
        raise ValueError('bbox的最后一维必须是4')
    if (prior_height <= 0 or prior_width <= 0 or
            grid_height <= 0 or grid_width <= 0):
        raise ValueError('请保证输入的高度和宽度参数是 > 0的数')

    b = bbox.reshape([-1, 4]) / np.array([grid_height, grid_width, 1, 1])
    if (b < 0).any():
        raise ValueError('输入参数bbox的坐标不能小于0')
    if np.logical_or((b[:, 0] - c_y) >= 1, (b[:, 0] - c_y) < 0).any():
        raise ValueError('输入参数无法保证sigmoid(tx)在[0, 1)的范围内')
    if np.logical_or((b[:, 1] - c_x) >= 1, (b[:, 1] - c_x) < 0).any():
        raise ValueError('输入参数无法保证sigmoid(tx)在[0, 1)的范围内')

    t_x = (b[:, 1] - c_x).reshape([-1, 1])
    t_y = (b[:, 0] - c_y).reshape([-1, 1])
    t_w = np.log((b[:, 3]+1e-5) / prior_width).reshape([-1, 1])
    t_h = np.log((b[:, 2]+1e-5) / prior_height).reshape([-1, 1])
    return np.concatenate([t_y, t_x, t_h, t_w], axis=1).reshape(bbox.shape)


def t2b_yolo(bbox_t,
             c_y,
             c_x,
             prior_height,
             prior_width,
             grid_height,
             grid_width):
    '''
        按照Yolo v2和v3的方式将bbox转换成训练时用的描述方式
    bbox_t:
        bbox在Yolo v2或v3空间下bbox的坐标，最后一维的维度是4，[ty, tx, th, tw]
    c_y:
        grid_cell左上角的行标
    c_x:
        grid_cell左上角的列标
    prior_height:
        预设的prior的高度
    prior_width:
        预设的prior的宽度
    grid_height:
        grid cell的高度
    grid_width:
        grid_cell的宽度
    返回值:
        返回原始图像坐标空间下的bbox坐标[center_y, center_x, bbox_h, bbox_w]
    '''
    if not isinstance(bbox_t, np.ndarray):
        raise ValueError('bbox_t的类型必须是ndarray')
    if bbox_t.shape[-1] != 4:
        raise ValueError('bbox_t的最后一维必须是4')
    if (prior_height <= 0 or prior_width <= 0 or
            grid_height <= 0 or grid_width <= 0):
        raise ValueError('请保证输入的高度和宽度参数是 > 0的数')

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    t = bbox_t.reshape([-1, 4])
    b_x = ((sigmoid(t[:, 1]) + c_x) * grid_width).reshape([-1, 1])
    b_y = ((sigmoid(t[:, 0]) + c_y) * grid_height).reshape([-1, 1])
    b_w = (prior_width * np.exp(t[:, 3])).reshape([-1, 1])
    b_h = (prior_height * np.exp(t[:, 2])).reshape([-1, 1])
    return np.concatenate([b_y, b_x, b_h, b_w], axis=1).reshape(bbox_t.shape)


def is_bboxes_center_in_grid(bboxes, grid):
    '''
        判断bboxes的中心是否在grid指定的bbox内部
    bboxes:
        输入的bbox list，每个bbox的坐标为[center_y, center_x, height, width]
    grid:
        grid的坐标依次是[up, left, down, right]

    返回值：
        一个bool型的list，True表示对应的bbox中心在这个grid中，False表示不在
    '''
    # 这里设置 <= grid[2] - 1, grid[3] - 1，是因为bbox可能是小数，使用半开半闭
    # 的整数区间可能bbox中心越界grid的情况发生
    is_y_in = np.logical_and(bboxes[:, 0] >= grid[0],
                             bboxes[:, 0] <= grid[2] - 1)
    is_x_in = np.logical_and(bboxes[:, 1] >= grid[1],
                             bboxes[:, 1] <= grid[3] - 1)
    return np.logical_and(is_y_in, is_x_in)
