# -----------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要： 根据skimage.measure.regionprops返回的region的信息计算与
#           region相关的feature
#   当前版本 : 0.0
#   作   者 ：于川汇 陈瑞侠
#   完成日期 : 2018-2-2
# -----------------------------------------------------------------
import numpy as np
from skimage import measure, morphology


def region_props(binary_image, intensity_img=None):
    '''
        返回一个Region的list，list的每个元素是region相关的信息，这些信息只有在
        调用对应属性时才会计算
    '''
    if isinstance(intensity_img, np.ndarray) and (binary_image.shape !=
                                                  intensity_img.shape):
        raise ValueError('输入的二值图和灰度图尺寸不一致')
    unique_val = np.unique(binary_image)
    if not set(unique_val).issubset({0, 1}):
        raise ValueError('输入的图必须为二值图')
    label_img = morphology.label(binary_image, neighbors=8)
    region_props_list = measure.regionprops(
        label_img, intensity_img, coordinates='rc')
    regions = []
    for region in region_props_list:
        regions.append(__RegionProperties(region, intensity_img))
    return regions


def get_region_feature(regions, feature):
    '''
        返回每个regions中region参数feature指定的feature
    '''
    return [r[feature] for r in regions]


def get_region_features(regions, features):
    '''
        返回每个regions中region参数features指定的多个feature，
        返回的list的每个元素是一个ndarray，对应features指定的特征
    '''
    features_values = []
    for feature in features:
        features_values.append(np.array([r[feature] for r in regions]))
    return features_values


class RegionSelector:
    '''
    类说明 ：
            根据输入的regions信息进行区域筛选，并支持多次筛选结果的求交集和并集
    注　意 :
            要求输入的regions是通过调用region_props()函数生成的regions，该
            regions包含的区域特征信息比较全面。

    使用方法：
            输入的regions信息为skimage.measure.regionprops返回的regions的信息。
        筛选函数返回的结果是与输入的regions的list相对应的选中标识位list，这个结果作为
        union和intersect的输入参数进行求并集和交集。
            通过get_selected_region获取选中标识位list指定的region。
            通过selected_region_binary获取选中标识位list指定region对应的二值图。

    警   告：
            skimage.measure.regionprops获取的region信息中，bbox_area，convex_area，
            由于库中的bug返回的是图像面积，在没有确定新版本skimage修改了这个bug
            之前不要使用这两个特征进行筛选。
    '''

    features = ('area', 'bbox_area', 'convex_area', 'eccentricity',
                'equivalent_diameter', 'euler_number', 'extent', 'filled_area',
                'major_axis_length', 'max_intensity', 'min_intensity',
                'mean_intensity', 'minor_axis_length', 'orientation',
                'perimeter', 'solidity', 'circularity', 'compactness')

    def __init__(self, regions):
        self.regions = regions

    def select_region_max(self, feature):
        '''
            选取各个region中参数feature指定的特征的最大值
        '''
        return self.__select_region_extreme(feature, True)

    def select_region_min(self, feature):
        '''
            选取各个region中参数features指定特征的最小值
        '''
        return self.__select_region_extreme(feature, False)

    def select_region(self, features, mins, maxs, and_or):
        '''
            作用：
                选择regions中符合指定特征的区域
            参数：
                features：需要满足的特征名称列表
                mins    ：features中每个特征对应的最小值
                max     ：features中每个特征对应的最大值
                and_or  ：选择区域时features间的逻辑关系，只能是“and”或者“or”
            返回值：
                regions筛选结果的标识位列表
            使用方法：
                如果某个特征只需要选取大于等于某值，maxs中该特征的值需设置为np.inf；
                如果某个特征只需要选取小于等于某值，mins中该特征的值需设置位-np.inf；
                如果某个特征需要设置等于某值，mins和maxs中该特征值均设置为该值。
        '''
        if not (len(features) == len(maxs) and len(maxs) == len(mins)):
            raise ValueError('输入参数maxs，mins，features的维度不匹配')
        if not features or not maxs or not mins:
            raise ValueError('输入的参数为空')
        for min_val, max_val in zip(mins, maxs):
            is_mins_int = isinstance(min_val, int)
            is_mins_float = isinstance(min_val, float)
            is_maxs_int = isinstance(max_val, int)
            ismaxsfloat = isinstance(max_val, float)
            if not(is_mins_int or is_mins_float) or not(
                    is_maxs_int or ismaxsfloat):
                raise ValueError('输入参数类型错误')
        if and_or != 'and' and and_or != 'or':
            raise ValueError('只能输入and或or的逻辑操作')
        self.__check_feature_param(features)
        maxs = np.array(maxs)
        mins = np.array(mins)
        test_sum = np.sum((maxs >= mins).astype(np.int))
        if test_sum != len(maxs):
            raise ValueError('输入的maxs列表的值小于mins列表的值')

        if and_or == 'and':
            logical_op = np.logical_and
        else:
            logical_op = np.logical_or
        min_results = self.__select_regions_greater(features, np.array(mins))
        max_results = self.__select_regions_less(features, np.array(maxs))
        features_result = np.logical_and(min_results, max_results)
        regions_selected_label = features_result[0]
        for k in range(len(features) - 1):
            regions_selected_label = logical_op(regions_selected_label,
                                                features_result[k + 1])
        return regions_selected_label

    def union(self, region_selected_label1, region_selected_label2):
        '''
            作用：
                求两次筛选结果的并集
            参数：
                region_selected_label1： 待合并区域对应的标识位列表1
                region_selected_label2： 待合并区域对应的标识位列表2
            返回值：
                区域并集的标识位列表。
        '''
        self.__check_union_intersect_param(
            region_selected_label1, region_selected_label2)
        return np.logical_or(region_selected_label1, region_selected_label2)

    def intersect(self, region_selected_label1, region_selected_label2):
        '''
            作用：
                求两次筛选结果的交集
            参数：
                region_selected_label1： 待合并区域对应的标识位列表1
                region_selected_label2： 待合并区域对应的标识位列表2
            返回值：
                区域交集的标识位列表。
        '''
        self.__check_union_intersect_param(
            region_selected_label1, region_selected_label2)
        return np.logical_and(region_selected_label1, region_selected_label2)

    def get_selected_regions(self, region_selected_label):
        '''
            作用：
                获取区域标识位list对应的region区域
            参数：
                region_selected_label： 待挑选的区域对应的标识位list
        '''
        if len(region_selected_label) != len(self.regions):
            raise ValueError('输入测参数维度不匹配')
        self.__check_label(region_selected_label)
        selected_region = []
        for label, region in zip(region_selected_label, self.regions):
            if label:
                selected_region.append(region)
        return selected_region

    def selected_region_binary(self, region_selected_label,
                               ori_binary_img_shape, binary_low,
                               binary_high):
        '''
            作用：
                根据输入二直图最大值、最小值生成区域标识位列表对应的二直图
            参数：
                region_selected_label： 待挑选的区域对应的标识位列表
                binary_low           ： 二直图的最小值
                binary_high          ： 二直图的最大值
        '''
        binary_image = np.ones(ori_binary_img_shape) * binary_low
        selected_regions = self.get_selected_regions(region_selected_label)
        for region in selected_regions:
            coords = np.array(region['coords'])
            binary_image[coords[:, 0], coords[:, 1]] = binary_high
        return binary_image

    def __check_feature_param(self, features):
        if not isinstance(features, list):
            features = [features]
        for feature in features:
            if feature not in RegionSelector.features:
                raise ValueError('输入的特征不在支持select的特征列表内')

    def __check_label(self, label):
        unique_label = np.unique(label)
        if not (set(unique_label).issubset({0, 1}) or
                set(label).issubset({True, False})):
            raise ValueError('输入的标签列表必须是0,1或者True，False')

    def __check_union_intersect_param(self,
                                      region_selected_label1,
                                      region_selected_label2):
        if (len(region_selected_label1) != len(self.regions) or
                len(region_selected_label2) != len(self.regions)):
            raise ValueError('输入参数label1和label2的维度和region的维度不匹配')
        self.__check_label(region_selected_label1)
        self.__check_label(region_selected_label2)

    def __select_regions_greater(self, features, values):
        '''
            作用：
                筛选出特征列表中大于等于最小值列表的特征
            参数：
                features： 特征列表
                values  ： 每个特征待比较的值
        '''
        return get_region_features(self.regions, features) >= \
            (np.array(values)).reshape(len(features), 1)

    def __select_regions_less(self, features, value):
        '''
            作用：
                筛选出特征列表中小于等于最大值列表的特征
            参数：
                features： 特征列表
                values  ： 每个特征待比较的值
        '''
        return get_region_features(self.regions, features) <=\
            value.reshape(len(features), 1)

    def __select_region_extreme(self, feature, is_max):
        self.__check_feature_param(feature)
        feature_vals = get_region_features(self.regions, [feature])[0]
        if is_max:
            extreme_func = np.max
        else:
            extreme_func = np.min
        extreme_val = np.int(extreme_func(feature_vals))
        return self.select_region(
            [feature], [extreme_val], [extreme_val], 'and')


class __RegionProperties:
    def __init__(self, skimage_region_props, ori_img):
        self.__region_props = skimage_region_props
        self.__ori_img = ori_img

    @property
    def area(self):
        return self.__region_props['area']

    @property
    def bbox(self):
        return self.__region_props['bbox']

    @property
    def bbox_area(self):
        return self.__region_props['bbox_area']

    @property
    def centroid(self):
        return self.__region_props['centroid']

    @property
    def convex_area(self):
        return self.__region_props['convex_area']

    @property
    def convex_image(self):
        return self.__region_props['convex_image']

    @property
    def coords(self):
        return self.__region_props['coords']

    @property
    def eccentricity(self):
        return self.__region_props['eccentricity']

    @property
    def equivalent_diameter(self):
        return self.__region_props['equivalent_diameter']

    @property
    def euler_number(self):
        return self.__region_props['euler_number']

    @property
    def extent(self):
        return self.__region_props['extent']

    @property
    def filled_area(self):
        return self.__region_props['filled_area']

    @property
    def filled_image(self):
        return self.__region_props['filled_image']

    @property
    def image(self):
        return self.__region_props['image']

    @property
    def inertia_tensor(self):
        return self.__region_props['inertia_tensor']

    @property
    def inertia_tensor_eigvals(self):
        return self.__region_props['inertia_tensor_eigvals']

    @property
    def intensity_image(self):
        return self.__region_props['intensity_image']

    @property
    def label(self):
        return self.__region_props['label']

    @property
    def local_centroid(self):
        return self.__region_props['local_centroid']

    @property
    def major_axis_length(self):
        return self.__region_props['major_axis_length']

    @property
    def max_intensity(self):
        return self.__region_props['max_intensity']

    @property
    def mean_intensity(self):
        return self.__region_props['mean_intensity']

    @property
    def min_intensity(self):
        return self.__region_props['min_intensity']

    @property
    def minor_axis_length(self):
        return self.__region_props['minor_axis_length']

    @property
    def moments(self):
        return self.__region_props['moments']

    @property
    def moments_central(self):
        return self.__region_props['moments_central']

    @property
    def moments_hu(self):
        return self.__region_props['moments_hu']

    @property
    def moments_normalized(self):
        return self.__region_props['moments_normalized']

    @property
    def orientation(self):
        return self.__region_props['orientation']

    @property
    def perimeter(self):
        return self.__region_props['perimeter']

    @property
    def solidity(self):
        return self.__region_props['solidity']

    @property
    def weighted_centroid(self):
        return self.__region_props['weighted_centroid']

    @property
    def weighted_local_centroid(self):
        return self.__region_props['weighted_local_centroid']

    @property
    def weighted_moments(self):
        return self.__region_props['weighted_moments']

    @property
    def weighted_moments_central(self):
        return self.__region_props['weighted_moments_central']

    @property
    def weighted_moments_hu(self):
        return self.__region_props['weighted_moments_hu']

    @property
    def weighted_moments_normalized(self):
        return self.__region_props['weighted_moments_normalized']

    @property
    def circularity(self):
        return self.area / (np.square(self.major_axis_length / 2) * np.pi)

    @property
    def compactness(self):
        return np.square(self.perimeter) / (4 * self.area * np.pi)

    @property
    def variance(self):
        return np.var(self.__ori_img[self.coords[:, 0], self.coords[:, 1]])

    @property
    def standard_deviation(self):
        return np.std(self.__ori_img[self.coords[:, 0], self.coords[:, 1]])

    @property
    def median(self):
        foreground = self.__ori_img[self.coords[:, 0], self.coords[:, 1]]
        foreground = np.sort(foreground)
        return foreground[np.int(len(foreground) / 2)]

    def __getitem__(self, key):
        return getattr(self, key)
