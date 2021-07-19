# -----------------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称:
#   摘   要: 实现region_operator模块的单元测试
#   当前版本: 0.0
#   作   者： 陈瑞侠
#   完成日期: 2018-2-2
# -----------------------------------------------------------------------

import unittest
import numpy as np
from image_processing_operator import region_operator
import cv2
class test_region_operator(unittest.TestCase):
    #资源初始化
    def setUp(self):
        intensity_img = np.zeros((1000, 1000))
        x = 100
        intensity_img[2*x:4*x, 2*x:4*x] = 255
        intensity_img[2 * x:3 * x, 2 * x:3 * x] = 150
        intensity_img[5*x:7*x, 5*x:8*x] = 255
        cv2.circle(intensity_img, (900,900), 90, (255, 255, 255), -1)
        intensity_img[0:x, 0:x] = 125
        _,binary_img = cv2.threshold(intensity_img,100, 1,cv2.THRESH_BINARY)
        self.binary_img = binary_img
        self.intensity_img = intensity_img
        self.regions = region_operator.region_props(self.binary_img,
                                                    intensity_img)
        self.region_selector = region_operator.RegionSelector(self.regions,
                                                              self.binary_img)
    def tearDown(self):
        pass

    def test_region_props(self):
        FEATURES = ['area', 'bbox', 'bbox_area', 'centroid', 'convex_area',
                    'convex_image', 'coords', 'eccentricity',
                    'equivalent_diameter',
                    'euler_number', 'extent', 'filled_area', 'filled_image',
                    'image',
                    'inertia_tensor', 'inertia_tensor_eigvals',
                    'intensity_image',
                    'label', 'local_centroid', 'major_axis_length',
                    'max_intensity',
                    'mean_intensity', 'min_intensity', 'minor_axis_length',
                    'moments',
                    'moments_central', 'moments_hu', 'moments_normalized',
                    'orientation', 'perimeter', 'solidity', 'weighted_centroid',
                    'weighted_local_centroid', 'weighted_moments',
                    'weighted_moments_central', 'weighted_moments_hu',
                    'weighted_moments_normalized', 'circularity', 'compactness',
                    'variance', 'standard_deviation']
        self.assertRaises(ValueError,
                          region_operator.region_props,
                          self.intensity_img,
                          self.intensity_img)
        self.assertRaises(ValueError,
                          region_operator.region_props,
                          self.intensity_img)
        self.assertRaises(ValueError,
                          region_operator.region_props,
                          self.intensity_img,
                          None)
        self.assertRaises(ValueError,
                          region_operator.region_props,
                          None)
        self.assertRaises(ValueError,
                          region_operator.region_props,
                          self.intensity_img,
                          self.intensity_img)

        self.assertRaises(ValueError,
                          region_operator.region_props,
                          self.binary_img,
                          self.binary_img[100:800,:])
        for r in self.regions:
            for feature in FEATURES:
                self.assertIsNotNone(r[feature], '特征对应的值不应该为空')

    def test_get_region_features(self):
        features = [ 'circularity', 'compactness', 'variance',
                     'standard_deviation', 'median']
        feature_value = region_operator.get_region_features(self.regions,
                                                            features)
        self.assertEqual(len(feature_value), len(features),
                         '得到的特征值的维度和特征个数不匹配')
        cal_value = [[0.955, 0.955, 0.6366, 1], [1.247, 1.260, 1.315, 1.110],
                     [0,2067.188,0,0],[0,45.466,0,0], [125, 255, 255, 255]]
        for i in range(len(features)):
            self.assertEqual(len(feature_value[i]),
                             len(self.regions),
                             '得到的每个维度的特征的个数和region的个数不匹配')
            for j in range(len(self.regions)):
                self.assertAlmostEqual(feature_value[i][j],
                                       cal_value[i][j] ,
                                       2,
                                       '计算的数值差异过大')

    def test_union(self):
        region_selected_label1 = [0, 1, 0]
        region_selected_label2 = [0, 1, 1, 1]
        self.assertRaises(ValueError,
                          self.region_selector.union,
                          region_selected_label1,
                          region_selected_label2)
        region_selected_label1 = [0, 1, 0]
        region_selected_label2 = [0, 1, 1]
        self.assertRaises(ValueError,
                          self.region_selector.union,
                          region_selected_label1,
                          region_selected_label2)
        region_selected_label1 = [0, 2, 0]
        region_selected_label2 = [0, 1, 1]
        self.assertRaises(ValueError,
                          self.region_selector.union,
                          region_selected_label1,
                          region_selected_label2)
        label1 = [True, True, False]
        label2 = [False, True]
        self.assertRaises(ValueError,
                          self.region_selector.union,
                          label1,
                          label2)
        label1 = [True, None, False]
        label2 = [False, True, False]
        self.assertRaises(ValueError,
                          self.region_selector.union,
                          label1,
                          label2)
        label_list1 = [ False, False, True, False]
        label_list2 = [False, False, True, False]
        label_list = self.region_selector.union(label_list1, label_list2)
        self.assertListEqual(label_list.tolist(),
                             [False, False, True, False],
                             '得到的结果不正确')
        label_list1 = [0, 1, 0, 1]
        label_list2 = [1, 0, 0, 1]
        label_list = self.region_selector.union(label_list1, label_list2)
        self.assertListEqual(label_list.tolist(),
                             [True, True, False,True],
                             '得到的结果不正确')
    def test_intersect(self):
        region_selected_label1 = [0, 1, 0]
        region_selected_label2 = [0, 1, 1, 1]
        self.assertRaises(ValueError,
                          self.region_selector.intersect,
                          region_selected_label1,
                          region_selected_label2)
        region_selected_label1 = [0, 1, 0]
        region_selected_label2 = [0, 1, 1]
        self.assertRaises(ValueError,
                          self.region_selector.intersect,
                          region_selected_label1,
                          region_selected_label2)
        region_selected_label1 = [0, 2, 0]
        region_selected_label2 = [0, 1, 1]
        self.assertRaises(ValueError,
                          self.region_selector.intersect,
                          region_selected_label1,
                          region_selected_label2)
        label1 = [False, True, False]
        label2 = [False, True]
        self.assertRaises(ValueError,
                          self.region_selector.intersect,
                          label1,
                          label2)
        label1 = [True, None, False]
        label2 = [False, True, False]
        self.assertRaises(ValueError,
                          self.region_selector.intersect,
                          label1,
                          label2)
        label1 = [True, None, False]
        label2 = [False, True, None]
        self.assertRaises(ValueError,
                          self.region_selector.intersect,
                          label1,
                          label2)
        label_list1 = [0, 1, 0, 1]
        label_list2 = [1, 0, 0, 1]
        label_list = self.region_selector.intersect(label_list1, label_list2)
        self.assertListEqual(label_list.tolist(),
                             [False, False, False, True],
                             '得到的结果不正确')
        label_list1 = [False, False, True, False]
        label_list2 = [False, False, True, False]
        label_list = self.region_selector.intersect(label_list1, label_list2)
        self.assertListEqual(label_list.tolist(),
                             [False, False, True, False],
                             '得到的结果不正确')
    def test_select_region(self):
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area'],
                          [60000], [10000],
                          '输入测最大值必须大于等于最小值')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'],
                          [100], [40000, 50000],
                          'and')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'],
                          [10000, 20000],
                          [40000, 50000],
                          'not')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,\
                          ['area', 'perimeter'],
                          [],
                          [40000, 50000],
                          'and')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,\
                          ['area', 'perimeter'],
                          ['aa', 20000],
                          [40000, 50000],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'],
                          ['aa', 'aa'],
                          [40000, 50000],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'],
                          [10000, 20000],
                          ['aa', 'aa'],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'],
                          [None, 'aa'],
                          [40000, 50000],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'],
                          [10000, 20000],
                          [None, None],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'], [100, 20000],
                          [None, 50000],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          [],
                          [100, 20000],
                          [None, 50000],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          [None],
                          [100, 20000],
                          [None, 50000],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'],
                          [np.inf, np.inf],
                          [-np.inf, -np.inf],
                          'or')
        self.assertRaises(ValueError,
                          self.region_selector.select_region,
                          ['area', 'perimeter'],
                          [-np.inf, np.inf],
                          [np.inf, -np.inf],
                          'or')
        label_list3 = self.region_selector.select_region(['area', 'perimeter'],
                                                         [-np.inf, 900],
                                                         [40000, np.inf],
                                                         'or')
        self.assertListEqual(label_list3.tolist(),
                             [True, True, True, True],
                             '得到的结果和正确结果不一致')
        label_list4 = self.region_selector.select_region(['area', 'perimeter'],
                                                         [-np.inf, np.inf],
                                                         [-np.inf, np.inf],
                                                         'or')
        self.assertListEqual(label_list4.tolist(),
                             [False, False, False, False],
                             '得到的结果和正确结果不一致')
        label_list5 = self.region_selector.select_region(['area', 'perimeter'],
                                                         [-np.inf, -np.inf],
                                                         [np.inf, np.inf],
                                                         'or')
        self.assertListEqual(label_list5.tolist(),
                             [True, True, True, True],
                             '得到的结果和正确结果不一致')
        max_area = 60000
        label_list1 = self.region_selector.select_region(['area', 'perimeter'],
                                                         (200, 600),
                                                         (600, 8000),
                                                         'or')
        label_list2 = self.region_selector.select_region(['area'],
                                                         [max_area],
                                                         [max_area],
                                                         'and')
        self.assertListEqual(label_list1.tolist(),
                             [False, True, True, False],
                             '得到的结果和正确结果不一致')
        self.assertListEqual(label_list2.tolist(),
                             [False, False, True, False],
                             '得到的结果和正确结果不一致')

    def test_get_selected_region(self):
        self.assertRaises(ValueError,
                          self.region_selector.select_region_max,
                          [0, 1])
        self.assertRaises(ValueError,
                          self.region_selector.select_region_max,
                          [True, False, True])
        self.assertRaises(ValueError,
                          self.region_selector.select_region_max,
                          [True, False, None, True])
        label_list = [True, False, True, False]
        select_region = self.region_selector.get_selected_regions(label_list)
        self.assertEqual(select_region[0],
                         self.regions[0],
                         '得到的结果和正确结果不一致')
        self.assertEqual(select_region[1],
                         self.regions[2],
                         '得到的结果和正确结果不一致')

    def test_select_region_max(self):
        value = self.region_selector.select_region_max('area')
        self.assertListEqual(value.tolist(),
                             [False, False, True, False],
                             '得到的结果和正确结果不一致')
        self.intensity_img[800:1000,0:300] = 125
        _, binary_img = cv2.threshold(
                    self.intensity_img, 100, 1, cv2.THRESH_BINARY)
        binary_img = binary_img
        intensity_img = self.intensity_img
        regions = region_operator.region_props(binary_img, intensity_img)
        region_selector = region_operator.RegionSelector(regions, binary_img)
        value = region_selector.select_region_max('area')
        self.assertListEqual(value.tolist(),
                             [False, False, True, True, False],
                             '得到的结果和正确结果不一致')

    def test_select_region_min(self):
        value = self.region_selector.select_region_min('area')
        self.assertListEqual(value.tolist(),
                             [True, False, False, False],
                             '得到的结果和正确结果不一致不正确')
        self.intensity_img[0:100, 900:1000] = 125
        _, binary_img = cv2.threshold(
                            self.intensity_img, 100, 1, cv2.THRESH_BINARY)
        binary_img = binary_img
        intensity_img = self.intensity_img
        regions = region_operator.region_props(binary_img, intensity_img)
        region_selector = region_operator.RegionSelector(regions, binary_img)
        value = region_selector.select_region_min('area')
        self.assertListEqual(value.tolist(),
                             [True, True , False, False, False],
                             '得到的结果和正确结果不一致')

if __name__ == '__main__':
   unittest.main()





