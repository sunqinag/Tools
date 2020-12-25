import abc

class IFeatureExtractor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, **feature_extractor_params):
        self._feature_maps = []

    @abc.abstractmethod
    def __call__(self, imgs, training):
        pass

    @property
    def feature_maps(self):
        '''
            返回不同feature extractor提取的不同尺度的feature map的list，用于进
        行多尺度特征的融合。
            feature_maps返回的尺度特征由子类决定，但一般返回的是每个步长为2卷积
        或者池化后的结果。每个子类最好返回所有尺度的feature map，供后续的操作进
        行feature map的选取
        '''
        return self._feature_maps
