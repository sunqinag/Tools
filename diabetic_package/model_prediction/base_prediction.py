# ------------------------------------------------------------------
#   !Copyright(C) 2019, 北京博众
#   All right reserved.
#   文件名称：base_prediction
#   摘   要：prediction的基类
#   当前版本: 2019091117
#   作   者：于川汇 陈瑞侠,崔宗会
#   完成日期: 2019-09-11
# ------------------------------------------------------------------
from abc import ABCMeta, abstractmethod

class BasePrediction(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, image):
        pass


