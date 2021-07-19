# ------------------------------------------------------------------
#   !Copyright(C) 2019, 北京博众
#   All right reserved.
#   文件名称：classification_prediction
#   摘   要：classification的predict 类
#   当前版本: 2019091117
#   作   者：于川汇 陈瑞侠,崔宗会
#   完成日期: 2019-09-11
# ------------------------------------------------------------------
import tensorflow as tf

from .base_prediction import BasePrediction

class ClassificationPrediction(BasePrediction):
    def __init__(self, model_dir):
        self.sess = tf.Session()
        try:
            tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_dir)
            self.feature_placeholder = self.sess.graph.get_tensor_by_name('img:0')
            self.predict_tensor = self.sess.graph.get_tensor_by_name('classes:0')
        except IOError:
            raise IOError(model_dir + '中不存在模型或者模型错误！')

    def predict(self, image):
        if len(image.shape) != 4:
            raise ValueError('输入image的shape必须是4维！')
        return self.sess.run(self.predict_tensor, feed_dict={self.feature_placeholder: image})

    def __del__(self):
        if self.sess is not None:
            self.sess.close()
