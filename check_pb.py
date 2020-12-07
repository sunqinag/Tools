import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow

def check_ckpy(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)


def check_pb(pb_path,draw_tensorboard=None):
    GRAPH_PB_PATH = pb_path # path to your .pb file
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            # Note: one of the following two lines work if required libraries are available
            # text_format.Merge(f.read(), graph_def)
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            with open('graph.txt','a') as f:
                for i, n in enumerate(graph_def.node):
                    print("Name of the node - %s" % n.name)
                    f.write(n.name+'\n')
            print('pb都区完成，上数据结果写入graph.txt')
    if draw_tensorboard:
        print('pb转化为tensorboard')
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(tf.gfile.FastGFile(pb_path, 'rb').read())
        tf.import_graph_def(graph_def, name='graph')
        summaryWriter = tf.summary.FileWriter('log/', graph)# log存放地址
        print('转化完成，存于log文件夹')

if __name__ == '__main__':
    # check_pb(pb_path='detection_frozen_model.pb',draw_tensorboard=True)
    checkpoint_path = '/media/bzjgsq/6000cd6e-6c7d-4f5d-b5e3-1d2073ae9f32/pycharm_project/code/diabetic_package/pre_train_model/detection/yolov3_coco_demo.ckpt'
    check_ckpy(checkpoint_path)