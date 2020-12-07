from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
import skimage.filters.rank as sfr
import os


# 实现最小值滤波
def min_box(image, kernel_size=15):
    # 半径为15的最小值滤波器
    min_image = sfr.minimum(image, disk(kernel_size))
    return min_image


# 第一步，获取暗通道----获取三个颜色通道的最小值
def calculate_dark(image):
    """
    get dark channel
    Args:
        image: numpy type image
    Returns:
        dark channel
    """
    dark = np.minimum(np.minimum(image[:, :, 0], image[:, :, 1]), image[:, :, 2]).astype(np.float32)
    # 如果kernel_size过小会得到过亮的darkimg，得到的A值过大
    #从经验来看，使用更小的kernel能够得到更好的的效果
    dark = min_box(dark, kernel_size=3)
    return dark / 255


# 估计大气光值
def estimation_A(image, dark):
    '''
    1.从暗通道中选取亮度前0.1%的像素
    2.根据1中像素的位置对应找到有雾图像中的像素作为大气光的估计值
    :param image:
    :param dark:
    :return:
    '''
    # calculate the number of pixels of the top 0.1%
    top_pixels_num = int(image.shape[0] * image.shape[1] * 0.001)
    # reshape image and dark into shape (-1, 3) and (-1)
    # copy the data, in case of it changed
    image_copy = image.copy()
    dark_copy = dark.copy()
    image_copy = image_copy.reshape(-1, 3)
    dark_copy = dark_copy.reshape(-1)
    index_of_sort = dark_copy.argsort()[::-1]
    index_of_sort = index_of_sort[0:top_pixels_num]

    A = image_copy[index_of_sort]
    A = np.mean(A, axis=0)#压缩行，对各列求均值得到（3，）
    return A.mean()#得到最终值

#透光率T(X)--矩阵，它依赖于暗通道和大气光值的推导
def estimation_transmission(dark,A,w=0.95):
    """
        get transmission by t(x) = 1 - w*dark/A
        Args:
            dark:  dark channel
            A: air light, scalar
            w: control the dense of haze

        Returns:
            transmission
    """
    transmission = 1-w*dark/A
    transmission = np.maximum(transmission,0.1)
    return transmission

#根据大气散射模型去雾
def de_haze(image,A,trans):
    trans = np.array([trans, trans, trans]).transpose(1, 2, 0)
    clear = (image - A) / trans + A
    return clear


def operate_and_show(image_path):
    name = os.path.split(image_path)[1]
    image = np.array(Image.open(image_path))[:, :, :] / 255
    dark = calculate_dark(image)
    A = estimation_A(image, dark)
    trans = estimation_transmission(dark, A, w=0.95)
    clear = de_haze(image=image, A=A, trans=trans)
    min = np.min(clear)
    max = np.max(clear)
    clear = (clear-min)/(max-min)
    plt.imsave(r'./clear_pic/'+name,clear)
    # real_clear = np.array(Image.open('clear.jpg'))[:, :, :] / 255

    # 展示一番

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Haze")

    plt.subplot(2, 2, 2)
    plt.imshow(dark, cmap="gray")
    plt.title("Dark Channel")

    plt.subplot(2, 2, 3)
    plt.imshow(trans, cmap="gray")
    plt.title("Transmission Map")

    plt.subplot(2, 2, 4)
    plt.imshow(clear)
    plt.title("Predict Image")



    plt.show()
if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    image_folder = r'C:\Users\sunqiang\PycharmProjects\darkimg_quwu\de_pic'
    imgs = os.listdir(image_folder)
    for img in imgs:
        operate_and_show(image_folder+os.sep+img)

    # # 比较一下有雾图像的暗通道和无雾图像的暗通道
    # haze = np.array(Image.open("timg.jpg"))[:, :, :] / 255
    # clear = np.array(Image.open("clear.jpg"))[:, :, :] / 255
    #
    # dark_haze = calculate_dark(haze)
    #
    # dark_clear = calculate_dark(clear)
    #
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(haze)
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(dark_haze, cmap="gray")
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(clear)
    #
    # plt.subplot(2, 2, 4)
    # plt.imshow(dark_clear, cmap="gray")
    #
    # plt.show()

    # # 检验下取得大气光值对不对
    # image = np.array(Image.open("timg.jpg"))[:, :, 0:3] / 255
    # dark = calculate_dark(image)
    # A = estimation_A(image, dark)
    # print("air light A is: {}".format(A))

    # # 可视化一下估计到的透射图t（x
    # image = np.array(Image.open("timg.jpg"))[:, :, 0:3] / 255
    # dark = calculate_dark(image)
    # A = estimation_A(image, dark)
    # trans = estimation_transmission(dark, A, w=0.95)
    # plt.subplot(2, 2, 3)
    # plt.imshow(image)
    #
    # plt.subplot(2, 2, 4)
    # plt.imshow(trans, cmap="gray")
    # plt.title("透视图")
    # plt.show()

    # #最终计算
    # image = np.array(Image.open('timg.jpg'))[:,:,:]/255
    # dark = calculate_dark(image)
    # A = estimation_A(image,dark)
    # trans = estimation_transmission(dark,A,w=0.95)
    # clear=de_haze(image=image,A=A,trans=trans)
    # # plt.imsave('clear.jpg',clear)
    # real_clear = np.array(Image.open('clear.jpg'))[:,:,:]/255
    #
    # #展示一番
    #
    #
    # plt.figure()
    # plt.subplot(2, 3, 1)
    # plt.imshow(image)
    # plt.title("Haze")
    #
    # plt.subplot(2, 3, 2)
    # plt.imshow(dark, cmap="gray")
    # plt.title("Dark Channel")
    #
    # plt.subplot(2, 3, 4)
    # plt.imshow(trans, cmap="gray")
    # plt.title("Transmission Map")
    #
    # plt.subplot(2, 3, 5)
    # plt.imshow(clear)
    # plt.title("Predict Image")
    #
    # plt.subplot(2, 3, 6)
    # plt.imshow(real_clear)
    # plt.title("Ground Truth")
    #
    # plt.show()
