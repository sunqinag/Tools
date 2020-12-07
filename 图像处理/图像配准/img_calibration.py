import cv2
import numpy as np
import math
import os
from tqdm import tqdm
from diabetic_package.file_operator import bz_path


def img_seg(img, thresh):
    _, RedThresh = cv2.threshold(img, thresh, 255,
                                 cv2.THRESH_BINARY)  # 设定阈值（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)  # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def img_binary(input_dir):
    original_img = cv2.imread(input_dir)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）
    binary_img = img_seg(blurred, 68)
    # 开运算（去噪点）
    return original_img, binary_img


def findContours_bbox_angle(opened):
    # contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # 找轮廓
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)
    # print(rect)
    angle = rect[2]
    # print("angle", angle)

    boxpt = cv2.boxPoints(rect)
    # print(boxpt)
    box = np.array([[math.ceil(boxpt[0][0]), math.ceil(boxpt[0][1])],
                    [math.ceil(boxpt[1][0]), math.ceil(boxpt[1][1])],
                    [math.ceil(boxpt[2][0]), math.ceil(boxpt[2][1])],
                    [math.ceil(boxpt[3][0]), math.ceil(boxpt[3][1])]])

    line1 = np.sqrt((box[1][1] - box[0][1]) * (
            box[1][1] - box[0][1]) + (
                            box[1][0] - box[0][0]) * (
                            box[1][0] - box[0][0]))

    line2 = np.sqrt((box[3][1] - box[0][1]) * (
            box[3][1] - box[0][1]) + (
                            box[3][0] - box[0][0]) * (
                            box[3][0] - box[0][0]))
    # // 为了让正方形横着放，所以旋转角度是不一样的。竖放的，给他加90度，翻过来
    if (line1 > line2):
        angle = 90 + angle
    return box, angle, line1, line2


def findContours_img(original_img, opened):
    box, angle, *_ = findContours_bbox_angle(opened)

    # print(box)
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 255, 255), 3)
    rows, cols = original_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    result_img = cv2.warpAffine(original_img, M, (cols, rows))
    res_img = result_img.copy()
    gray_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    binary_img = img_seg(gray_img, 68)
    box1, angle1, w, h = findContours_bbox_angle(binary_img)

    # correct_img = np.zeros([int(h), int(w), 3])
    x = np.array(box1[:, 0])
    min_h = np.min(box1[:, 0])
    max_h = np.max(box1[:, 0])
    min_w = np.min(box1[:, 1])
    max_w = np.max(box1[:, 1])

    correct_img = result_img[min_w:max_w, min_h:max_h, :]

    return correct_img, draw_img


def image_calibration(image_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    image_list = [image_dir + os.sep + image for image in os.listdir(image_dir)]
    for image in image_list:
        print(image)
        original_img, binary_img = img_binary(image)
        result_img, draw_img = findContours_img(original_img, binary_img)
        if result_img is None:
            continue
        image_name = os.path.basename(image)
        new_image = save_dir + os.sep + image_name
        cv2.imwrite(new_image, result_img)


if __name__ == "__main__":
    image_dir = 'ok_use'
    save_dir ='校准'
    image_calibration(image_dir,save_dir)
    # input_dir = "/home/crx/code_project/pad/NG/image/autocode1_10.jpg"
    # original_img, binary_img = img_binary(input_dir)
    # result_img, draw_img = findContours_img(original_img, binary_img)
    #
    # # cv2.imshow("original_img", original_img)
    # # cv2.imwrite("./gray_img.jpg", gray_img)
    # # cv2.imwrite("./RedThresh.jpg", RedThresh)
    # # cv2.imshow("Close", closed)
    # # cv2.imshow("Open", opened)
    # cv2.imwrite("./draw_img.jpg", draw_img)
    # # cv2.imshow("result_img", result_img)
    # cv2.imwrite("./srcimg.jpg", original_img)
    # cv2.imwrite("./correct.jpg", result_img)
