# -*- coding: UTF-8 -*-
import dlib
from skimage import io


def get_labels():
    # 使用Dlib的正面人脸检测器frontal_face_detector
    detector = dlib.get_frontal_face_detector()

    # 图片所在路径
    path_pic = "./"
    img = io.imread(path_pic + "mface.jpg")

    # 生成dlib的图像窗口
    win = dlib.image_window()
    win.set_image(img)

    # 使用detector检测器来检测图像中的人脸
    dets = detector(img, 1)
    print("人脸数：", len(dets))

    for i, d in enumerate(dets):
        print("第", i + 1, "个人脸的矩形框坐标：",
              "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())

    # 绘制矩阵轮廓
    win.add_overlay(dets)

    # 保持图像
    dlib.hit_enter_to_continue()

if __name__ == '__main__':
    get_labels()
    print("end")
