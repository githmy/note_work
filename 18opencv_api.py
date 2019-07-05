import cv2
import numpy as np


def opencv_demo():
    """
    非压缩格式的AVI文件 MPEG1格式
    DIVX格式的AVI      MPEG4
    XVID格式的AVI  也是MPEG4的一种
    WMV9格式的AVI 微软自己的MPEG4
    VP6格式的AVI   也是一种MPEG4
    
    以图像组（GOP）为一个单元的，由I帧B、P帧构成。
    I PBB PBB PBB……结构。
    I帧是一个能够完全记载这一帧全部图像数据的帧。
    P帧去掉与前帧相似的数据而构成的帧。
    B帧是双向预测帧，是根据与前后一帧图像的比较而得到的帧。
    """

    # 2. 写入
    # 格式：http://www.fourcc.org/codecs.php
    # 常用的有 “DIVX"、”MJPG"、“XVID”、“X264"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 第一个参数是要保存的文件的路径
    # fourcc 指定编码器
    # fps 要保存的视频的帧率
    # frameSize 要保存的文件的画面尺寸
    # isColor 指示是黑白画面还是彩色的画面
    isColor = True
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480), isColor)

    path = "路径.格式"  # 记得不要有中文路径
    # 读文件
    img = cv2.imread(path)

    cap = cv2.VideoCapture('baby.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        # 图片翻转
        frame = cv2.flip(frame, 0)
        # 区域覆盖 覆盖的图片，覆盖的位置
        img.paste(img2, (0, 0, 50, 50))
        # 颜色空间转化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = img.shape[0:2]
    # 开始操作
    thresh = cv2.inRange(img, np.array([0, 0, 0]), np.array([192, 192, 192]))
    scan = np.ones((3, 3), np.uint8)
    # 腐蚀特征
    # 方框中，哪一种颜色所占的比重大，9个方格中将都是这种颜色
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imshow('erosion', erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 进行膨胀操作
    # 从左到右，从上到下的平移，如果方框中存在白色，那么这个方框内所有的颜色都是白色
    # 输入的图片, 方框的大小, 迭代的次数
    cor = cv2.dilate(thresh, scan, iterations=1)
    # 2. 还原修补
    specular = cv2.inpaint(img, cor, 5, flags=cv2.INPAINT_TELEA)
    # 2.1 基于快速行进算法
    mask = cv2.imread('../data/mask2.png', 0)
    # 目标图像 = 源图像, 二进制掩码_指示要修复的像素, 像素周围的邻域补绘_如果要修复的区域很薄_使用较小的值会产生较少的模糊。
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    cv2.imshow('1', dst)
    # 2.2 基于流体动力学并使用了偏微分方程
    dst2 = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    cv2.imshow('2', dst)
    color = frame[y][x].tolist()  # 关键点1 tolist
    # 矩形操作
    #             图像   左上角    右下角       颜色   线宽-1为填满
    cv2.rectangle(frame, left_up, right_down, color, -1)

    # 3. 梯度运算：表示的是将膨胀以后的图像 - 腐蚀后的图像
    cv2.morphologyEx(src, cv2.GRADIENT, kernel)

    # 操作结束，下面开始是输出图片的代码
    cv2.namedWindow("image", 0)
    cv2.resizedWindow("image", int(width / 2), int(height / 2))
    cv2.imshow("image", img)

    cv2.namedWindow("modified", 0)
    cv2.resizeWindow("modified", int(width / 2), int(height / 2))
    cv2.imshow("modified", specular)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 显示关闭图片。
    cv2.namedWindow('logo_image', 0)
    cv2.startWindowThread()
    cv2.imshow('logo_image', logo_image)
    cv2.imshow('video_image', frame)
    # 给定的时间内(单位ms)等待用户按键触发，设置waitKey(0),则表示程序会无限制的等待用户的按键事件
    cv2.waitKey(0)
    cv2.destoryAllWindows()


if __name__ == "__main__":
    opencv_demo()
