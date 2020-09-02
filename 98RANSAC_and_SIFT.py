# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:29:53 2019

@author: erio
"""
import os
import numpy as np
import imutils
import cv2
import time


class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them

        (imageB, imageA) = images
        start = time.time()
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        end = time.time()
        print('%.5f s' % (end - start))

        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        start = time.time()
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        end = time.time()
        print('%.5f s' % (end - start))

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        start = time.time()
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        end = time.time()
        print('%.5f s' % (end - start))

        # check to see if the keypoint matches should be visualized
        if showMatches:
            start = time.time()
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)
            end = time.time()
            print('%.5f s' % (end - start))
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    # 接收照片，检测关键点和提取局部不变特征
    # 用到了高斯差分（Difference of Gaussian (DoG)）关键点检测，和SIFT特征提取
    # detectAndCompute方法用来处理提取关键点和特征
    # 返回一系列的关键点
    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    # matchKeypoints方法需要四个参数，第一张图片的关键点和特征向量，第二张图片的关键点特征向量。
    # David Lowe’s ratio测试变量和RANSAC重投影门限也应该被提供。
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    # 连线画出两幅图的匹配
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis


if __name__ == '__main__':
    # load the two images and resize them to have a width of 400 pixels
    # (for faster processing)
    bpath = os.path.join("c:\\", "project", "data", "frame")
    imgname_01 = os.path.join(bpath, "basic.png")
    imgname_02 = os.path.join(bpath, "train", "image008.jpg")
    # imgname_01 = os.path.join(bpath, "1.jpg")
    # imgname_02 = os.path.join(bpath, "2.jpg")
    imageA = cv2.imread(imgname_01)
    # cv_show("imageA", imageA)
    imageB = cv2.imread(imgname_02)

    # imageA = cv2.imread('D:/test1.jpg')
    # imageB = cv2.imread('D:/test2.jpg')
    # imageA = imutils.resize(imageA, width=400)
    # imageB = imutils.resize(imageB, width=400)
    # stitch the images together to create a panorama
    # showMatches=True 展示两幅图像特征的匹配,返回vis
    start = time.time()
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    # show the images
    end = time.time()
    print('%.5f s' % (end - start))
    cv2.imwrite('D:/vis1.jpg', vis)
    cv2.imwrite('D:/result.jpg', result)
# if __name__ == '__main__':
#     # 读取图像
#     bpath = os.path.join("c:\\", "project", "data", "frame")
#     imgname_01 = os.path.join(bpath, "basic.jpg")
#     imgname_02 = os.path.join(bpath, "train", "image008.jpg")
#     imgname_01 = os.path.join(bpath, "1.jpg")
#     imgname_02 = os.path.join(bpath, "2.jpg")
#     imageA = cv2.imread(imgname_01)
#     # cv_show("imageA", imageA)
#     imageB = cv2.imread(imgname_02)
#     # cv_show("imageB", imageB)
#     # 计算SIFT特征点和特征向量
#     (kpsA, featuresA) = detectAndCompute(imageA)
#     (kpsB, featuresB) = detectAndCompute(imageB)
#     # 基于最近邻和随机取样一致性得到一个单应性矩阵
#     (M, matches, status) = matchKeyPoints(kpsA, kpsB, featuresA, featuresB)
#     # 绘制匹配结果
#     # drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
#     # 拼接
#     stich(imageA, imageB, M)
