'''代码2
使用RANSAC算法对SIFT特征进行改进
'''
%matplotlib inline
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


#导入两张图片
imgname_01 = './05.jpg'
imgname_02 = './06.jpg'
#利用现有的cv2模块方法，创建一个SIFT的对象
sift = cv2.xfeatures2d.SIFT_create()


img_01 = cv2.imread(imgname_01)
img_02 = cv2.imread(imgname_02)
# BFmatcher（Brute-Force Matching）暴力匹配   暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配

#应用BFMatch暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配er.knnMatch( )函数来进行核心的匹配，knnMatch（k-nearest neighbor classification）k近邻分类算法。
# 进行特征检测，得到2张图片的特征点和描述子

img_01 = cv2.imread(imgname_01)
img_02 = cv2.imread(imgname_02)
print(img_01.shape)





keypoint_01, descriptor_01 = sift.detectAndCompute(img_01, None)
keypoint_02, descriptor_02 = sift.detectAndCompute(img_02, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)
flann = cv2.FlannBasedMatcher(index_params,search_params)


matches = flann.knnMatch(descriptor_01, descriptor_02, k = 2)
ratio = 0.9

good = []
for m,n in matches:
    if m.distance < ratio * n.distance:
        '''
        这里和之前SIFT的语句不一样，因为之前要使用 good.append([m]) (对于cv2.drawMatchesKnn方法) 
        而如果这里good内的每个元素是一个[]则会出现AttributeError: 'list' object has no attribute 'queryIdx
        '''
        good.append(m)  

#如果找到了足够的匹配，就提取两幅图像中匹配点的坐标，把它们传入到函数中做变换
print(type(good))

min_match_count = 10
if len(good) > 10:
    '''
    获取关键点的坐标
    DMatch.trainIdx - 训练描述子里的描述子索引
    DMatch.queryIdx - 查询描述子里的描述子索引
    将所有好的匹配的对应点的坐标存储下来，就是为了从序列中随机选取4组，以便下一步计算单应矩阵
    '''
    src_pts = np.float32([keypoint_01[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoint_02[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #print(keypoint_01[m.queryIdx].pt for m in good)
    print(src_pts.shape)
    print(dst_pts.shape)
    
    '''
    核心函数：cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold) 利用RANSAC方法计算单应矩阵
    ransacReprojThreshold为阈值，当某一个匹配与估计的假设小于阈值时，则被认为是一个内点，默认是3
    confidence：置信度，默认为0.995
    maxIters：为初始迭代次数，默认是2000
    返回值：M 和 mask
    mask：标记矩阵，标记内点和外点 他和m1，m2的长度一样，当一个m1和m2中的点为内点时，mask相应的标记为1，反之为0
    M(model)：需要求解的单应矩阵
    '''
    ransacReprojThreshold = 8.0

    #返回值中 M 为单应性矩阵。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
    #print(mask)
    print(mask.shape)
    #将多维数组转换为一维数组
    matchesMask = mask.ravel().tolist()
    # 获取原图像的高和宽
    h, w, mode = img_01.shape
    # 使用得到的变换矩阵对原图像的四个变换获得在目标图像上的坐标
    pts = np.float32([[0, 0], [0, h -1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #透视变换函数cv2.perspectiveTransform: 输入的参数是两种数组，并返回dst矩阵——扭转矩阵
    dst = cv2.perspectiveTransform(pts, M)
    print("dst")
    print(dst)
    print(dst.shape)
    # cv2.polylines绘制多边形 圈出目标图像 thickness = 3 线段粗细， lineType = cv2.LINE_AA 现在类型 
    img_02 = cv2.polylines(img_02, [np.int32(dst)], True, (127,255,0), 3, cv2.LINE_AA)
else:
    print('Can not  matches!')
    matchesMask = None

draw_params = dict(
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)
img3 = cv2.drawMatches(img_01, keypoint_01, img_02, keypoint_02, good, None, **draw_params)

img_ransac = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB) #灰度处理图像

plt.rcParams['figure.figsize'] = (20.0, 20.0)
plt.imshow(img_ransac)
plt.savefig('img_SIFT_by_RANASC_02.png')

cv2.destroyAllWindows()













