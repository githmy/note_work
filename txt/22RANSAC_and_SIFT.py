'''
RANSAC算法
一、RANSAC算法
1.RANSAC算法简述
    随机抽样一致算法（RANdom SAmple Consensus,RANSAC）,采用迭代的方式从一组包含离群的被观测数据中估算出数学模型的参数。RANSAC算法被广泛应用在计算机视觉领域和数学领域，例如直线拟合、平面拟合、计算图像或点云间的变换矩阵、计算基础矩阵等方面。     RANSAC算法假设数据中包含正确数据和异常数据（或称为噪声）。正确数据记为局内点（inliers），异常数据记为外点（outliers），也是异常值。同时RANSAC也假设，给定一组正确的数据，存在可以计算出符合这些数据的模型参数的方法。该算法核心思想就是随机性和假设性，随机性是根据正确数据出现概率去随机选取抽样数据，根据大数定律，随机性模拟可以近似得到正确结果。假设性是假设选取出的抽样数据都是正确数据，然后用这些正确数据通过问题满足的模型，去计算其他点，然后对这次结果进行一个评分。

2.RANSAC流程
    RANSAC算法的输入是一组观测数据（往往含有较大的噪声或无效点），一个用于解释观测数据的参数化模型以及一些可信的参数。假设观测数据中包含局内点和局外点，其中局内点近似的被直线所通过，而局外点远离于直线。简单的最小二乘法不能找适应于局内点的直线，原因是最小二乘法尽量去适应包括局外点在内的所有点，相反，RANSA能得出一个仅仅用局内点计算出模型，并且概率还足够高，但是不能保证结果一定正确。RANSAC通过反复选择数据中的一组随机子集来达成目标。被选取的子集被假设为局内点，并用下述方法进行验证： (1)有一个模型适应于假设的局内点，即所有的未知参数都能从假设的局内点计算得出。 (2)用上步得到的模型去测试所有的其它数据，如果某个点适用于估计的模型，认为它也是局内点。 (3)如果有足够多的点被归类为假设的局内点，那么估计的模型就足够合理。 (4)然后，用所有假设的局内点去重新估计模型，因为它仅仅被初始的假设局内点估计过。 (5)最后，通过估计局内点与模型的错误率来评估模型。     上述过程被重复执行固定的迭代次数，每次产生的模型要么因为局内点太少而被舍弃，要么因为比现有的模型更好而被选用。 RANSAC算法用于消除图像误匹配中是寻找一个最佳单应性矩阵H，RANSAC算法从匹配数据集中随机抽出4个样本并保证这四个样本之间不共线。然后利用这个模型测试所有数据，并计算满足这个模型数据点的个数与投影误差（即代价函数）若此模型为最优模型。

3. RANSAC算法优缺点
    RANSAC的优点是能鲁棒的估计模型参数。例如，它能从包含大量局外点的数据集中估计出高精度的参数。     RANSAC的缺点是计算参数的迭代次数没有上限；如果设置迭代次数的上限，得到的结果可能不是最优的结果，甚至可能得到错误的结果。另一个缺点是它要求设置跟问题相关的阀值，而且RANSAC只能从特定的数据集中估计出一个模型，如果存在两个（或多个）模型，RANSAC不能找到别的模型。

二、实验结果
    我分别使用2组图片，每组2个图片，从图片中可以直观的发现经过RANSAC处理后的图片明显减少了图片特征点的的误匹配。
**参考资料**：
https://blog.csdn.net/tymatlab/article/details/79009618

https://blog.csdn.net/lovebyz/article/details/84999282

https://blog.csdn.net/github_39611196/article/details/81164752

https://stackoverflow.com/questions/41504686/opencv-attributeerror-list-object-has-no-attribute-queryidx

https://blog.csdn.net/zhuquan945/article/details/79946868

https://blog.csdn.net/lhanchao/article/details/52849446
'''

def code1():
    '''代码1
    先使用SIFT算法提取特征，完成图像的匹配
    '''
    # % matplotlib inline
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    from PIL import Image

    # 导入两张图片
    imgname_01 = './05.jpg'
    imgname_02 = './06.jpg'
    # 利用现有的cv2模块方法，创建一个SIFT的对象
    sift = cv2.xfeatures2d.SIFT_create()

    # BFmatcher（Brute-Force Matching）暴力匹配   暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配

    # 应用BFMatch暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配er.knnMatch( )函数来进行核心的匹配，knnMatch（k-nearest neighbor classification）k近邻分类算法。
    # 进行特征检测，得到2张图片的特征点和描述子

    img_01 = cv2.imread(imgname_01)
    img_02 = cv2.imread(imgname_02)
    keypoint_01, descriptor_01 = sift.detectAndCompute(img_01, None)
    keypoint_02, descriptor_02 = sift.detectAndCompute(img_02, None)

    bf = cv2.BFMatcher()  # 默认是欧氏距离 cv2.NORM_L2
    # k = 2 返回点集1中每个描述点在点集2中 距离最近的2个匹配点
    matches = bf.knnMatch(descriptor_01, descriptor_02, k=2)

    print(matches[0][0])
    # 调整ratio
    ratio = 0.8
    good = []

    #  m n 相比较各自的距离
    for m, n in matches:
        # 第一个m匹配的是最近邻，第二个n匹配的是次近邻。直觉上，一个正确的匹配会更接近第一个邻居。
        if m.distance < ratio * n.distance:
            good.append([m])
    img5 = cv2.drawMatchesKnn(img_01, keypoint_01, img_02, keypoint_02, good, None, flags=2)

    img_sift = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)  # 灰度处理图像

    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.imshow(img_sift)
    plt.savefig('img_SIFT_02.png')

    cv2.destroyAllWindows()

def code2():
    '''代码2
    使用RANSAC算法对SIFT特征进行改进
    '''
    # % matplotlib inline
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    from PIL import Image

    # 导入两张图片
    imgname_01 = './05.jpg'
    imgname_02 = './06.jpg'
    # 利用现有的cv2模块方法，创建一个SIFT的对象
    sift = cv2.xfeatures2d.SIFT_create()

    img_01 = cv2.imread(imgname_01)
    img_02 = cv2.imread(imgname_02)
    # BFmatcher（Brute-Force Matching）暴力匹配   暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配

    # 应用BFMatch暴力方法找到点集1中每个descriptor在点集2中距离最近的descriptor；找寻到的距离最小就认为匹配er.knnMatch( )函数来进行核心的匹配，knnMatch（k-nearest neighbor classification）k近邻分类算法。
    # 进行特征检测，得到2张图片的特征点和描述子

    img_01 = cv2.imread(imgname_01)
    img_02 = cv2.imread(imgname_02)
    print(img_01.shape)

    keypoint_01, descriptor_01 = sift.detectAndCompute(img_01, None)
    keypoint_02, descriptor_02 = sift.detectAndCompute(img_02, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptor_01, descriptor_02, k=2)
    ratio = 0.9

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            '''
            这里和之前SIFT的语句不一样，因为之前要使用 good.append([m]) (对于cv2.drawMatchesKnn方法) 
            而如果这里good内的每个元素是一个[]则会出现AttributeError: 'list' object has no attribute 'queryIdx
            '''
            good.append(m)

            # 如果找到了足够的匹配，就提取两幅图像中匹配点的坐标，把它们传入到函数中做变换
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
        # print(keypoint_01[m.queryIdx].pt for m in good)
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

        # 返回值中 M 为单应性矩阵。
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
        # print(mask)
        print(mask.shape)
        # 将多维数组转换为一维数组
        matchesMask = mask.ravel().tolist()
        # 获取原图像的高和宽
        h, w, mode = img_01.shape
        # 使用得到的变换矩阵对原图像的四个变换获得在目标图像上的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # 透视变换函数cv2.perspectiveTransform: 输入的参数是两种数组，并返回dst矩阵——扭转矩阵
        dst = cv2.perspectiveTransform(pts, M)
        print("dst")
        print(dst)
        print(dst.shape)
        # cv2.polylines绘制多边形 圈出目标图像 thickness = 3 线段粗细， lineType = cv2.LINE_AA 现在类型
        img_02 = cv2.polylines(img_02, [np.int32(dst)], True, (127, 255, 0), 3, cv2.LINE_AA)
    else:
        print('Can not  matches!')
        matchesMask = None

    draw_params = dict(
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2)
    img3 = cv2.drawMatches(img_01, keypoint_01, img_02, keypoint_02, good, None, **draw_params)

    img_ransac = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)  # 灰度处理图像

    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.imshow(img_ransac)
    plt.savefig('img_SIFT_by_RANASC_02.png')

    cv2.destroyAllWindows()


def code_缩放():
    '''
    裁剪缩放图片 def img_Resize(path, imgNum)
    '''
    import cv2
    img = cv2.imread('06.jpg', 1)
    imgInfo = img.shape
    print(imgInfo)
    height = imgInfo[0]
    width = imgInfo[1]
    mode = imgInfo[2]
    # 1 方法 缩小 2 等比例
    dstHeight = int(height * 0.25)
    dstWidth = int(width * 0.25)
    dst = cv2.resize(img, (dstWidth, dstHeight))
    imgInfo = dst.shape
    print(imgInfo)
    # cv2.imshow('img', dst)
    cv2.imwrite('06.jpg', dst)  # 路径是相对这个.ipynb文件的
    # cv2.waitKey(0)
    print("img_Resize OK!")

if __name__ == '__main__':
    code1()
    code2()
