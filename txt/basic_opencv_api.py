import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def cv2色差():
    tmpfile = os.path.join("..", "data", "maskpaper", "val", "image00291.jpg")
    image = cv2.imread(tmpfile)
    # 色差变换
    plt.imshow(image[:, :, [2, 1, 0]], interpolation='nearest')
    plt.show()


def dim2_code():
    from MyQR import myqr  # 注意大小写
    # myqr.run(words="http://test.htmanage.com/Account/Login")
    myqr.run(words='http://test.htmanage.com/Account/Login', picture='a.png', colorized=True)


def audio_demo():
    import pyaudio
    import wave
    import sys

    CHUNK = 1024
    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)

    wf = wave.open(sys.argv[1], 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)
    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()


def opencv_prepare():
    """
    """
    import os
    import matplotlib.pyplot as plt

    # 直方图均衡化实例
    def equalizeHist():
        tmpfile = os.path.join("C:\project\data\latexhand\hand\images\images_test", "62.png")
        img = cv2.imread(tmpfile, 0)
        # 第二步: 使用cv2.equalizeHist实现像素点的均衡化
        ret = cv2.equalizeHist(img)
        # 第三步：使用plt.hist绘制像素直方图
        plt.subplot(121)
        plt.hist(img.ravel(), 256)
        plt.subplot(122)
        plt.hist(ret.ravel(), 256)
        plt.show()
        # 第四步：绘值均衡化的图像
        plt.imshow(np.hstack((img, ret)), interpolation='nearest')
        plt.show()

    # 使用自适应直方图均衡化
    # 不会使细节消失
    def createCLAHE():
        tmpfile = os.path.join("C:\project\data\latexhand\hand\images\images_test", "62.png")
        # 使用自适应直方图均衡化
        image = cv2.imread(tmpfile, 0)
        # 第一步：实例化自适应直方图均衡化函数
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 第二步：进行自适应直方图均衡化
        clahe = clahe.apply(image)
        # 第三步：使用plt.hist绘制像素直方图
        plt.subplot(121)
        plt.hist(image.ravel(), 256)
        plt.subplot(122)
        plt.hist(clahe.ravel(), 256)
        plt.show()
        # 第四步：绘值均衡化的图像
        plt.imshow(np.hstack((image, clahe)), interpolation='nearest')
        plt.show()

    equalizeHist()
    createCLAHE()


def opencv_basic():
    image = cv2.imread(tmpfile, 0)
    # 1. 尺寸resize
    image = cv2.resize(image, (int(tw * 0.5), int(th * 0.4)), interpolation=cv2.INTER_LINEAR)
    # 2. padding
    top = 0
    bottom = th * 9
    left = 0
    right = tw * 9
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    # 3. 高斯模糊
    blur = cv2.GaussianBlur(img, kernel_size, 0)
    # 4. 得出轮廓 输入必须是bool, true为轮廓筛选的目标
    from skimage import morphology
    skele = morphology.skeletonize(image.astype(np.bool))
    # 4. 得出轮廓 方式2 # 二值图，即黑白的（不是灰度图）,轮廓的检索模式，轮廓的近似办法
    #  cv2.RETR_EXTERNAL表示只检测外轮廓
    # cv2.RETR_LIST检测的轮廓不建立等级关系
    # cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    # cv2.RETR_TREE建立一个等级树结构的轮廓。
    # cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1 - x2），abs（y2 - y1）） == 1
    # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    # cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh - Chinl chain 近似算法
    contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    # 一个是轮廓本身，还有一个是每条轮廓对应的属性。hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
    # 5. 二值化 方式1 固定阈值二值化 thresh： 阈值 maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
    ret, img_thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, img_thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, img_thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, img_thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, img_thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    # 5. 二值化 方式2
    # maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
    # thresh_type： 阈值的计算方法，包含以下2种类型：cv2.ADAPTIVE_THRESH_MEAN_C； cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    # type：二值化操作的类型，与固定阈值函数相同，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV.
    # Block Size： 图片中分块的大小
    # C：阈值计算方法中的常数项
    img_th = cv2.adaptiveThreshold(img, 255, adaptive_method, adaptive_thrstype, 11, 5)
    # 6. 图片合并
    img_add = cv2.addWeighted(t_img, alpha, n_img, beta, gamma)


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
    # thresh：表示的是阈值（起始值） maxval：表示的是最大值 type：表示的是这里划分的时候使用的是什么类型的算法，常用值为0（cv2.THRESH_BINARY）
    ret, binary_img = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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


def movie_py():
    from moviepy.editor import *
    from moviepy.audio.fx import all
    from moviepy.video.compositing.concatenate import concatenate_videoclips

    # 说明： https://github.com/Zulko/moviepy/blob/master/README.rst
    video = VideoFileClip("myHolidays.mp4").subclip(50, 60)

    # Make the text. Many more options are available.
    txt_clip = (TextClip("My Holidays 2013", fontsize=70, color='white')
                .set_position('center')
                .set_duration(10))

    result = CompositeVideoClip([video, txt_clip])  # Overlay text on video
    result.write_videofile("myHolidays_edited.webm", fps=25)  # Many options...
    # 合并
    video = concatenate_videoclips(unit_videos)


def vispy_demo():
    from moviepy.editor import VideoClip
    import numpy as np
    from vispy import app, scene
    from vispy.gloo.util import _screenshot

    canvas = scene.SceneCanvas(keys='interactive')
    view = canvas.central_widget.add_view()
    view.set_camera('turntable', mode='perspective', up='z', distance=2,
                    azimuth=30., elevation=65.)

    xx, yy = np.arange(-1, 1, .02), np.arange(-1, 1, .02)
    X, Y = np.meshgrid(xx, yy)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = lambda t: 0.1 * np.sin(10 * R - 2 * np.pi * t)
    surface = scene.visuals.SurfacePlot(x=xx - 0.1, y=yy + 0.2, z=Z(0),
                                        shading='smooth', color=(0.5, 0.5, 1, 1))
    view.add(surface)
    canvas.show()

    # 用MoviePy转换为动画

    def make_frame(t):
        surface.set_data(z=Z(t))  # 更新曲面
        canvas.on_draw(None)  # 更新Vispy的画布上的 图形
        return _screenshot((0, 0, canvas.size[0], canvas.size[1]))[:, :, :3]

    animation = VideoClip(make_frame, duration=1).resize(width=350)
    animation.write_gif('sinc_vispy.gif', fps=20, opt='OptimizePlus')


def mayavi_demo():
    import numpy as np
    import mayavi.mlab as mlab
    import moviepy.editor as mpy

    duration = 2  # duration of the animation in seconds (it will loop)

    # 用Mayavi制作一个图形

    fig_myv = mlab.figure(size=(220, 220), bgcolor=(1, 1, 1))
    X, Y = np.linspace(-2, 2, 200), np.linspace(-2, 2, 200)
    XX, YY = np.meshgrid(X, Y)
    ZZ = lambda d: np.sinc(XX ** 2 + YY ** 2) + np.sin(XX + d)

    # 用MoviePy将图形转换为动画，编写动画GIF

    def make_frame(t):
        mlab.clf()  # 清掉图形（重设颜色）
        mlab.mesh(YY, XX, ZZ(2 * np.pi * t / duration), figure=fig_myv)
        return mlab.screenshot(antialiased=True)

    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_gif("sinc.gif", fps=20)


def matplotlib_demo():
    import matplotlib.pyplot as plt
    import numpy as np
    from moviepy.video.io.bindings import mplfig_to_npimage
    import moviepy.editor as mpy

    # 用matplotlib绘制一个图形

    duration = 2

    fig_mpl, ax = plt.subplots(1, figsize=(5, 3), facecolor='white')
    xx = np.linspace(-2, 2, 200)  # x向量
    zz = lambda d: np.sinc(xx ** 2) + np.sin(xx + d)  # （变化的）Z向量
    ax.set_title("Elevation in y=0")
    ax.set_ylim(-1.5, 2.5)
    line, = ax.plot(xx, zz(0), lw=3)

    # 用MoviePy制作动（为每个t更新曲面）。制作一个GIF

    def make_frame_mpl(t):
        line.set_ydata(zz(2 * np.pi * t / duration))  # 更新曲面
        return mplfig_to_npimage(fig_mpl)  # 图形的RGB图像

    animation = mpy.VideoClip(make_frame_mpl, duration=duration)
    animation.write_gif("sinc_mpl.gif", fps=20)


def mumpy_demo():
    import urllib
    import numpy as np
    from scipy.ndimage.filters import convolve
    import moviepy.editor as mpy

    #### 从网络上检索地图


    filename = ("http://upload.wikimedia.org/wikipedia/commons/a/aa/"
                "France_-_2011_population_density_-_200_m_%C3%"
                "97_200_m_square_grid_-_Dark.png")
    urllib.urlretrieve(filename, "france_density.png")

    #### 参数和约束条件


    infection_rate = 0.3
    incubation_rate = 0.1

    dispersion_rates = [0, 0.07, 0.03]  # for S, I, R

    # 该内核会模拟人类/僵尸如何用一个位置扩散至邻近位置
    dispersion_kernel = np.array([[0.5, 1, 0.5],
                                  [1, -6, 1],
                                  [0.5, 1, 0.5]])

    france = mpy.ImageClip("france_density.png").resize(width=400)
    SIR = np.zeros((3, france.h, france.w), dtype=float)
    SIR[0] = france.get_frame(0).mean(axis=2) / 255

    start = int(0.6 * france.h), int(0.737 * france.w)
    SIR[1, start[0], start[1]] = 0.8  # infection in Grenoble at t=0

    dt = 1.0  # 一次更新=实时1个小时
    hours_per_second = 7 * 24  # one second in the video = one week in the model
    world = {'SIR': SIR, 't': 0}

    ##### 建模


    def infection(SIR, infection_rate, incubation_rate):
        """ Computes the evolution of #Sane, #Infected, #Rampaging"""
        S, I, R = SIR
        newly_infected = infection_rate * R * S
        newly_rampaging = incubation_rate * I
        dS = - newly_infected
        dI = newly_infected - newly_rampaging
        dR = newly_rampaging
        return np.array([dS, dI, dR])

    def dispersion(SIR, dispersion_kernel, dispersion_rates):
        """ Computes the dispersion (spread) of people """
        return np.array([convolve(e, dispersion_kernel, cval=0) * r
                         for (e, r) in zip(SIR, dispersion_rates)])

    def update(world):
        """ spread the epidemic for one time step """
        infect = infection(world['SIR'], infection_rate, incubation_rate)
        disperse = dispersion(world['SIR'], dispersion_kernel, dispersion_rates)
        world['SIR'] += dt * (infect + disperse)
        world['t'] += dt

    # 用MoviePy制作动画


    def world_to_npimage(world):
        """ Converts the world's map into a RGB image for the final video."""
        coefs = np.array([2, 25, 25]).reshape((3, 1, 1))
        accentuated_world = 255 * coefs * world['SIR']
        image = accentuated_world[::-1].swapaxes(0, 2).swapaxes(0, 1)
        return np.minimum(255, image)

    def make_frame(t):
        """ Return the frame for time t """
        while world['t'] < hours_per_second * t:
            update(world)
        return world_to_npimage(world)

    animation = mpy.VideoClip(make_frame, duration=25)
    # 可以将结果写为视频或GIF（速度较慢）
    # animation.write_gif(make_frame, fps=15)
    animation.write_videofile('test.mp4', fps=20)


def concat_demo():
    import moviepy.editor as mpy
    # 我们使用之前生成的GIF图以避免重新计算动画
    clip_mayavi = mpy.VideoFileClip("sinc.gif")
    clip_mpl = mpy.VideoFileClip("sinc_mpl.gif").resize(height=clip_mayavi.h)
    animation = mpy.clips_array([[clip_mpl, clip_mayavi]])
    animation.write_gif("sinc_plot.gif", fps=20)

    # 或者更有艺术气息一点：
    # 在in clip_mayavi中将白色变为透明
    clip_mayavi2 = (clip_mayavi.fx(mpy.vfx.mask_color, [255, 255, 255])
                    .set_opacity(.4)  # whole clip is semi-transparent
                    .resize(height=0.85 * clip_mpl.h)
                    .set_pos('center'))

    animation = mpy.CompositeVideoClip([clip_mpl, clip_mayavi2])
    animation.write_gif("sinc_plot2.gif", fps=20)


def concat_with_grid_illustraion_demo():
    import moviepy.editor as mpy
    import skimage.exposure as ske  # 改变尺度，直方图
    import skimage.filter as skf  # 高斯模糊

    clip = mpy.VideoFileClip("sinc.gif")
    gray = clip.fx(mpy.vfx.blackwhite).to_mask()

    def apply_effect(effect, title, **kw):
        """ Returns a clip with the effect applied and a title"""
        filtr = lambda im: effect(im, **kw)
        new_clip = gray.fl_image(filtr).to_RGB()
        txt = (mpy.TextClip(title, font="Purisa-Bold", fontsize=15)
               .set_position(("center", "top"))
               .set_duration(clip.duration))
        return mpy.CompositeVideoClip([new_clip, txt])

    # 为原始动画应用4种不同的效果
    equalized = apply_effect(ske.equalize_hist, "Equalized")
    rescaled = apply_effect(ske.rescale_intensity, "Rescaled")
    adjusted = apply_effect(ske.adjust_log, "Adjusted")
    blurred = apply_effect(skf.gaussian_filter, "Blurred", sigma=4)

    # 将片段一起放在2 X 2的网格上，写入一个文件
    final_clip = mpy.clips_array([[equalized, adjusted],
                                  [blurred, rescaled]])
    final_clip.write_gif("test2x2.gif", fps=20)


def concat_with_sequence_illustraion_demo():
    import moviepy.editor as mpy
    import skimage.exposure as ske
    import skimage.filter as skf

    clip = mpy.VideoFileClip("sinc.gif")
    gray = clip.fx(mpy.vfx.blackwhite).to_mask()

    def apply_effect(effect, label, **kw):
        """ Returns a clip with the effect applied and a top label"""
        filtr = lambda im: effect(im, **kw)
        new_clip = gray.fl_image(filtr).to_RGB()
        txt = (mpy.TextClip(label, font="Amiri-Bold", fontsize=25,
                            bg_color='white', size=new_clip.size)
               .set_position(("center"))
               .set_duration(1))
        return mpy.concatenate_videoclips([txt, new_clip])

    equalized = apply_effect(ske.equalize_hist, "Equalized")
    rescaled = apply_effect(ske.rescale_intensity, "Rescaled")
    adjusted = apply_effect(ske.adjust_log, "Adjusted")
    blurred = apply_effect(skf.gaussian_filter, "Blurred", sigma=4)

    clips = [equalized, adjusted, blurred, rescaled]
    animation = mpy.concatenate_videoclips(clips)
    animation.write_gif("sinc_cat.gif", fps=15)


def movie_py_demo():
    from moviepy.editor import *
    from moviepy.audio.fx import all
    # 安装 imagemagick
    # That may also mean that you are using a deprecated version of FFMPEG

    # 字体名字不能含有中文
    FONT_URL = './font/heimi.TTF'

    input_video = "./模板.mp4", output_video = "new_video.mp4"

    # 剪个10s的720x1280px的视频
    background_clip = VideoFileClip(input_video, target_resolution=(720, 1280)).subclip(0, 10)

    # 音乐只要前10s 时间剪辑
    audio_clip = AudioFileClip('yuna.mp3').subclip(0, 10)
    background_clip = background_clip.set_audio(audio_clip)

    # 左下角加文字, 持续10s
    text_clip1 = TextClip('我是左下角', fontsize=30, color='white', font=FONT_URL)
    text_clip1 = text_clip1.set_position(('left', 'bottom'))
    text_clip1 = text_clip1.set_duration(10)

    # 右下角加文字, 持续3s
    text_clip2 = TextClip('我是右下角', fontsize=30, color='white', font=FONT_URL)
    text_clip2 = text_clip2.subclip(0, 3).set_position(('right', 'bottom'))

    image_clip = ImageClip('shuoGG.png')

    # 图片放中间, 从第2s开始播持续6s
    image_clip = image_clip.set_duration(6).set_position('center').set_start(2)
    video = CompositeVideoClip([background_clip, text_clip1, text_clip2, image_clip])

    # 文件写入
    # codec='mpeg4'来使用自己指定的编解码。
    video.write_videofile(output_video)
    myclip.write_videofile('movie.mp4', fps=15)
    myclip.write_videofile('movie.webm')
    myclip.write_videofile('movie.webm', audio=False)  # 不使用音频

    # 调节音量
    video = all.volumex(video, 0.8)

    # 1. 创建clip
    # VIDEO CLIPS
    clip = VideoClip(make_frame, duration=4)  # 自定义动画
    clip = VideoFileClip("my_vedio_file.mp4")  # 文件格式还可以是avi、webm、 gif等
    # 一系列图片创建的clip
    clip = ImageSequenceClip(['imagefile.jpeg', ...], fps=24)
    clip = ImageSequenceClip(images_list, fps=25)
    clip = ImageClip('my_picture.png')  # 文件格式还可以是 png、tiff等
    clip = TextClip('Hello!', font="Amiri-Bold", fintsize=70, color='black')
    # 纯色clip
    clip = ColorClip(size=(460, 380), color=[R, G, B])
    shade = ColorClip(moviesize, color=(0, 0, 0), ismask=True)

    # AUDIO CLIPS
    clip = AudioFileClip("my_audio_file.mp3")  # 文件格式还可以是ogg、wav或者也可以是一个vedio
    clip = AudioArrayClip(numpy_array, fps=44100)  # 一个numpy数组
    clip = AudioClip(make_frame, duration=3)  # 使用一个方法make_frame(t)

    # 2. mask 定义
    maskclip = VideoClip(makeframe, duration=4, ismask=True)
    maskclip = ImageClip('my_mask.jpeg', ismask=True)
    maskclip = VideoFileClip("myvideo.mp4", ismask=True)
    # mask 使用到同样尺寸的myclip上
    myclip.set_mask(mask_clip)
    # mask 转化
    # video_clip都可以通过 clip.to_mask() 转换为一个mask
    # mask也可以通过 my_mask_clip.to_RGB()转换为标准的RGB video clip.

    # 重定义大小
    logo = ImageClip('logo_3.png').resize(width=logosize[0], height=logosize[1])

    # logo 贴到纯色背景上
    screen = logo.on_color(moviesize, color=(0, 0, 0), pos=(frame_fromx, frame_fromy))

    # 3. clip导出为一个gif动画
    my_clip.write_gif('test.gif', fps=12)
    # 保存画面
    myclip.save_frame('frame.png')  # 默认保存第一帧画面
    myclip.save_frame('frame.jpeg', t='01:00:00')  # 保存1h时刻的帧画面

    # 4. 属性
    print(ori_video.fps)
    print(ori_video.duration)
    print(ori_video.h)
    print(ori_video.w)

    # 5. 自定义编辑图像
    def image_func(clip_image):
        print(type(clip_image))
        print(clip_image)
        print(clip_image.shape)

    # 获取第1秒的图像
    aa = ori_video.get_frame(1)
    # 自定义图像编辑 必须要前面加等号
    ori_video = ori_video.fl_image(image_func, apply_to=['mask', 'audio'])
    # 速度加倍
    ori_video = ori_video.fl_time(t_func, apply_to=[], keep_duration=False)
    ori_video.write_images_sequence()


if __name__ == "__main__":
    opencv_demo()
