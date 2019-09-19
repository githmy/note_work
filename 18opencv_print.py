#! pip install opencv-python
import sys
from glob import glob
import cv2
import copy
from multiprocessing import Process
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd
import re
import time
import math
import pymysql
from utils.connect_mysql import MysqlDB
from moviepy.editor import *
import moviepy.editor as mpy
from moviepy.audio.fx import all
from moviepy.video.compositing.concatenate import concatenate_videoclips


# 捕获视频
def capture_video():
    # 参数：设备索引，也可以是视频文件
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# 播放视频
def play_video():
    # 参数：设备索引，也可以是视频文件
    cap = cv2.VideoCapture('baby.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('baby', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def save_video():
    # 捕获摄像头
    cap = cv2.VideoCapture(0)
    # 定义编解码器，创建VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def do_mosaic(frame, x, y, w, h, neighbor=9):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param frame: opencv frame
    :param int x :  马赛克左顶点
    :param int y:  马赛克右顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int neighbor:  马赛克每一块的宽
    """
    ypix_n = int(math.ceil(h / neighbor) * neighbor)
    xpix_n = int(math.ceil(w / neighbor) * neighbor)
    for i in range(0, ypix_n, neighbor):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, xpix_n, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(frame, left_up, right_down, color, -1)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 255, 0], 1)
    return frame


def do_mask(frame, logo, x, y, w, h):
    for i in range(0, h):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, w):
            tmplist = logo[i][j].tolist()
            # if tmplist[0] != 0 or tmplist[1] != 0 or tmplist[2] != 0:
            if tmplist[0] != 255 or tmplist[1] != 255 or tmplist[2] != 255:
                frame[i + y][j + x] = logo[i][j]
    return frame


# 只处理视频的主函数
def print_video(infile, outfile):
    # 0. 参数定义
    # print("print_video: ", infile, outfile)
    ratio_heigh, ratio_wide = 9, 5
    # ratio_heigh, ratio_wide = 1, 1
    erode_edge = 5
    # 1. 打开视频
    cap = cv2.VideoCapture(infile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("fps:", fps)
    # 1.2 打开图片
    pic_image = cv2.imread('logo_3.png')
    # print(pic_image.shape)
    # print(pic_image)
    # 是否打开了视频
    if cap.isOpened():
        success, frame = cap.read()
        # 获取图片大小
        [height, width, pixels] = frame.shape
        # 2. 写入视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
        # print(height, width, pixels)
        logo_heigh, logo_wide = int(height / ratio_heigh), int(width / ratio_wide)
        logo_image = cv2.resize(pic_image, (logo_wide, logo_heigh), interpolation=cv2.INTER_CUBIC)
        frame_fromy, frame_fromx = height - logo_heigh, width - logo_wide
        erode_fromy, erode_fromx = frame_fromy - erode_edge, frame_fromx - erode_edge
    else:
        print("error: when read", )
        return 0
    while success:
        # 1. 马赛克
        frame = do_mosaic(frame, frame_fromx, frame_fromy, logo_wide, logo_heigh, neighbor=50)
        # 2. 替换
        frame = do_mask(frame, logo_image, frame_fromx, frame_fromy, logo_wide, logo_heigh)
        # print(logo_image)
        # frame[frame_fromy:, frame_fromx:, :] = logo_image
        # 3. 腐蚀
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # frame[erode_fromy:, erode_fromx:, :] = cv2.erode(frame[erode_fromy:, erode_fromx:, :], kernel, iterations=1)
        # frame[erode_fromy:, erode_fromx:, :] = cv2.inpaint(frame[erode_fromy:, erode_fromx:, :] , mask, 3, cv2.INPAINT_NS)
        # 写视频帧
        videoWriter.write(frame)
        # Capture frame-by-frame
        success, frame = cap.read()
    # 4. 关闭
    videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()


# 处理视频+音频的主函数
# def moviepy_trans(infile, outfile, start_t=0.0, end_t=0.0):
def moviepy_trans(infile, outfile):
    # 0. 参数定义
    ratio_wide, ratio_heigh = 0.14, 0.1
    # 1. 打开视频
    ori_video = VideoFileClip(infile)
    moviesize = (ori_video.w, ori_video.h)
    logosize = (int(ori_video.w * ratio_wide), int(ori_video.h * ratio_heigh))
    # frame_fromx, frame_fromy = moviesize[0] - logosize[0] - 40, moviesize[1] - logosize[1] - 29
    frame_fromx, frame_fromy = moviesize[0] - logosize[0] - 39, moviesize[1] - logosize[1] - 5

    # print("moviesize", moviesize)
    # print("logosize", logosize)
    # print(frame_fromy, frame_fromx)
    # start_t = 243.5
    # end_t = 355
    # ori_video_h = ori_video.subclip(0, start_t)
    #
    # ori_video_m = ori_video.subclip(start_t, end_t)
    # # ori_video_m = ori_video_m.fl_image(
    # #     Mosaic(50, 0, 200, 100, neighbor=10),
    # #     apply_to=['mask'])
    # logomid = ImageClip('../洋葱.png')
    # screen = (logomid.fx(mpy.vfx.mask_color, [254, 254, 254])
    #           .set_opacity(.99)  # whole clip is semi-transparent
    #           .resize(width=40, height=18)
    #           .set_pos((194, 25)))
    # midresult = CompositeVideoClip([ori_video_m, screen], size=moviesize)
    # # midresult.set_duration(ori_video_m.duration).write_videofile(outfile, fps=ori_video.fps)
    # ori_video_t = ori_video.subclip(end_t, None)
    # # 3. 输出
    # result = concatenate_videoclips([ori_video_h, midresult.set_duration(ori_video_m.duration), ori_video_t])
    # result.write_videofile(outfile, fps=ori_video.fps)
    # return
    # 2. 马赛克
    class Mosaic:
        def __init__(self, x, y, w, h, neighbor=9):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.neighbor = neighbor

        def __call__(self, image):
            xpix_n = int(self.w)
            ypix_n = int(self.h)
            x_noint = 0
            y_noint = 0
            x_n = 0
            y_n = 0
            if not isinstance(self.w / self.neighbor, int):
                x_noint = 1
                x_n = int(math.ceil(self.w / self.neighbor))
            else:
                x_n = self.w / self.neighbor
            if not isinstance(self.h / self.neighbor, int):
                y_noint = 1
                y_n = int(math.ceil(self.h / self.neighbor))
            else:
                y_n = self.h / self.neighbor
            for i in range(0, y_n):  # 关键点0 减去neightbour 防止溢出
                for j in range(0, x_n):
                    tmp_recx = self.neighbor
                    if 1 == x_noint and j + 1 == x_n:
                        tmp_recx = xpix_n - j * self.neighbor
                    tmp_recy = self.neighbor
                    if 1 == y_noint and i + 1 == y_n:
                        tmp_recy = ypix_n - i * self.neighbor
                    rect = [j * self.neighbor + self.x, i * self.neighbor + self.y, tmp_recx, tmp_recy]
                    # print(i, j, x_n, y_n)
                    color = image[i * self.neighbor + self.y][j * self.neighbor + self.x].tolist()  # 关键点1 tolist
                    left_up = (rect[0], rect[1])
                    h_tmp = rect[1] + tmp_recy - 1
                    w_tmp = rect[0] + tmp_recx - 1
                    right_down = (w_tmp, h_tmp)  # 关键点2 减去一个像素
                    cv2.rectangle(image, left_up, right_down, color, -1)
            return image

    ori_video = ori_video.fl_image(Mosaic(frame_fromx, frame_fromy, logosize[0] + 20, logosize[1] + 10, neighbor=50),
                                   apply_to=['mask'])
    # 3. 打logo2
    logo = ImageClip('logo_3.png')
    screen = (logo.fx(mpy.vfx.mask_color, [254, 254, 254])
              .set_opacity(.99)  # whole clip is semi-transparent
              .resize(width=logosize[0], height=logosize[1])
              .set_pos((frame_fromx - 20, frame_fromy + 5)))
    # 4. 输出
    # result = CompositeVideoClip([ori_video], size=moviesize)
    result = CompositeVideoClip([ori_video, screen], size=moviesize)
    result.set_duration(ori_video.duration).write_videofile(outfile, fps=ori_video.fps)


# 处理视频+音频的主函数
def moviepy_dehead(infile, outfile, start_t=0.0, end_t=0.0):
    # 1. 读入
    ori_video = VideoFileClip(infile)
    # 3. 输出
    result = CompositeVideoClip([ori_video]).subclip(start_t, ori_video.duration - end_t)
    # result.set_duration(ori_video.duration).write_videofile(outfile, fps=ori_video.fps)
    result.write_videofile(outfile, fps=ori_video.fps)


# 处理视频+音频的主函数
def moviepy_demid(infile, outfile, start_t=0.0, end_t=0.0):
    # 1. 读入
    ori_video = VideoFileClip(infile)
    # 2. 剪切
    ori_video_h = ori_video.subclip(0, start_t)
    ori_video_m = ori_video.subclip(start_t, end_t)
    # ori_video_m = ori_video_m.fl_image(
    #     Mosaic(50, 0, 200, 100, neighbor=10),
    #     apply_to=['mask'])
    ori_video_t = ori_video.subclip(end_t, None)
    # 3. 输出
    result = concatenate_videoclips([ori_video_h, ori_video_t])
    # result = CompositeVideoClip([ori_video]).subclip(start_t, ori_video.duration - end_t)
    # result.set_duration(ori_video.duration).write_videofile(outfile, fps=ori_video.fps)
    result.write_videofile(outfile, fps=ori_video.fps)


# 批量我文件合并
def concat_video_m3u8(source_path, target_full_name):
    # 获取需要转换的路径
    def get_user_path(argv_dir):
        if os.path.isdir(argv_dir):
            return argv_dir
        elif os.path.isabs(argv_dir):
            return argv_dir
        else:
            return False

    # 对转换的TS文件进行排序
    def get_sorted_ts(user_path):
        _, headfi = os.path.split(user_path)
        ts_list = glob(os.path.join(user_path, '*.ts'))
        # base_full_file = glob(os.path.join(user_path, '*.m3u8'))[0]
        base_full_file = os.path.join(user_path, headfi + '.ts')
        base_file, _ = os.path.splitext(os.path.basename(base_full_file))
        dic_file = {}
        for ts in ts_list:
            if os.path.exists(ts):
                file, _ = os.path.splitext(os.path.basename(ts))
                if not file.startswith("all_"):
                    tmpu = re.findall(base_file + '(.*?)$', file)
                    dic_file[ts] = int(tmpu[0])
        dic_ord = sorted(dic_file.items(), key=lambda x: x[1])
        return [os.path.basename(i1[0]) for i1 in dic_ord], base_file

    def convert_m3u8(boxer, o_file_name):
        # cmd_arg = str(ts0)+"+"+str(ts1)+" "+o_file_name
        # print(boxer)
        # tmp = []
        # for ts in boxer:
        #     tmp.append(str(ts[0]))
        cmd_str = '+'.join(boxer)
        exec_str = "copy /b " + cmd_str + ' "' + o_file_name + '"'
        print("copy /b " + cmd_str + ' "' + o_file_name + '"')
        os.system(exec_str)

    user_path = get_user_path(source_path)  # print(user_path)
    # print(os.getcwd())

    if not user_path:
        print("您输入的路径不正确，:-(")
    else:
        boxer, base_file = get_sorted_ts(user_path)
        # all_base_file = "all_{}.ts".format(base_file)
        # if os.path.exists(os.path.join(user_path, all_base_file)):
        #     print('目标文件已存在，跳过。', os.path.join(user_path, all_base_file))
        #     return 0
        if os.path.exists(target_full_name):
            print('目标文件已存在，跳过。', target_full_name)
            return 0
        os.chdir(user_path)
        # convert_m3u8(boxer, all_base_file)
        convert_m3u8(boxer, target_full_name)


def get_merge_paths(source_root, target_root):
    # 1. 遍历
    list_1 = os.listdir(source_root)
    for i1 in list_1:
        i1 = i1.strip()
        list_2 = os.listdir(os.path.join(source_root, i1))
        for i2 in list_2:
            i2 = i2.strip()
            if i2.endswith("DS_Store"):
                continue
            list_3 = os.listdir(os.path.join(source_root, i1, i2))
            for i3 in list_3:
                i3 = i3.strip()
                if i3.startswith("pcM_"):
                    # 输入路径
                    in_content = os.path.join(source_root, i1, i2, i3)
                    # 输出绝对路径
                    out_content = os.path.join(target_root, i1, i2)
                    out_file_full_notail = os.path.join(out_content, i3)
                    yield in_content, out_file_full_notail, out_content
                else:
                    pass


def get_merge_paths_l1(source_root, target_root):
    # 1. 遍历
    list_1 = os.listdir(source_root)
    for i1 in list_1:
        i1 = i1.strip()
        list_2 = os.listdir(os.path.join(source_root, i1))
        for i2 in list_2:
            i2 = i2.strip()
            if i2.endswith("DS_Store"):
                continue
            if i2.startswith("pc"):
                # 输入路径
                in_content = os.path.join(source_root, i1)
                # 输出绝对路径
                out_content = os.path.join(target_root)
                out_file_full_notail = os.path.join(out_content, i1)
                yield in_content, out_file_full_notail, out_content
            else:
                pass


def get_dir_list1(source_root, target_root):
    # 1. 遍历
    list_1 = os.listdir(source_root)
    for i1 in list_1:
        yield source_root, ".".join(i1.split(".")[:-1]), target_root


def get_trans_paths(source_root, target_root):
    # 1. 遍历
    list_1 = os.listdir(source_root)
    for i1 in list_1:
        i1 = i1.strip()
        list_2 = os.listdir(os.path.join(source_root, i1))
        for i2 in list_2:
            i2 = i2.strip()
            if i2.endswith("DS_Store"):
                continue
            list_3 = os.listdir(os.path.join(source_root, i1, i2))
            for i3 in list_3:
                i3 = i3.strip()
                if i3.startswith("pcM_"):
                    # 输入路径
                    in_content = os.path.join(source_root, i1, i2)
                    # 输出绝对路径
                    out_content = os.path.join(target_root, i1, i2)
                    out_file_notail = ".".join(i3.split(".")[:-1])
                    yield in_content, out_file_notail, out_content


def one_task_merge(inhead, outhead, dircontent):
    try:
        # 0. 程序记录库
        config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "333",
            'database': "ycdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)

        print("one_task: ", inhead, outhead)
        # 1. SQL
        req_sql = """SELECT COUNT(*) as cot FROM `merge_status` WHERE dir3="{}";"""
        new_outhead = outhead.replace("\\", "\\\\")
        res_count = mysql.exec_sql(req_sql.format(new_outhead))
        print("had merge:", res_count)
        # if 1:
        if res_count[0]["cot"] == 0:
            add_sql = """insert into `merge_status` (dir3,had_merge) VALUES ("{}", {});"""
            print(add_sql.format(new_outhead, 1))
            # 创建该任务的目录
            if not os.path.exists(dircontent):
                os.makedirs(dircontent)
            # 2. 碎文件合并
            concat_video_m3u8(inhead, outhead + ".ts")
            res_count = mysql.exec_sql(add_sql.format(new_outhead, 1))
            print(res_count)
    except Exception as e:
        print(e)
    print("outhead: {}".format(outhead))


def one_task_trans(indir, fhead, outdir):
    time_s = time.time()
    try:
        # 0. 程序记录库
        config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "333",
            'database': "ycdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)

        print("one_task: ", indir, fhead, outdir)
        # 1. 水印视频处理
        outhead = os.path.join(outdir, fhead)
        new_outhead = outhead.replace("\\", "\\\\")
        req_sql = """SELECT COUNT(*) as cot FROM `trans_status` WHERE dir3="{}" AND had_trans=1;"""
        res_count = mysql.exec_sql(req_sql.format(new_outhead))
        print("had trans:", res_count)
        if res_count[0]["cot"] == 0:
            # 创建该任务的目录
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            moviepy_trans(os.path.join(indir, fhead) + ".ts", outhead + ".mp4")
            # print_video(outhead + ".ts", outhead + ".mp4")
            # 4. 完成的写入数据库
            add_sql = """insert into `trans_status` (dir3,had_trans) VALUES ("{}", {});"""
            print(add_sql.format(new_outhead, 1))
            res_count = mysql.exec_sql(add_sql.format(new_outhead, 1))
            print(res_count)
    except Exception as e:
        print(e)
    print("time: {}s. {}".format(time.time() - time_s, outhead))


def find_not_in(source_root, target_root):
    res = get_merge_paths(source_root, target_root)
    for i1 in res:
        if os.path.isfile(i1[1] + ".ts"):
            pass
        else:
            yield i1[0], i1[1], i1[2]


def main():
    # # 1. 合并
    # source_root = os.path.join("D:\\", "video_data", "洋葱小学数学2")
    # target_root = os.path.join("D:\\", "video_data", "洋葱小学数学2merged")
    # # not_in_res = find_not_in(source_root, target_root)
    # # for id1, i1 in enumerate(not_in_res):
    # #     print(i1[0])
    # # exit(0)
    # res = get_merge_paths_l1(source_root, target_root)
    # for i1 in res:
    #     # p.apply_async(one_task_merge, args=(i1[0], i1[1], i1[2]))
    #     one_task_merge(i1[0], i1[1], i1[2])
    # exit(0)
    # 2. 转换
    # source_root = os.path.join("D:\\", "video_data", "洋葱小学数学merged")
    # target_root = os.path.join("D:\\", "video_data", "洋葱小学数学mergedtransd")
    source_root = os.path.join("D:\\", "video_data", "洋葱小学数学2merged")
    target_root = os.path.join("D:\\", "video_data", "洋葱小学数学2mergedtrand")
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    # res = get_trans_paths(source_root, target_root)
    res = get_dir_list1(source_root, target_root)
    cores = multiprocessing.cpu_count()
    print("cores:", cores)
    p = Pool(int(cores - 7))
    # for i1 in not_in_res:
    for i1 in res:
        # print(i1)
        # one_task_trans(i1[0], i1[1], i1[2])
        p.apply_async(one_task_trans, args=(i1[0], i1[1], i1[2]))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == "__main__":
    # 视频转化单版测试
    filehead = "pcM_5c08f39ae8200039b3d4543d"
    # # start_t = 3.01
    # # end_t = 6.1
    # # start_t = 36
    # # end_t = 80
    start_t = 9
    end_t = 0
    source_path = os.path.join("D:\\", "video_data", "洋葱incom")
    source_root = os.path.join(source_path, filehead)
    mid_path = os.path.join("D:\\", "video_data", "洋葱incommid")
    mid_root = os.path.join(mid_path, filehead)
    if not os.path.exists(mid_path):
        os.makedirs(mid_path)
    target_path = os.path.join("D:\\", "video_data", "洋葱incomtrans")
    target_root = os.path.join(target_path, filehead)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    moviepy_dehead(source_root + ".mp4", mid_root + ".mp4", start_t, end_t)
    # moviepy_demid(mid_root + ".mp4", target_root + ".mp4", start_t, end_t)
    moviepy_trans(mid_root + ".mp4", target_root + ".mp4")

    # # 视频批量转化
    # main()
