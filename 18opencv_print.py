#! pip install opencv-python
import sys
from glob import glob
import cv2
import os
import numpy as np
import re


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


def print_video(infile, outfile):
    # 0. 参数定义
    ratio_heigh, ratio_wide = 5, 5
    erode_edge = 5
    # 1. 打开视频
    cap = cv2.VideoCapture(infile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("fps:", fps)
    # 1.2 打开图片
    pic_image = cv2.imread('logo_b.png')
    print(pic_image.shape)
    print(pic_image)
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
        # 1. 替换
        frame[frame_fromy:, frame_fromx:, :] = logo_image
        # 2. 腐蚀
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


# 批量我文件合并
def concat_video_m3u8(taget_path, purpose_path):
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
        ts_list = glob(os.path.join(user_path, '*.ts'))
        base_full_file = glob(os.path.join(user_path, '*.m3u8'))[0]
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
        print(boxer)
        # tmp = []
        # for ts in boxer:
        #     tmp.append(str(ts[0]))
        cmd_str = '+'.join(boxer)
        exec_str = "copy /b " + cmd_str + ' ' + o_file_name
        print("copy /b " + cmd_str + ' ' + o_file_name)
        os.system(exec_str)

    purpose_path
    user_path = get_user_path(taget_path)  # print(user_path)
    # print(user_path)
    # print(os.getcwd())

    if not user_path:
        print("您输入的路径不正确，:-(")
    else:
        boxer, base_file = get_sorted_ts(user_path)
        all_base_file = "all_{}.ts".format(base_file)
        if os.path.exists(os.path.join(user_path, all_base_file)):
            print('目标文件已存在，跳过。', os.path.join(user_path, all_base_file))
            return 0
        os.chdir(user_path)
        convert_m3u8(boxer, all_base_file)

def get_paths():
    data_path = os.path.join("..", "vvdd")
    file_name = "pcM_586cf28c065b7e9d7142946a5.ts"
    file_full = os.path.join(data_path, file_name)
    print_file = "delphies.jpg"

    return ""

def main():
    source_root =""
    merge_root =""
    mp4_root =""
    res = get_paths()
    # 1. 人工去文件头
    # outfile = 'output.avi'
    # 2. 碎文件合并
    # taget_path = data_path
    # purpose_path = ""
    # concat_video_m3u8(taget_path, purpose_path)
    # 3. 水印视频处理
    # outfile = 'output.avi'
    outfile = 'output.mp4'
    infile = file_full
    print_video(infile, outfile)


if __name__ == "__main__":
    main()
