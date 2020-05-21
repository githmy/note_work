#! pip install opencv-python
# ! pip install MoviePy==1.0.0
import os
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


# 处理视频+音频的主函数
def moviepy_trans(infile, outfile, div):
    # 1. 打开视频
    ori_video = VideoFileClip(infile)
    moviesize = (ori_video.w, ori_video.h)
    newmoviesize = (ori_video.w // div, ori_video.h // div)
    ori_video = ori_video.resize(newmoviesize)
    print("from size ", moviesize, " to ", newmoviesize)
    print(ori_video.duration)
    # 4. 输出
    result = concatenate_videoclips([ori_video])
    result.write_videofile(outfile, fps=ori_video.fps, audio=True)


# 处理视频+音频的主函数
def moviepy_dehead(infile, outfile, start_t=0.0, end_t=0.0):
    # 1. 读入
    ori_video = VideoFileClip(infile)
    print(ori_video.duration)
    # 3. 输出
    ori_video_m = ori_video.subclip(start_t, end_t)
    result = concatenate_videoclips([ori_video_m])
    result.write_videofile(outfile, fps=ori_video.fps, audio=True)


# 处理视频+音频的主函数
def moviepy_demid(infile, outfile, start_t=0.0, end_t=0.0):
    # 1. 读入
    ori_video = VideoFileClip(infile)
    print(ori_video.duration)
    # 2. 剪切
    ori_video_h = ori_video.subclip(0, start_t)
    ori_video_m = ori_video.subclip(start_t, end_t)
    ori_video_t = ori_video.subclip(end_t, None)
    # 3. 输出
    result = concatenate_videoclips([ori_video_h, ori_video_t])
    result.write_videofile(outfile, fps=ori_video.fps, audio=True)


def main():
    # 视频转化单版测试
    # filehead = "聊斋古卷"
    filehead = "黑暗正义联盟"
    # # start_t = 3.01
    # # end_t = 6.1
    # start_t = 6
    # end_t = 6
    start_t = 0
    end_t = 6996
    source_path = os.path.join("F:\\", "download")
    # source_path = os.path.join("F:\\", "download", "鬼吹灯之龙岭神宫")
    source_root = os.path.join(source_path, filehead)
    target_path = source_path
    target_root = os.path.join(target_path, filehead)
    # moviepy_demid(source_root + ".mp4", target_root + "_min.mp4", start_t, end_t)
    # moviepy_dehead(source_root + ".mp4", target_root + "_min.mp4", start_t, end_t)
    moviepy_trans(source_root + ".mp4", target_root + "_new.mp4", 2)


if __name__ == "__main__":
    main()
