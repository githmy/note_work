# -*- coding: UTF-8 -*-
from modules.voice2txt.neural_network import train, predict
import os
import re
from time import time as nowTime
from modules.voice2txt.xunfei_api import voice_api, picture_print_api
from modules.voice2txt.baidu_api import baidu_voice, baidu_picture


def get_labels():
    wav_path = '/home/mla/data/corpus/data_thchs30/data'
    if os.path.isdir(wav_path):
        # 列出目录内容
        projects = os.listdir(wav_path)
    labellist = []
    for i1 in projects:
        if i1.endswith('.wav.trn'):
            tmphead = re.sub('\.wav\.trn$', '', i1)
            if os.path.isfile(os.path.join(wav_path, tmphead + ".wav")):
                with open(os.path.join(wav_path, i1), "r", encoding="utf-8") as f:
                    content = f.readline()
                    # labellist.append(tmphead + " " + content.rstrip("\n\r"))
                    labellist.append(tmphead + " " + content)
    # print(labellist)
    # print(len(labellist))
    with open(os.path.join(wav_path, "..", "wav_label2.txt"), "w", encoding="utf-8") as f:
        for i1 in labellist:
            f.writelines(i1)


if __name__ == '__main__':
    # 产生音频标签
    # get_labels()

    # start = nowTime()
    # # 音频训练
    # train()
    # predict()
    # print(nowTime() - start)

    # # 讯飞接口
    # voice_api()
    picture_print_api()

    # 百度接口
    baidu_voice()
    # baidu_picture()
    print("end")
