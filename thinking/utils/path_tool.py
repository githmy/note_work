# -*- coding: utf-8 -*-
import os


def makesurepath(pathn):
    if not os.path.exists(pathn):
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(pathn)
