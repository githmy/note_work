# -*- coding: utf-8 -*-
import os
import sys

print('\nPython 路径为：', sys.path)

print("os.getcwd()=%s" % os.getcwd())

print("sys.path[0]=%s" % sys.path[0])

print("sys.argv[0]=%s" % sys.argv[0])

# 目录
os.path.exists("modelName")

# 判断目录
if os.path.isfile(modelpath):
    pass
# 判断目录
if os.path.isdir(modelpath):
    # 列出目录内容
    projects = os.listdir(_modelpath)

# 文件存在
if os._exists("reslogfile"):
    # 文件删除
    os.remove("reslogfile")

# 拼接
os.path.join("rootDir", 'data', 'embeddings', "embeddingSource")
# 分割
embeddings_format = os.path.splitext("")[1][1:]
