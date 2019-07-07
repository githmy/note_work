# -*- coding: utf-8 -*-
import os
import sys

print('\nPython 路径为：', sys.path)

print("os.getcwd()=%s" % os.getcwd())

print("sys.path[0]=%s" % sys.path[0])

print("sys.argv[0]=%s" % sys.argv[0])

# 递归f_dir 下的文件目录 文件 , dirs 和 files 都为[]
for root, dirs, files in os.walk(f_dir, topdown=True):
    pass

# 目录
os.path.exists("modelName")

# 如果不存在则创建目录
if not os.path.exists(path):
    os.makedirs(path)

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

# 某文件的基本路径
os.path.basename("full_path")

# 切换工作路径
os.chdir("user_path")

# 程序当前路径
os.getcwd()

# 绝对路径
os.path.abspath(__file__)

# 相对路径
os.path.realpath(__file__)
