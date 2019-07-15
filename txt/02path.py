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
file_path = "D:/test/test.py"
(filepath, tempfilename) = os.path.split(file_path)
(filename, extension) = os.path.splitext(tempfilename)
# filepath为文件的目录,即D:/test
# filename为文件的名字,即test
# extension为文件的扩展名,即.py

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

# 文件操作
import shutil, os

# 复制单个文件
shutil.copy("C:\\a\\1.txt", "C:\\b")
# 复制并重命名新文件
shutil.copy("C:\\a\\2.txt", "C:\\b\\121.txt")
# 复制整个目录(备份)
shutil.copytree("C:\\a", "C:\\b\\new_a")

# 删除文件
os.unlink("C:\\b\\1.txt")
os.unlink("C:\\b\\121.txt")
# 删除空文件夹
try:
    os.rmdir("C:\\b\\new_a")
except Exception as ex:
    print("错误信息：" + str(ex))  # 提示：错误信息，目录不是空的
# 删除文件夹及内容
shutil.rmtree("C:\\b\\new_a")

# 移动文件
shutil.move("C:\\a\\1.txt", "C:\\b")
# 移动文件夹
shutil.move("C:\\a\\c", "C:\\b")

# 重命名文件
shutil.move("C:\\a\\2.txt", "C:\\a\\new2.txt")
# 重命名文件夹
shutil.move("C:\\a\\d", "C:\\a\\new_d")
