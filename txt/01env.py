#!/usr/bin/env python3
import os

# 设置环境变量
os.environ['WORKON_HOME'] = "value"
# 获取环境变量方法1
os.environ.get('WORKON_HOME')
# 获取环境变量方法2(推荐使用这个方法)
os.getenv('path')
# 删除环境变量
del os.environ['WORKON_HOME']

# 其他key值：
os.environ['HOMEPATH']  # 当前用户主目录。
os.environ['TEMP']  # 临时目录路径。
os.environ['PATHEXT']  # 可执行文件。
os.environ['SYSTEMROOT']  # 系统主目录。
os.environ['LOGONSERVER']  # 机器名。
os.environ['PROMPT']  # 设置提示符。
