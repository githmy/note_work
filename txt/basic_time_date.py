# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import time
from datetime import datetime

if __name__ == '__main__':
    # 1. 时间计时秒 时间戳 1547387938
    timsteemp = time.time()
    print(timsteemp)
    # 2. 日期 2019-01-13 22:00:20.486176
    end = datetime.today()
    print(end)
    # 3. 时间 推算
    start = datetime(end.year - 1, end.month, end.day)
    print(start)
    # 3. 时间戳 转化 datetime 和 字符串
    timeArray = time.localtime(int(timsteemp))
    print(timeArray)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    print(otherStyleTime)
    # 4. 字符串 转 datetime 时间戳
    tss1 = '2013-10-10 23:40:00'
    timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    print(timeArray)
    tmstemp = time.mktime(timeArray)
    print(tmstemp)
