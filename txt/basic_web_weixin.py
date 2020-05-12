# import antigravity
import this
# exit()
import os
import re
import shutil
import time
import datetime
import itchat
from itchat.content import *

print('\n'.join([''.join([('Love'[(x - y) % len('Love')] if ((x * 0.05) ** 2 + (y * 0.1) ** 2 - 1) ** 3 - (
                                                                                                              x * 0.05) ** 2 * (
                                                                                                                                   y * 0.1) ** 3 <= 0 else ' ')
                          for x in range(-30, 30)]) for y in range(30, -30, -1)]))


def weixin定时发送():
    # 登录微信
    itchat.auto_login(hotReload=False)
    # 获取朋有列表
    friends_list = itchat.get_friends(update=True)
    name = itchat.search_friends(name=u'阿樱')
    Aying = name[0]["UserName"]
    # 获取时间
    while True:
        now = datetime.datetime.now()
        if now.hour == 6 and now.minute == 00:
            itchat.send('早安', Aying)
            itchat.send(yulu.qinghua[random.randint(0, 50)], Aying)
        elif now.hour == 22 and now.minute == 00:
            itchat.send('晚安', Aying)
        time.sleep(30)




if __name__ == '__main__':
    "python3 -m http.server 8080"
    pass
