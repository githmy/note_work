# -*- coding: utf-8 -*-
from django.core.cache import cache


# 1. *** 通过缓存查看 ***
def cacheRead(keyname, checkFunc, getFunc):
    # 0. *** 1.缓存键名。  2.产看相似情况(返回假 或数据)。  3.从数据库设置到缓存。  ***
    # 1. get键是否存在，是否为空
    cachobj = cache.get(keyname)
    # 2. 空则读取， 非空跳过
    if cachobj is None:
        cachobj = getFunc()
    # 3. 对比redis 老数据 没变动返回false 返回原始值，有变动直接返回新值同时写入变动
    checkRes = checkFunc(cachobj)
    # 4. 老数据有变动，回写到数据库
    if checkRes is False:
        return cachobj
    else:
        cache.set(keyname, checkRes, None)
        return checkRes


# 2. *** 通过缓存插入 ***
def cacheAlter(keyname, setFunc, getFunc):
    # 0. *** 1.缓存键名。 2.直接设置到数据库。 3.从数据库设置到缓存。  ***
    # 1. 直接写入数据库
    setFunc()
    # 2. 回读到redis
    cachobj = getFunc()
    cache.set(keyname, cachobj, None)
    return cachobj
