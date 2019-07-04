# -*- coding: utf-8 -*-

from time import time as nowTime
from functools import wraps


def timeit(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        start = nowTime()
        ret = fn(*args, **kwargs)
        print(nowTime() - start)
        return ret

    return wrap
