# /usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

if __name__ == '__main__':
    print np.sqrt(6 * np.sum(1 / np.arange(1, 1000000, dtype=np.float) ** 2))
