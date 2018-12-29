# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import os
import numpy as np
import tushare as ts
import matplotlib as mpl
import matplotlib.pyplot as plt

# from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num, datestr2num
from datetime import datetime

# ~)00. 不同学习的检验图
# ~)01. 分类正确性的ROC图
