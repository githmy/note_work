# coding:utf-8
import os
import time
import copy
import json
import math
import shutil
import re
import codecs
import math
import numpy as np
import pandas as pd
import platform
import datetime
import logging
import logging.handlers
import matplotlib.pyplot as plt
from txt.basic_mlp import plot_curve, plot_line_scatter_demo


def implant_data(filename):
    data = codecs.open(filename=filename, mode="r")
    data1 = data.readlines()
    del data1[0]
    pddata = [i1.split()[:2] for i1 in data1]
    pddata = [[float(i1[0]), float(i1[1])] for i1 in pddata]
    # pddata = [[i1[0]/40,i1[1]] for i1 in pddata]
    pddata = [i1 for i1 in pddata if i1[0] < 0.02]
    pddata = pd.DataFrame(pddata, dtype=np.float)
    aa = np.power(10, pddata[1])
    # aa = pddata[1]
    # plot_curve(pddata[0], [aa], [""])
    plot_line_scatter_demo(pddata[0], aa)


def fit_vars():
    # doping = range(2, 21, 2)
    # fdoping = np.array([1.25, 3, 6, 9, 11.5, 17.5])
    conc300 = np.array([14, 18, 18.5, 18.9, 19.6, 19.6])
    conc300 = np.power(10, conc300, dtype=np.float64) * 1e6
    conc600 = np.array([18.3, 18.6, 19.1, 19.3, 19.4, 19.6])
    conc600 = np.power(10, conc600, dtype=np.float64) * 1e6
    resis300 = np.array([2, 1.1, -2, -2, -2.2, -2.6])
    resis300 = np.power(10, resis300, dtype=np.float64) * 1e-2
    resis600 = np.array([1, 0.5, -1.4, -1.8, -2.2, -2.6])
    resis600 = np.power(10, resis600, dtype=np.float64) * 1e-2
    # resis_all = np.hstack((resis300, resis600))
    # conc_all = np.hstack((conc300, conc600))
    resis_all = resis300
    conc_all = conc300
    mobility = 1 / (resis_all * conc_all * 1.6 * np.power(10, -19, dtype=np.float64))
    mobility = np.expand_dims(mobility, -1)
    datalen = len(mobility)
    Xc_T = np.vstack(([1] * datalen, conc_all, conc_all ** 2))
    Xc = np.transpose(Xc_T)
    Cabc = np.dot(np.linalg.inv(np.dot(Xc_T, Xc)), np.dot(Xc_T, mobility))
    print(Xc)
    print(mobility)
    print("Cabc:", Cabc)
    print(Xc * np.transpose(Cabc))
    print(np.dot(Xc, Cabc))
    # plot_line_scatter_demo(fdoping, conc300)
    # plot_line_scatter_demo(fdoping, conc600)
    # plot_line_scatter_demo(fdoping, resis300)
    # plot_line_scatter_demo(fdoping, resis600)


def main():
    pass


if __name__ == '__main__':
    fit_vars()
    exit()
    filename = os.path.join("c:/", "project", "simuproject", "lee", "IGZO_TFT", "fluorine.txt")
    implant_data(filename)
    main()
    exit()
