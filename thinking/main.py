# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import simplejson
from server import Delphis
from models.model_cnn import TextCNN
from utils.data_trans import data2js
from utils.log_tool import logger
import os


def datatrans():
    # csv的数据转化为js文件
    bpath = os.path.join("..", "data")
    inpath = os.path.join(bpath, "thinking2", "res_obj.csv")
    outpath = os.path.join(bpath, "thinking2", "question_tmp.js")
    data2js(inpath, outpath)


def get_conf(conf_path):
    server_json = simplejson.load(open(os.path.join(conf_path, "server.json"), encoding="utf8"))
    model_json = simplejson.load(open(os.path.join(conf_path, "model.json"), encoding="utf8"))
    return server_json, model_json


def main():
    # 1. 获取参数
    os.environ['prtest'] = ""
    conf_path = os.path.join(os.getcwd(), "config")
    server_json, model_json = get_conf(conf_path)
    # 2. 起服务
    delphis_ins = Delphis(server_json, model_json)
    logger.info('Started http server on port %s' % server_json["port"])
    delphis_ins.app.run('0.0.0.0', server_json["port"])
    # # 3. 调模型
    model = TextCNN(server_json["model_type"], server_json["model_name"], model_json)
    # x_test, x_train, x_dev, y_train_m, y_dev_m, y_train_r, y_dev_r, y_train_l, y_dev_l = model.data4train()
    # model.build()
    # model.fit(x_train, x_dev, y_train_m, y_dev_m, y_train_r, y_dev_r, y_train_l, y_dev_l)
    # # model.load_mode(model_json["model_name"])
    # model.load_mode("")
    # reslist = model.predict(x_test)
    # # print(reslist)


if __name__ == '__main__':
    logger.info("".center(100, "*"))
    logger.info("welcome to Delphis".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    logger.info("")
    # datatrans()
    main()
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
