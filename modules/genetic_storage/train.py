# @Time : 2019/4/22 10:54
# @Author : YingbinQiu
# @Site :
# @File : main.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import time

import numpy as np
import pandas as pd
import simplejson

# from interface_db import InterfaceDB
from simu_env import SimuEnv
from simu_strategy import SimuStrategy
# from gene import engine

import argparse
import logging
from functools import wraps

import simplejson
import six
from builtins import str
import argparse
import logging

conf_path = os.path.join(os.getcwd(), "conf")

logger = logging.getLogger(__name__)


def get_info(model_json=None):
    print("get_info")
    if model_json is None:
        raise Exception("error, no model_json.")
    # instance_data = InterfaceDB(model_json)
    startt = time.time()
    if model_json["simu_flag"] == 1:
        inpath = os.path.join("..", "data", "tt_orderattribute8tt.csv")
        typedict = {
            'OrderID': np.str,
            'SubmitDate': np.str,
            'TotalAmount': np.int,
            'SaleDate': np.str,
            'OrderAmount': np.float64,
            'ItemsValue': np.float64,
            'SellingPlatformID': np.str,
            'CreateDate': np.str,
            'ID': np.str,
            'Province': np.str,
            'City': np.str,
            'Area': np.str,
            'isUrgency': np.bool,
            'ActionID': np.int,
            'DeliveryPlanTime': np.str,
        }
        order_info_data = pd.read_csv(inpath, header=0, index_col="SubmitDate", encoding="utf8", dtype=typedict,
                                      sep=',', low_memory=True)
        inpath = os.path.join("..", "data", "tt_orderdetail7tt.csv")
        typedict = {
            'OrderDetailID': np.str,
            'OrderID': np.str,
            'CommodityID': np.int,
            'Amount': np.int,
            'TotalPrice': np.float64,
            'total': np.float64,
        }
        order_detail_data = pd.read_csv(inpath, header=0, encoding="utf8", dtype=typedict, sep=',', low_memory=True)
        inpath = os.path.join("..", "data", "tm_master_commodity7.csv")
        typedict = {
            'CommodityID': np.int,
            'MemberID': np.int,
            'CatalogID': np.int,
            'Weight': np.float64,
            'Price': np.float64,
            'CreateDate': np.str,
            'UpdateDate': np.str,
            'Long': np.float64,
            'Wide': np.float64,
            'Height': np.float64,
            'Volume': np.float64,
        }
        commodity_info_data = pd.read_csv(inpath, header=0, index_col="CommodityID", encoding="utf8", dtype=typedict,
                                          sep=',', low_memory=True)
        inpath = os.path.join("..", "data", "shelf_info.csv")
        typedict = {
            'shelf_id': np.int,
            'cell_id': np.int,
            'commodity_id': np.int,
            'num': np.int,
            'occupynum': np.int,
            'batchcode': np.str,
        }
        shelf_info_data = pd.read_csv(inpath, header=0, encoding="utf8", dtype=typedict, sep=',',
                                      low_memory=True)
        # inpath = os.path.join("..", "data", "shelf_class.csv")
        # shelf_class_data = pd.read_csv(inpath, header=0, encoding="utf8", dtype=str, sep=',')
        # inpath = os.path.join("..", "data", "commodity_cell_info.csv")
        # commodity_cell_info = pd.read_csv(inpath, header=0, encoding="utf8", dtype=str, sep=',')
        data = {
            "order_info": order_info_data,
            "order_detail": order_detail_data,
            "commodity_info": commodity_info_data,
            "shelf_info": shelf_info_data,
            # "shelf_class": shelf_class_data,
            # "commodity_cell_info": commodity_cell_info,
        }
    else:
        data = {
            "order_info": instance_data.get_order_info(),
            "order_detail": instance_data.get_order_detail(),
            "commodity_info": instance_data.get_commodity_info(),
            "shelf_info": instance_data.get_shelf_info(),
            # "shelf_class": instance_data.shelf_class_data(),
            # "commodity_cell_info": instance_data.commodity_cell_info(),
        }
    print("load data time = %s s" % (time.time() - startt))
    return data


def put_info(outdata, connection_type=None):
    print("put_info")
    data = outdata
    startt = time.time()
    if connection_type is None:
        outpath = os.path.join("..", "data", "replenish_list.csv")
        outdata["replenish_list"].to_csv(outpath, header=None, encoding="utf8", dtype=str, sep=',')
        outpath = os.path.join("..", "data", "batch_list.csv")
        outdata["batch_list"].to_csv(outpath, header=None, encoding="utf8", dtype=str, sep=',')
    else:
        connection_type.put_batch(outdata["batch_list"]),
        connection_type.put_replenish(outdata["replenish_list"]),
    print("load data time = %s s" % (time.time() - startt))
    return data


def deal_classify(indata):
    """分类统计"""
    print("deal_classify")
    indata2 = indata
    # 按时间戳计数
    # 按天计数
    # 按周计数
    # 按月计数
    # 按年计数
    # 聚合周几数量
    # 聚合几月数量
    # 聚合几年数量
    pass


def plot_data():
    print("plot_data")
    # 1. 总时间段不同类的统计
    # 2. 每年的递变,同期递变
    # 3. 总时间递变
    # 4. 属性的数量，一级二级分类


def deal_order_replenish(indata, simuenv, simustrategy):
    # # 1. 遗传算法框架
    # engine.run(ng=100)
    # 2. 循环调用当前处理函数
    outdata = simuenv.strategy_flow(indata, simustrategy, threadid=1)
    # 3. 导出数据
    # put_info(instance_data, outdata)
    # plot_data()
    return outdata


def data_analysis(model_json):
    indata = get_info(model_json)
    # 1. 获取数据
    # 2. 处理数据
    outdata = deal_classify(indata)
    # 3. 导出数据
    # put_info(instance_data, outdata)
    plot_data()


def decode_parameters(request):
    """Make sure all the parameters have the same encoding.
    Ensures  py2 / py3 compatibility."""
    return {
        key.decode('utf-8', 'strict'): value[0].decode('utf-8', 'strict')
        for key, value in request.args.items()}


def get_conf(conf_path):
    env_json = simplejson.load(open(os.path.join(conf_path, "env.json"), encoding="utf8"))
    para_json = simplejson.load(open(os.path.join(conf_path, "gene.json"), encoding="utf8"))
    model_json = simplejson.load(open(os.path.join(conf_path, "model.json"), encoding="utf8"))
    return env_json, para_json, model_json


if __name__ == '__main__':
    # os.environ['prtest'] = ""
    # 1. 获取参数
    env_json, para_json, model_json = get_conf(conf_path)
    # 2. 环境初始化
    simuenv = SimuEnv(env_json, model_json)
    # 3. 策略初始化
    simustrategy = SimuStrategy(para_json, model_json)
    # 4. 获取数据
    indata = get_info(model_json)
    # # 5. 训练
    # deal_order_replenish(indata, simuenv, simustrategy)
    # data_analysis(model_json)
    # 6. 常态运行
    simustrategy.pick_replenish_strategy()
    print("end")


# 接受：
# 订单列表，订单详情列表，货架状态列表，订单完成消息，上架完成消息，货架排队状态，分拣台数量状态，
# 发送：
# 订单波次信息，补货建议状态，分拣台数量变动建议

# 初始化时，读入订单缓存池状态为1的订单，再根据容量读入状态为0的订单。根据待执行波次表和正在执行波次表，读入货架货物信息表，刷新可用虚拟货位。
