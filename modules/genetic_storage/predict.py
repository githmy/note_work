# @Time : 2019/4/22 10:54
# @Author : YingbinQiu
# @Site :
# @File : predict.py

# from pyspark import SparkContext
import os
import time
import pymysql
import pandas as pd
import numpy as np
import pandas as pd
import datetime
import re
from interface_db import InterfaceDB

LOCAL_FLAG = True


def get_info(connection_type=None):
    print("get_info")
    startt = time.time()
    if connection_type is None:
        inpath = os.path.join("..", "data", "tt_orderattribute7.csv")
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
        inpath = os.path.join("..", "data", "tt_orderdetail7.csv")
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
        inpath = os.path.join("..", "data", "tt_orderdetail.csv")
        shelf_info_data = pd.read_csv(inpath, header=0, encoding="utf8", dtype=str, sep=',')
        data = {
            "order_info": order_info_data,
            "order_detail": order_detail_data,
            "commodity_info": commodity_info_data,
            "shelf_info": shelf_info_data,
        }
    else:
        data = {
            "order_info": connection_type.get_order_info(),
            "order_detail": connection_type.get_order_detail(),
            "commodity_info": connection_type.get_commodity_info(),
            "shelf_info": connection_type.get_shelf_info(),
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


def gene_model():
    print("gene")
    real_info = {
        "rack_num": 100,
        "cell_num": 8,
        "cell_volume": 1000,
    }
    config = {
        "gene": {
            "dna_lenth": real_info["rack_num"] * real_info["cell_num"]
        },
        "env": {},
        "loss_func": {},

    }


def deal_order_replenish(indata):
    print("deal_data")
    # composite_1 = [a1*sku_1..an*sku_n]
    # 单客户
    # customer_1 = [composite_1..composite_m] = [b1*sku_1..bn*sku_l]
    # all_data = [customer_1..customer_o] = [composite_1..composite_p] = [b1*sku_1..bn*sku_q]
    # 1. 基本kmeans knn 决策树。每个sku(大类)为一个维度，对订单做聚类。
    # 2. 遗传算法 趋势。神经网络 cell rack 权重。
    # 2.1 起始时间，迭代次数
    # 2.2 续接时间，迭代次数
    # 2.3 是否续接
    gene_model()
    # 3. 规则


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

    # composite_1 = [a1*sku_1..an*sku_n]
    # 单客户
    # customer_1 = [composite_1..composite_m] = [b1*sku_1..bn*sku_l]
    # all_data = [customer_1..customer_o] = [composite_1..composite_p] = [b1*sku_1..bn*sku_q]
    # 1. 基本kmeans knn 决策树。每个sku(大类)为一个维度，对订单做聚类。
    # 2. 遗传算法 趋势。神经网络 cell rack 权重。
    # 2.1 起始时间，迭代次数
    # 2.2 续接时间，迭代次数
    # 2.3 是否续接
    gene_model()
    # 3. 规则


def plot_data():
    print("plot_data")
    # 1. 总时间段不同类的统计
    # 2. 每年的递变,同期递变
    # 3. 总时间递变
    # 4. 属性的数量，一级二级分类


def strategy(conf_path):
    if LOCAL_FLAG is True:
        instance_data = None
    else:
        instance_data = InterfaceDB()
    # 1. 获取数据
    indata = get_info(instance_data)
    # 2. 处理数据
    outdata = deal_order_replenish(indata)
    # 3. 导出数据
    put_info(instance_data, outdata)
    plot_data()


def data_analysis(conf_path):
    if LOCAL_FLAG is True:
        instance_data = None
    else:
        instance_data = InterfaceDB()
    # 1. 获取数据
    indata = get_info(instance_data)
    # 2. 处理数据
    outdata = deal_classify(indata)
    # 3. 导出数据
    put_info(instance_data, outdata)
    plot_data()


if __name__ == '__main__':
    # todo: 待整理
    conf_path = os.path.join("conf")
    strategy(conf_path)
    # data_analysis(conf_path)
    print("end")
