# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import os
import numpy as np
import logging
import tushare as ts
import datetime
import matplotlib.pyplot as plt

cmd_path = os.getcwd()

input_path = os.path.join(cmd_path, "..", "..", "..", "nocode", "customer", "input")
data_path = os.path.join(cmd_path, "..", "..", "..", "nocode", "customer")
# input_path = os.path.join(cmd_path, "..", "..", "..", "nocode", "customer", "input")
# data_path = os.path.join(cmd_path, "..", "..", "..", "nocode", "customer")
datalogfile = os.path.join(data_path, 'logs', 'finance_analysis.log')

# 创建一个logger
logger1 = logging.getLogger('logger_out')
logger1.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(datalogfile)
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()

# 定义handler的输出格式formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# logger1.addFilter(filter)
logger1.addHandler(fh)
logger1.addHandler(ch)


class TSstockdata:
    def __init__(self):
        cmd_path = os.getcwd()
        self.data_path = os.path.join(cmd_path, "..", "nocode", "customer")
        data_pa = os.path.join(self.data_path, "input", "data")
        self.data_path_stock = os.path.join(data_pa, "stock")
        self.file_stock_info = os.path.join(self.data_path_stock, "stock_info.csv")
        self.data_path_recover = os.path.join(data_pa, "recover")
        self.data_path_res = os.path.join(data_pa, "res")
        # self.file_tmp_feature = os.path.join(self.data_path_res, "profit_date.csv")
        self.file_liquids_order = os.path.join(self.data_path_res, "liquids_order.csv")
        self.file_liquids_mount = os.path.join(self.data_path_res, "liquids_mount.csv")
        self.file_profit_date = os.path.join(self.data_path_res, "profit_date.csv")

    # 常规数据
    def all_store(self, startdate):
        # 1.股票基本信息
        # 1.1 更新信息
        filePath = 'stock_info.csv'
        tmp_path = os.path.join(self.data_path_stock, filePath)
        if os.path.isfile(tmp_path):
            df1 = pd.read_csv(tmp_path, header=0, encoding="gbk", dtype=str)
        else:
            df1 = None
        df2 = ts.get_stock_basics()  # 获取5分钟k线数据
        # df2[['two', 'three']] = df2[['two', 'three']].astype(float)
        df2.reset_index(level=0, inplace=True)  # （the first）index 改为 column
        df1 = pd.concat([df1, df2], join='outer', axis=0)
        # df1.reset_index()  # （all）index 改为 column
        df1.drop_duplicates(['code'], inplace=True)
        df1.set_index('code', inplace=True)
        df1.sort_index(axis=0, ascending=True, inplace=True)
        # df.drop(df.index[[0]], axis=0, inplace=True)
        # df.sort_values(by=['date'])
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        df1.to_csv(tmp_path)

        # 1.2 去除多余列
        # df = ts.get_stock_basics()
        # df.to_csv('stock_info.csv')
        df1.reset_index(level=0, inplace=True)  # （the first）index 改为 column
        # print(df.head())
        droplist = ["name", "industry", "area", "pe", "outstanding", "totals", "totalAssets", "liquidAssets",
                    "fixedAssets",
                    "reserved", "reservedPerShare", "esp", "bvps", "pb", "timeToMarket", "undp", "perundp", "rev",
                    "profit",
                    "gpr", "npr", "holders"]
        df1.drop(droplist, axis=1, inplace=True)
        # df1.drop([0], axis=0, inplace=True)
        # 2.循环每一只股票
        otherlist = ['sh', 'sz', 'hs300', 'sz50', 'zxb', 'cyb']
        for i in otherlist:
            logger1.info(i)
            self.single_store(i, startdate)
        for i in df1["code"]:
            logger1.info(i)
            self.single_store(str(i).rjust(6, "0"), startdate)

    # 复权数据，不完善
    def store_recover(self):
        tfunc = ts.get_hist_data
        df2 = ts.get_stock_basics()  # 获取5分钟k线数据
        df2.reset_index(level=0, inplace=True)  # （the first）index 改为 column
        for i in df2["code"]:
            filePath = i + '.csv'
            tmp_path = os.path.join(self.data_path_recover, filePath)
            try:
                logger1.info(i)
                df1 = ts.get_h_data(i)
                df1.sort_index(axis=0, ascending=True, inplace=True)
                df1.to_csv(tmp_path)
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
                df1.to_csv(tmp_path)
            except Exception as e:
                logger1.info("error with code: %s" % i)
                logger1.info(e)

    def single_store(self, code, startdate):
        ktpye = ["W", "M", "D", "5", "15", "30", "60"]
        for i in ktpye:
            self.single_stock_type(code, i, startdate)

    def single_stock_type(self, code, stype, startdate):
        tfunc = ts.get_hist_data
        self.update_file_by_index(tfunc, code, stype, self.data_path_stock, startdate)

    def update_file_by_index(self, func, code, stype, tpath, startdate):
        # print(func, code, stype, tpath)
        filePath = code + '_' + stype + '.csv'
        tmp_path = os.path.join(tpath, filePath)
        try:
            if os.path.isfile(tmp_path):
                # df1 = pd.read_csv(tmp_path, header=0, index_col='date')
                df1 = pd.read_csv(tmp_path, header=0, encoding="utf8", dtype=str)
                # df1 = pd.DataFrame(df1.iloc[:, 1:6], index=df1.iloc[:, 0])
            else:
                df1 = None
            df2 = func(code, ktype=stype, start=startdate)  # 获取5分钟k线数据
            df2.reset_index(level=0, inplace=True)  # （the first）index 改为 column
            df1 = pd.concat([df1, df2], join='outer', axis=0)
            df1.drop_duplicates(['date'], inplace=True)
            df1.set_index('date', inplace=True)
            df1.sort_index(axis=0, ascending=True, inplace=True)
            # df1.reset_index()  # （all）index 改为 column
            # df.drop(df.index[[0]], axis=0, inplace=True)
            # df.sort_values(by=['date'])
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
            df1.to_csv(tmp_path)
        except Exception as e:
            pastr = "_".join([code, stype])
            logger1.info("error with parameters: %s" % pastr)
            logger1.info(e)

    def panda_get_data(self):
        # spy = wb.Datareader(
        #     "SPY", "yahoo",
        #     datetime.datetime(2007, 1, 1),
        #     datetime.datetime(2015, 6, 15)
        # )
        # print(spy.tail())
        pass

    def for_stock(self):
        # ts.get_hist_data('600848', start='2015-01-05', end='2015-01-09')
        # df = ts.get_hist_data('600848')
        # filePath = '600848_d.csv'
        # if os.path.isfile(filePath):
        #     os.remove(filePath)
        # df.to_csv(filePath)
        # df = ts.get_hist_data('600848', ktype='5')  # 获取5分钟k线数据
        # filePath = '600848_5.csv'
        # if os.path.isfile(filePath):
        #     os.remove(filePath)
        # df.to_csv(filePath)
        filePath = '600848_15.csv'
        if os.path.isfile(filePath):
            # df1 = pd.read_csv(filePath, header=0, index_col='date')
            df1 = pd.read_csv(filePath, header=0, encoding="gb2312")
            df1.set_index('date', inplace=True)
            # df1 = pd.DataFrame(df1.iloc[:, 1:6], index=df1.iloc[:, 0])
        else:
            df1 = None
        df2 = ts.get_hist_data('600848', ktype='15')  # 获取5分钟k线数据
        df1 = pd.concat([df1, df2], join='inner', axis=0)
        df1.sort_index(axis=0, ascending=True, inplace=True)
        df1.reset_index(level=0, inplace=True)  # （the first）index 改为 column
        # df1.reset_index()  # （all）index 改为 column
        df1.drop_duplicates(['date'], inplace=True)
        df1.set_index('date', inplace=True)
        # df.drop(df.index[[0]], axis=0, inplace=True)
        # df.sort_values(by=['date'])
        if os.path.isfile(filePath):
            os.remove(filePath)
        df1.to_csv(filePath)

        # # axis=0 是行拼接，拼接之后行数增加，列数也根据join来定，join=’outer’时，列数是两表并集。同理join=’inner’,列数是两表交集。
        # concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None,
        #        verigy_integrity=False)
        # print(df)
        # filePath = '600848_15.csv'
        # # if os.path.isfile(filePath):
        # #     os.remove(filePath)
        # df.to_csv(filePath, mode='a')


class Stockdata:
    def __init__(self):
        cmd_path = os.getcwd()
        data_pa = os.path.join(cmd_path, "..", "..", "..", "nocode", "customer", "input", "data")
        self.data_path_stock = os.path.join(data_pa, "stock")
        self.file_stock_info = os.path.join(self.data_path_stock, "stock_info.csv")
        self.data_path_recover = os.path.join(data_pa, "recover")
        self.data_path_res = os.path.join(data_pa, "res")
        self.file_liquids_order = os.path.join(self.data_path_res, "liquids_order.csv")
        self.file_liquids_mount = os.path.join(self.data_path_res, "liquids_mount.csv")
        self.file_profit_date = os.path.join(self.data_path_res, "profit_date.csv")

    def data_stock_info(self):
        typedict = {
            'code': np.str,
            'name': np.str,
            'industry': np.str,
            'area': np.str,
            'pe': np.float64,
            'outstanding': np.float64,
            'totals': np.float64,
            'totalAssets': np.float64,
            'liquidAssets': np.float64,
            'fixedAssets': np.float64,
            'reserved': np.float64,
            'reservedPerShare': np.float64,
            'esp': np.float64,
            'bvps': np.float64,
            'pb': np.float64,
            'timeToMarket': np.float64,
            'undp': np.float64,
            'perundp': np.float64,
            'rev': np.float64,
            'profit': np.float64,
            'gpr': np.float64,
            'npr': np.float64,
            'holders': np.float64
        }
        # df = pd.read_csv(self.file_stock_info, header=0, encoding="utf8", dtype=typedict)
        df = pd.read_csv(self.file_stock_info, header=0, encoding="gbk", dtype=typedict)
        return df

    def data_stocklist(self):
        #df = pd.read_csv(self.file_stock_info, header=0, encoding="utf8", dtype=str)
        df = pd.read_csv(self.file_stock_info, header=0, encoding="gbk", dtype=str)
        droplist = ["name", "industry", "area", "pe", "outstanding", "totals", "totalAssets", "liquidAssets",
                    "fixedAssets", "reserved", "reservedPerShare", "esp", "bvps", "pb", "timeToMarket", "undp",
                    "perundp", "rev", "profit", "gpr", "npr", "holders"]
        df.drop(droplist, axis=1, inplace=True)
        nparray = np.array(df)
        # otherlist = ['sh', 'sz', 'hs300', 'sz50', 'zxb', 'cyb']
        otherlist = []
        # 添加行
        nparray = np.row_stack((nparray, np.transpose([otherlist])))
        return nparray

    def data_stocklist_value(self, stpye, datalist):
        res = {}
        typedict = {
            'open': np.float64,
            'high': np.float64,
            'close': np.float64,
            'low': np.float64,
            'volume': np.float64,
            'price_change': np.float64,
            'p_change': np.float64,
            'ma5': np.float64,
            'ma10': np.float64,
            'ma20': np.float64,
            'v_ma5': np.float64,
            'v_ma10': np.float64,
            'v_ma20': np.float64,
            'turnover': np.float64
        }
        for i1 in datalist:
            tmpfile = os.path.join(self.data_path_stock, i1[0] + "_" + stpye + ".csv")
            try:
                res.__setitem__(i1[0], pd.read_csv(tmpfile, header=0, encoding="utf8", parse_dates=[0], index_col=0,
                                                   dtype=typedict))
            except Exception as e:
                logger1.info("error when read %s" % tmpfile)
        return res

    def data_middle(self):
        # 读取中间文件
        typp = {0: np.str}
        df1 = pd.read_csv(self.file_liquids_order, header=0, encoding="utf8", dtype=typp, index_col="code")
        df2 = pd.read_csv(self.file_liquids_mount, header=0, encoding="utf8", dtype=typp, index_col="code")
        df3 = pd.read_csv(self.file_liquids_mount, header=0, encoding="utf8", dtype=typp, index_col="code")
        pdobj = {
            "liquids_order": df1,
            "liquids_mount": df2,
            "profile_date": df3,
        }
        return pdobj

    def data_feature(self, name):
        typp = {0: np.str}
        stpye = "D"
        tmpfile = os.path.join(self.data_path_res, name + "_" + stpye + "feature.csv")
        df1 = pd.read_csv(tmpfile, header=0, encoding="utf8", dtype=typp, index_col="code")
        # 读取中间文件
        pdobj = {
            name + "_" + stpye + "feature": df1,
        }
        return pdobj

    def generate_middles(self):
        # 0.产生中间文件
        print("generating middle files...")
        col = "liquidAssets"
        stock_info = self.data_stock_info()
        stock_info.set_index("code", inplace=True)
        stocklist = self.data_stocklist()
        data_list = self.data_stocklist_value("D", stocklist)
        # 1. 当日流动量生成
        for i1 in data_list:
            data_list[i1][col] = stock_info.loc[i1, col] / data_list[i1]["close"][
                data_list[i1].shape[0] - 1] * data_list[i1]["close"]
        # 1.2 流动量提取
        tmp_obj = [data_list[i].rename(columns={col: i})[i] for i in data_list]
        liquids_pd = pd.concat(tmp_obj, axis=1)
        # 1.3 流动量转置
        liquids_pd = liquids_pd.T
        # 1.4 流动量每日情况
        if os.path.isfile(self.file_liquids_mount):
            os.remove(self.file_liquids_mount)
        liquids_pd.to_csv(self.file_liquids_mount, index=True, index_label="code")
        print("file: %s" % self.file_liquids_mount)
        # 2. 流动量排序号
        orderl_pd = pd.DataFrame(data={})
        for i in liquids_pd:
            orderl_pd[i] = liquids_pd[i].rank(ascending=1, method='first')
            orderl_pd[[i]] = orderl_pd[[i]].fillna(1e6).astype(int)
        # 2.1 流动量每日排序
        if os.path.isfile(self.file_liquids_order):
            os.remove(self.file_liquids_order)
        orderl_pd.to_csv(self.file_liquids_order, index=True, index_label="code")
        print("file: %s" % self.file_liquids_order)
        # 3. 每日利润
        col = "close"
        data_list = self.data_stocklist_value("D", stocklist)
        tmp_obj = []
        for i2 in data_list:
            data_list[i2] = data_list[i2].rename(columns={col: i2})
            # tmp_np = np.append(np.diff(data_list[i2][i2], n=1), np.nan)
            tmp_np = np.insert(np.diff(data_list[i2][i2], n=1), 0, values=[np.nan], axis=0)
            tmp_obj.append(np.true_divide(tmp_np, data_list[i2][i2]))
        profile_pd = pd.concat(tmp_obj, axis=1)
        # for i in profile_pd:
        #     orderl_pd[i] = profile_pd[i].rank(ascending=1, method='first')
        #     orderl_pd[[i]] = profile_pd[[i]].fillna(1e6).astype(int)
        # 3.1 流动量每日排序
        if os.path.isfile(self.file_profit_date):
            os.remove(self.file_profit_date)
        profile_pd.to_csv(self.file_profit_date, index=True, index_label="code")
        print("file: %s" % self.file_profit_date)
        return 0

    def generate_feature(self, scode):
        # 0.产生中间文件
        print("generating single files...")
        # scode = "000001"
        stpye = "D"
        col = "close"
        daays = 200
        typedict = {
            'open': np.float64,
            'high': np.float64,
            'close': np.float64,
            'low': np.float64,
            'volume': np.float64,
            'price_change': np.float64,
            'p_change': np.float64,
            'ma5': np.float64,
            'ma10': np.float64,
            'ma20': np.float64,
            'v_ma5': np.float64,
            'v_ma10': np.float64,
            'v_ma20': np.float64,
            'turnover': np.float64
        }
        tmpfile = os.path.join(self.data_path_stock, scode + "_" + stpye + ".csv")
        single_pd = pd.read_csv(tmpfile, header=0, encoding="utf8", parse_dates=[0], index_col=0, dtype=typedict)
        # 1. 当日流动量生成
        lenth_col = len(single_pd[col])
        for i1 in range(1, daays):
            single_pd[col + str(i1)] = single_pd[col].copy()
            single_pd[col + str(i1)][0:i1] = np.array([np.nan, ] * i1)
            single_pd[col + str(i1)][i1:lenth_col] = single_pd[col][0:lenth_col - i1]
        # 1.4 流动量每日情况
        single_pd = single_pd.ix[daays - 1:, :]
        tmpfile = os.path.join(self.data_path_res, scode + "_" + stpye + "feature.csv")
        if os.path.isfile(tmpfile):
            os.remove(tmpfile)
        single_pd.to_csv(tmpfile, index=True, index_label="code")
        print("file: %s" % tmpfile)
        return 0

    def data_with_labels(self, scode):
        # 0.产生标签数据
        # scode = "000001"
        stpye = "D"
        shifdays = 1
        typedict = {
            'open': np.float64,
            'high': np.float64,
            'close': np.float64,
            'low': np.float64,
            'volume': np.float64,
            'price_change': np.float64,
            'p_change': np.float64,
            'ma5': np.float64,
            'ma10': np.float64,
            'ma20': np.float64,
            'v_ma5': np.float64,
            'v_ma10': np.float64,
            'v_ma20': np.float64,
            'turnover': np.float64
        }
        tmpfile = os.path.join(self.data_path_stock, scode + "_" + stpye + ".csv")
        try:
            single_pd = pd.read_csv(tmpfile, header=0, encoding="utf8", parse_dates=[0], index_col=0, dtype=typedict)
        except Exception as e:
            return pd.DataFrame(data={})
        # 1. 当日流动量生成
        for i1 in single_pd.columns:
            single_pd["ylabel_" + i1] = single_pd[i1].shift(shifdays)
        single_pd = single_pd[np.isnan(single_pd["ylabel_close"]) == False]
        return single_pd

    def data_without_labels(self, scode):
        # 0.产生标签数据
        # scode = "000001"
        stpye = "D"
        shifdays = 1
        typedict = {
            'open': np.float64,
            'high': np.float64,
            'close': np.float64,
            'low': np.float64,
            'volume': np.float64,
            'price_change': np.float64,
            'p_change': np.float64,
            'ma5': np.float64,
            'ma10': np.float64,
            'ma20': np.float64,
            'v_ma5': np.float64,
            'v_ma10': np.float64,
            'v_ma20': np.float64,
            'turnover': np.float64
        }
        tmpfile = os.path.join(self.data_path_stock, scode + "_" + stpye + ".csv")
        try:
            single_pd = pd.read_csv(tmpfile, header=0, encoding="utf8", parse_dates=[0], index_col=0, dtype=typedict)
        except Exception as e:
            return pd.DataFrame(data={})
        # 1. 新加特征
        single_pd["amplitude"] = single_pd["high"] - single_pd["low"]
        single_pd = single_pd[np.isnan(single_pd["close"]) == False]
        return single_pd

    def data_network_labels(self, scode, b_sequence=200, a_sequence=30, train_split=0.9):
        # 0.产生标签数据
        # scode = "000001"
        b_sequence = 200
        a_sequence = 30
        stpye = "D"
        shifdays = 1
        typedict = {
            'open': np.float64,
            'high': np.float64,
            'close': np.float64,
            'low': np.float64,
            'volume': np.float64,
            'price_change': np.float64,
            'p_change': np.float64,
            'ma5': np.float64,
            'ma10': np.float64,
            'ma20': np.float64,
            'v_ma5': np.float64,
            'v_ma10': np.float64,
            'v_ma20': np.float64,
            'turnover': np.float64
        }
        tmpfile = os.path.join(self.data_path_stock, scode + "_" + stpye + ".csv")
        try:
            single_pd = pd.read_csv(tmpfile, header=0, encoding="utf8", parse_dates=[0], index_col=0, dtype=typedict)
        except Exception as e:
            single_pd = pd.DataFrame(data={})
        # 1. 当日流动量生成
        data = single_pd.as_matrix()
        sequence_length = b_sequence + a_sequence

        result = []
        for index in range(len(data) - b_sequence - a_sequence + 1):
            result.append(data[index: index + sequence_length])  # index : index + 22days
        result = np.array(result)
        row = round(train_split * result.shape[0])  # 90% split
        train = result[:int(row), :]  # 90% date
        X_train = train[:, :-1]  # all data until day m
        y_train = train[:, -1][:, -1]

        X_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:, -1]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

        return single_pd


if __name__ == '__main__':
    # 1. 测试
    dclass = Stockdata()
    dclass.data_stock_info()
    aa = dclass.generate_middles()
    print(aa)
