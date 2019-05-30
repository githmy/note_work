# @Time : 2019/4/22 10:54
# @Author : YingbinQiu
# @Site :
# @File : simu_strategy.py

import copy
import itertools
import json
import os
import time
import urllib.request as librequest
import hashlib
import math
import pandas as pd
import simplejson
import logging
import logging.handlers

# pd.set_option('display.max_columns', None)
cmd_path = os.getcwd()
datalogfile = os.path.join(cmd_path, '..', 'data', 'log')
datalogfile = os.path.join(datalogfile, 'ttmp.log')

logger1 = logging.getLogger('log')
logger1.setLevel(logging.DEBUG)

fh = logging.handlers.RotatingFileHandler(datalogfile, maxBytes=104857600, backupCount=10)

# fh = logging.FileHandler(datalogfile)
ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger1.addHandler(fh)
logger1.addHandler(ch)


class SimuStrategy(object):
    def __init__(self, conf, model_json):
        self._conf = conf
        self._model_json = model_json
        # // 单分拣台队列货物挤压数量上阈值(开新线用)
        self._bench_queue_up = self._conf["bench_queue_up"]
        # // 单分拣台队列货物挤压数量下阈值(停线用)
        self._bench_queue_down = self._conf["bench_queue_down"]
        # // 单分拣台队列货物挤压数量上阈值(开新线用)
        self._replenish_queue_up = self._conf["replenish_queue_up"]
        # // 单分拣台队列货物挤压数量下阈值(停线用)
        self._replenish_queue_down = self._conf["replenish_queue_down"]
        # // cell数量百分比阈值(可能会分大小柜)
        self._cell_num_thresh = self._conf["cell_num_thresh"]
        # // 冷单阈值
        self._cold_round = self._conf["cold_round"]
        # // 订单提交过期时间上限(早单晚到)
        self._cold_seconds = self._conf["cold_seconds"]
        # // 订单缓存下限数量(开工用)
        self._order_buff_low = self._conf["order_buff_low"]
        # 新建波次下限
        self._bench_queue_ave = self._conf["bench_queue_ave"]
        # 订单池上限
        self._order_max = self._model_json["order_max"]
        # 分拣台数量
        self._batch_num = self._conf["batch_num"]
        self._para_path = self._model_json["para_path"]
        self._alg_user = self._model_json["alg_user"]
        self._alg_pass = self._model_json["alg_pass"]
        self._wcsurl = self._model_json["wcsurl"]
        self._wcsinit = self._model_json["wcsurl"] + "/" + self._model_json["wcsinit"]
        self._wcsget = self._model_json["wcsurl"] + "/" + self._model_json["wcsget"]
        self._wcsput = self._model_json["wcsurl"] + "/" + self._model_json["wcsput"]
        self._wcsupdate = self._model_json["wcsurl"] + "/" + self._model_json["wcsupdate"]
        self._class_over = self._model_json["class_over"]
        self._bench_maxnum = self._model_json["bench_maxnum"]
        self._replenish_maxnum = self._model_json["replenish_maxnum"]
        self._combine_shelf = self._model_json["combine_shelf"]
        # 读取间隔
        self._read_interval = self._model_json["read_interval"]
        self._error_retry = self._model_json["error_retry"]
        self._error_sleep = self._model_json["error_sleep"]
        self._cosy_sleep = self._model_json["cosy_sleep"]
        self._normal_sleep = self._model_json["normal_sleep"]
        # 数据
        self._replenish_list45 = []
        self._order_info = []
        self._order_detail = []
        self._shelf_info = []
        self._shelf_pos_num = []
        self._shelf_pos1_num = []
        self._bench_queue = {}
        self._replenish_queue = {}
        self._bench_usenum = 1
        self._order_num = len(self._order_info)
        self._batch_maxid = 0
        self._replenish_maxid = 0
        self._tmp_low_efficiency_order = []
        self._tmp_low_efficiency_shelf = []
        self._init_status()

    def _init_status(self):
        # 初始化为1的订单
        self._order_info = []
        self._order_detail = []
        self._bench_queue = {"1": [{"batchID": 3}, {"batchID": 4}], "2": [{"batchID": 5}, {"batchID": 6}]}

        if os.getenv('prtest') is None:
            # "maxOrderNum": -1 给全部的
            jsonstr = json.dumps({"maxOrderNum": -1, "getOrderStatus": 1})
            m = hashlib.md5()
            m.update((jsonstr + self._alg_user + self._alg_pass).encode(encoding="utf-8"))
            tstr = m.hexdigest()
            params = {"Token": tstr, "User": self._alg_user,
                      "Parameter": jsonstr}
            endata = bytes(json.dumps(params), "utf-8")
            request_headers = {"content-type": "application/json"}
            req = librequest.Request(url=self._wcsinit, data=endata, method='POST', headers=request_headers)
            with librequest.urlopen(req) as response:
                ori_page = response.read().decode('utf-8')
                the_page0 = simplejson.loads(ori_page)
                logger1.info("initial back information: %s" % the_page0)
                the_page = the_page0["data"][0]
        else:
            getstr = {
                "success": "",
                "message": "Succ",
                "data":
                    {
                        "algOrderInfos": [
                            {
                                "orderID": 4,
                                "submitDate": "2018-01-01 11:11:11",
                                "saleDate": "2018-01-01 11:11:11",
                                "deliveryPlanTime": "2018-01-01 11:11:11",
                                "orderDetails": [
                                    {"orderID": 4, "commodityID": 2, "productBatchID": 3, "stockType": 1, "amount": 6},
                                    {"orderID": 4, "commodityID": 3, "productBatchID": 3, "stockType": 1, "amount": 6}
                                ]
                            }
                        ],
                        "taskBaseInfo": {"maxReplenishID": 1155, "maxBatchID": 2266},
                        "batchCacheNum": 5,
                        "storageCapacities": [
                            {"commodityID": 1, "shelfID": 1, "shelfSide": 0, "posID": 1, "maxAmount": 5},
                            {"commodityID": 2, "shelfID": 1, "shelfSide": 0, "posID": 2, "maxAmount": 5},
                            {"commodityID": 3, "shelfID": 1, "shelfSide": 1, "posID": 3, "maxAmount": 5},
                            {"commodityID": 4, "shelfID": 1, "shelfSide": 1, "posID": 4, "maxAmount": 5},
                            {"commodityID": 1, "shelfID": 2, "shelfSide": 0, "posID": 5, "maxAmount": 5},
                            {"commodityID": 10, "shelfID": 2, "shelfSide": 0, "posID": 6, "maxAmount": 5},
                            {"commodityID": 10, "shelfID": 2, "shelfSide": 1, "posID": 7, "maxAmount": 5},
                            {"commodityID": 95, "shelfID": 2, "shelfSide": 1, "posID": 8, "maxAmount": 5},
                            {"commodityID": 1, "shelfID": 3, "shelfSide": 0, "posID": 9, "maxAmount": 5},
                            {"commodityID": 20, "shelfID": 3, "shelfSide": 0, "posID": 10, "maxAmount": 5},
                            {"commodityID": 96, "shelfID": 3, "shelfSide": 1, "posID": 11, "maxAmount": 5},
                            {"commodityID": 97, "shelfID": 3, "shelfSide": 1, "posID": 12, "maxAmount": 5},
                            {"commodityID": 1, "shelfID": 4, "shelfSide": 0, "posID": 13, "maxAmount": 5},
                            {"commodityID": 4, "shelfID": 4, "shelfSide": 0, "posID": 14, "maxAmount": 5},
                            {"commodityID": 98, "shelfID": 4, "shelfSide": 1, "posID": 15, "maxAmount": 5},
                            {"commodityID": 99, "shelfID": 4, "shelfSide": 1, "posID": 16, "maxAmount": 5},
                        ]
                    },
                "total": 10
            }
            the_page = getstr["data"]
        for i1 in the_page["algOrderInfos"]:
            i1["cold_counter"] = 0
            self._order_info.append(i1)
        self._order_num = len(self._order_info)
        self._batch_maxid = the_page["taskBaseInfo"]["maxBatchID"]
        self._replenish_maxid = the_page["taskBaseInfo"]["maxReplenishID"]
        self._shelf_pos1_num = the_page["storageCapacities"]

    def dump_paras(self, parafile):
        tmpjson = {
            "bench_queue_up": self._bench_queue_up,
            "bench_queue_down": self._bench_queue_down,
            "replenish_queue_up": self._replenish_queue_up,
            "replenish_queue_down": self._replenish_queue_down,
        }
        simplejson.dumps(tmpjson, open(parafile, mode='w'))

    # 策略常态
    def pick_replenish_strategy(self):
        # 补货优先级，0. 不需要，1. 缓存区有就补，没有就算了，2.正常按需求补，3.低效补，4.缺货补，5.新品
        while True:
            # 1. 数据请求
            time.sleep(self._normal_sleep)
            starttime = time.time()
            getnum = self._order_max - self._order_num
            if self._order_max - self._order_num < 0:
                getnum = 0
            jsonstr = json.dumps({"maxOrderNum": getnum, "getOrderStatus": 0})
            m = hashlib.md5()
            m.update((jsonstr + self._alg_user + self._alg_pass).encode(encoding="utf-8"))
            tstr = m.hexdigest()
            params = {"Token": tstr, "User": self._alg_user, "Parameter": jsonstr}
            # "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Parameter": jsonstr}
            endata = bytes(json.dumps(params), "utf-8")
            request_headers = {"content-type": "application/json"}
            req = librequest.Request(url=self._wcsget, data=endata, method='POST', headers=request_headers)
            try:
                if os.getenv('prtest') is None:
                    with librequest.urlopen(req) as response:
                        ori_page = response.read().decode('utf-8')
                        the_page0 = simplejson.loads(ori_page)
                        the_page = the_page0["data"][0]
                else:
                    gddd = {
                        "success": "", "message": "Succ",
                        "data": [
                            {
                                "algOrderInfos": [
                                    {
                                        "orderID": 1, "submitDate": "2019-01-01 11:11:11",
                                        "saleDate": "2019-01-01 11:11:11",
                                        "deliveryPlanTime": "2019-01-01 11:11:11",
                                        "orderDetails": [
                                            {"orderID": 1, "commodityID": 4, "productBatchID": 3, "stockType": 1,
                                             "amount": 2},
                                            {"orderID": 1, "commodityID": 2, "productBatchID": 3, "stockType": 1,
                                             "amount": 1}
                                        ]
                                    },
                                    {
                                        "orderID": 11, "submitDate": "2019-01-01 11:11:11",
                                        "saleDate": "2019-01-01 11:11:11",
                                        "deliveryPlanTime": "2019-01-01 11:11:11",
                                        "orderDetails": [
                                            {"orderID": 11, "commodityID": 3, "productBatchID": 3, "stockType": 1,
                                             "amount": 1}
                                        ]
                                    },
                                    {
                                        "orderID": 2, "submitDate": "2018-01-01 11:11:11",
                                        "saleDate": "2018-01-01 11:11:11",
                                        "deliveryPlanTime": "2018-01-01 11:11:11",
                                        "orderDetails": [
                                            {"orderID": 2, "commodityID": 1, "productBatchID": 3, "stockType": 1,
                                             "amount": 2},
                                            {"orderID": 2, "commodityID": 4, "productBatchID": 3, "stockType": 1,
                                             "amount": 90}
                                        ]
                                    },
                                    {
                                        "orderID": 3,
                                        "submitDate": "2019-01-01 11:11:11",
                                        "saleDate": "2019-01-01 11:11:11",
                                        "deliveryPlanTime": "2019-01-01 11:11:11",
                                        "orderDetails": [
                                            {"orderID": 3, "commodityID": 2, "productBatchID": 3, "stockType": 1,
                                             "amount": 6},
                                            {"orderID": 3, "commodityID": 3, "productBatchID": 3, "stockType": 1,
                                             "amount": 6}
                                        ]
                                    },
                                    # # 散单
                                    # {
                                    #     "orderID": 7,
                                    #     "submitDate": "2019-01-01 11:11:11",
                                    #     "saleDate": "2019-01-01 11:11:11",
                                    #     "deliveryPlanTime": "2019-01-01 11:11:11",
                                    #     "orderDetails": [
                                    #         {"orderID": 7, "commodityID": 4, "productBatchID": 3, "stockType": 1,
                                    #          "amount": 1},
                                    #         {"orderID": 7, "commodityID": 1, "productBatchID": 3, "stockType": 1,
                                    #          "amount": 10}
                                    #     ]
                                    # },
                                    # # 新品补货
                                    # {
                                    #     "orderID": 6,
                                    #     "submitDate": "2019-01-01 11:11:11",
                                    #     "saleDate": "2019-01-01 11:11:11",
                                    #     "deliveryPlanTime": "2019-01-01 11:11:11",
                                    #     "orderDetails": [
                                    #         {"orderID": 6, "commodityID": 101, "productBatchID": 3, "stockType": 1,
                                    #          "amount": 6},
                                    #         {"orderID": 6, "commodityID": 3, "productBatchID": 3, "stockType": 1,
                                    #          "amount": 6}
                                    #     ]
                                    # },
                                    # # 低效补货
                                    # {
                                    #     "orderID": 5, "submitDate": "2019-01-01 11:11:11",
                                    #     "saleDate": "2019-01-01 11:11:11",
                                    #     "deliveryPlanTime": "2019-01-01 11:11:11",
                                    #     "orderDetails": [
                                    #         {"orderID": 5, "commodityID": 3, "productBatchID": 3, "stockType": 1,
                                    #          "amount": 5}
                                    #     ]
                                    # },
                                ],
                                "canceledOrders": [4],
                                "batchCacheNum": 5,
                                "availableStorageInfos": [
                                    {"commodityID": 1, "shelfID": 1, "shelfSide": 0, "posID": 1, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 2, "shelfID": 1, "shelfSide": 0, "posID": 2, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 3, "shelfID": 1, "shelfSide": 1, "posID": 3, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 3},
                                    {"commodityID": 4, "shelfID": 1, "shelfSide": 1, "posID": 4, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 1, "onAmount": 2},
                                    {"commodityID": 1, "shelfID": 2, "shelfSide": 0, "posID": 5, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 10, "shelfID": 2, "shelfSide": 0, "posID": 6, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 10, "shelfID": 2, "shelfSide": 1, "posID": 7, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 95, "shelfID": 2, "shelfSide": 1, "posID": 8, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 1, "shelfID": 3, "shelfSide": 0, "posID": 9, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 20, "shelfID": 3, "shelfSide": 0, "posID": 10, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 3, "shelfID": 3, "shelfSide": 1, "posID": 11, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 3},
                                    {"commodityID": 97, "shelfID": 3, "shelfSide": 1, "posID": 12, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 1, "shelfID": 4, "shelfSide": 0, "posID": 13, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 4, "shelfID": 4, "shelfSide": 0, "posID": 14, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 98, "shelfID": 4, "shelfSide": 1, "posID": 15, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                    {"commodityID": 99, "shelfID": 4, "shelfSide": 1, "posID": 16, "productBatchID": 5,
                                     "stockType": 6, "availableAmount": 3, "onAmount": 4},
                                ]
                            }
                        ], "total": 10
                    }
                    the_page = gddd["data"][0]
            except Exception as e:
                print("error: when get order_info at %s. %s" % (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), e))
                time.sleep(self._error_sleep)
                continue
            logger1.info("normal information: %s" % the_page)
            res = self.get_data_judge_run(the_page["algOrderInfos"], the_page["canceledOrders"],
                                          the_page["availableStorageInfos"], self._bench_queue,
                                          the_page["batchCacheNum"], benchusenum=self._bench_usenum)
            if 0 == res:
                pass
            else:
                params2 = self._reformat(res, the_page["canceledOrders"])
                null_sig = 0
                for i1 in params2:
                    if 0 != len(params2[i1]):
                        null_sig = 1
                if 0 == null_sig:
                    continue
                jsonstr = json.dumps(params2)
                m = hashlib.md5()
                m.update((jsonstr + self._alg_user + self._alg_pass).encode(encoding="utf-8"))
                tstr = m.hexdigest()
                params2 = {"Token": tstr, "User": self._alg_user, "Parameter": jsonstr}
                endata2 = bytes(json.dumps(params2), "utf-8")
                request_headers2 = {"content-type": "application/json"}
                req2 = librequest.Request(url=self._wcsput, data=endata2, method='POST', headers=request_headers2)
                tmp_times = 1
                if os.getenv('prtest') is None:
                    while True:
                        try:
                            with librequest.urlopen(req2) as response:
                                the_page2 = response.read().decode('utf-8')
                                the_pagej2 = simplejson.loads(the_page2)
                                if the_pagej2["success"] == True:
                                    logger1.info("feedback info: %s" % jsonstr)
                                    break
                                else:
                                    logger1.info("feedback false: %s" % jsonstr)
                        except Exception as e:
                            print(e)
                        tmp_times += 1
                        if tmp_times > self._error_retry:
                            break
                        time.sleep(self._error_sleep)
            tmptime = time.time() - starttime
            logger1.info("usetime: %s" % tmptime)
            if os.getenv('prtest') is not None:
                break
            if self._read_interval < tmptime:
                continue
            else:
                if os.getenv('prtest') is not None:
                    continue

    # 订单合并初步判断
    def get_data_judge_run(self, order_new, cancel_list, shelf_info, bench_queue, batchcachenum, benchusenum=1):
        # 读取wcs库新增订单的总表。每隔半小时刷进新单，不满100单(参数) 不开工，除非16点半之后。
        tmplist = [i1["orderID"] for i1 in self._order_info]
        for i1 in order_new:
            if i1["orderID"] in tmplist:
                logger1.info("error: new order %s have the same orderid with old's." % i1["orderID"])
            else:
                i1["cold_counter"] = 0
                self._order_info.append(i1)
        self._order_info = [i1 for i1 in self._order_info if i1["orderID"] not in cancel_list]
        self._order_detail = list(itertools.chain(*[i1["orderDetails"] for i1 in self._order_info]))
        self._order_num = len(self._order_info)
        # 判断执行
        standt = time.strftime("%Y-%m-%d", time.localtime(time.time())) + " " + self._class_over
        nowt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        if self._bench_queue_ave < batchcachenum / benchusenum or (
                        nowt < standt and self._order_buff_low > self._order_num):
            time.sleep(self._cosy_sleep)
            logger1.info("cosy_sleep %s, order_num %s, batchcachenum %s, benchusenum %s." % (
                self._cosy_sleep, self._order_num, batchcachenum, benchusenum))
            return 0
        else:
            # 读入货品总量,内含已占用量。 目前模拟不读正在补货正在排队补货的。真实情况都读。
            self._shelf_info = shelf_info
            self._bench_queue = bench_queue
            print("*************************************************************")
            order_batchs, batchbench_all, replenish_lists, occupied_list, short4_json = self.strategy_policy(
                benchusenum=benchusenum)
            # 更新列表
            order_batchs_idlist = list(set([i1["orderID"] for i1 in order_batchs]))
            self._order_info = [i1 for i1 in self._order_info if i1["orderID"] not in order_batchs_idlist]
            self._order_detail = [i1 for i1 in self._order_detail if i1["orderID"] not in order_batchs_idlist]
            self._order_num = len(self._order_info)
            # 处理完后增加冷单计数
            for i1 in self._order_info:
                i1["cold_counter"] += 1
            print("*************************************************************")
            return order_batchs, batchbench_all, replenish_lists, occupied_list, short4_json

    # 人工规则
    def strategy_policy(self, benchusenum=1):
        # 可直接调用，但需要预先设置
        # self._bench_queue
        # self._replenish_queue
        # 只考虑发送的优先级
        lists = []
        lists.append([i1 for i1 in self._order_info if i1["deliveryPlanTime"] is None])
        lists.append([i1 for i1 in self._order_info if i1["deliveryPlanTime"] is not None])
        logger1.info("order_info length: %s" % len(self._order_info))
        order_batchs_all = []
        batchbench_all = {}
        replenish_list_all = []
        # 1.2(优先批次)
        # 下架：n轮冷单数超过阈值后，设为下一批订单。否则按默认优先级处理。
        # 1. 冷单处理：n轮冷单数还未被分配 或 经过m小时仍未被分配波次 的订单(系统参数)，直接筛选出来用8.2模式分波次。
        # 如果缺货，立即设置最高优先级高的补货任务，供后续补货策略添加任务。（只能在发现冷单处，识别这个问题）。
        # 常规订单：使用所有的未分配订单。
        # 2. 订单集合常规处理：如下图所示，用每个货架可用的货物跟订单做匹配，直到该货架不满足要求，遍历每一种货架取最多的订单，
        # 然后按下图的三种情况决定是否要分波次，或用更多的货架来匹配订单。
        # 以此类推，如果用了所有的货架都不匹配，说明严重缺货，设置二级补货任务，供后续补货策略添加任务。
        # 除了正补货货架和维护货架，其余货架都可以参与订单波次计算（这块临时屏蔽，对所有的货架可读）。
        # 1.2(优先货架)
        # 下架：对分捡好的单子先送走（每批次优先最多的完整单），
        # n轮冷单数超过阈值后，强制补全(避免早到的单子最晚发，同时便于热货架停顿补货)。
        # 添加完成单子数量的新单（根据货架匹配，填加单次最大取货量的单子）。
        # 算完占用量后，根据阈值设置是否缺货，先分配波次分拣，在执行补货任务。
        short4_json = {}
        for i1 in lists:
            if 0 != len(i1):
                fadet = time.time() - self._cold_seconds
                # print(i1)
                if os.getenv('prtest') is None:
                    cold_list = [i2 for i2 in i1 if i2["cold_counter"] > self._cold_round or fadet < time.mktime(
                        time.strptime(i2["submitDate"], "%Y-%m-%d %H:%M:%S.%f"))]
                else:
                    cold_list = [i2 for i2 in i1 if i2["cold_counter"] > self._cold_round or fadet < time.mktime(
                        time.strptime(i2["submitDate"], "%Y-%m-%d %H:%M:%S"))]
                cold_ids = [i2["orderID"] for i2 in cold_list]
                normal_list = [i2 for i2 in i1 if i2["orderID"] not in cold_ids]
                lencold = len(cold_list)
                order_batchs1 = []
                batchbench1 = []
                replenish_lists1 = []
                if 0 != lencold:
                    logger1.info("cold_list length: %s, %s." % (lencold, cold_list))
                    order_batchs1, batchbench1, replenish_lists1, short4_json = self.static_get_strategy(cold_list,
                                                                                                         benchusenum=benchusenum)
                order_batchs2, batchbench2, replenish_lists2, short4_json = self.static_get_strategy(normal_list,
                                                                                                     benchusenum=benchusenum)
                # print("out lists2")
                for i2 in order_batchs1:
                    order_batchs_all.append(i2)
                for i2 in order_batchs2:
                    order_batchs_all.append(i2)
                batchbench_all.update(batchbench1)
                batchbench_all.update(batchbench2)
                for i2 in replenish_lists1:
                    replenish_list_all.append(i2)
                for i2 in replenish_lists2:
                    replenish_list_all.append(i2)
                if 0 == len(order_batchs_all):
                    continue
                break
        occupied_list = []
        repljson = {}
        for i1 in order_batchs_all:
            posstr = str(i1["posID"]) + "-" + str(i1["productBatchID"])
            if posstr not in repljson:
                repljson[posstr] = [i1["amount"], i1["posID"], i1["productBatchID"], i1["commodityID"], i1["stockType"]]
            else:
                repljson[posstr][0] += i1["amount"]
        # for i1 in order_batchs_all:
        #     posstr = str(i1["posID"])
        #     if posstr not in repljson:
        #         repljson[posstr] = i1["amount"]
        #     else:
        #         repljson[posstr] += i1["amount"]
        for i1 in repljson:
            occupied_list.append(
                {"holdAmount": repljson[i1][0], "posID": repljson[i1][1], "productBatchID": repljson[i1][2],
                 "commodityID": repljson[i1][3], "stockType": repljson[i1][4]})
        return order_batchs_all, batchbench_all, replenish_list_all, occupied_list, short4_json

    # 静态下架策略
    def static_get_strategy(self, order_list, benchusenum=1):
        # print("static_get_strategy")
        logger1.info("deal list length: %s, %s" % (len(order_list), order_list))
        benchusenum += 1
        # 仅对输入的订单做补货任务和下架
        # 3. 组合设计，不同策略对分拣速度的影响。
        # 每隔n批订单后调一次货架位置的优先级。
        # 每单的时间消耗。
        # 4. 根据的策略的结果，评判最优的策略。
        # 5. 根据最优策略和输入的批量订单，返回具体分批方式。
        shelf_s_list = list(set(str(i1["shelfID"]) + "-" + str(i1["shelfSide"]) for i1 in self._shelf_info))
        shelf_s_lenth = len(shelf_s_list)
        order_batchs = []
        batchbench = {}
        dealorders = []

        def combikeynow(listobj):
            res = {}
            for i1 in listobj:
                ttstr = str(i1["commodityID"])
                if ttstr not in res:
                    res.__setitem__(ttstr, 0)
                res[ttstr] += i1["availableAmount"]
            return res

        def combikeymax(listobj):
            res = {}
            for i1 in listobj:
                ttstr = str(i1["commodityID"])
                if ttstr not in res:
                    res.__setitem__(ttstr, 0)
                res[ttstr] += i1["maxAmount"]
            return res

        # 遍历每种货架个数的组合
        batch_discrete = []
        discretebatch_sig = 0
        # 记录
        tmp_shelfids = []
        tmp_shelfinfo = []
        tmp_shelf_pos1_num = []
        tmp_ordersout = []
        tmp_ordersoutall = []
        for i1 in range(shelf_s_lenth):
            tmp_ordersin = copy.deepcopy(order_list)
            comblist = list(itertools.combinations(shelf_s_list, i1 + 1))
            # print("遍历每种货架个数的组合: %s,length %s %s" % (i1, len(comblist), comblist))
            if self._combine_shelf < len(comblist):
                # print("broken")
                break
            for i2 in comblist:
                tmp_shelfids.append(i2)
                tmp_shelfinfo.append(combikeynow(
                    [i3 for i3 in self._shelf_info if str(i3["shelfID"]) + "-" + str(i3["shelfSide"]) in i2]))
                tmp_shelf_pos1_num.append(combikeymax(
                    [i3 for i3 in self._shelf_pos1_num if str(i3["shelfID"]) + "-" + str(i3["shelfSide"]) in i2]))
                resinfo = self._virtualsuborder(tmp_shelfinfo[len(tmp_shelfids) - 1], tmp_ordersin)
                resinfoall = self._virtualsuborder(tmp_shelf_pos1_num[len(tmp_shelfids) - 1], tmp_ordersin)
                tmp_ordersoutall.append(resinfoall)
                tmp_ordersout.append(resinfo)
        # print("testsleep")
        # print(time.sleep(500))
        _tmp_less = list(set(sum(tmp_ordersout, [])))
        _tmp_more = list(set(sum(tmp_ordersoutall, [])))
        tmp_length = [len(i2) for i2 in tmp_ordersout]
        # print("multshelf")
        # print(shelf_s_list)
        # print(tmp_shelfids)
        # print(tmp_ordersout)
        # print(tmp_ordersoutall)
        # print(_tmp_less)
        # print(_tmp_more)
        # print(tmp_length)
        if 0 == len(tmp_length):
            logger1.info("ERROR, MINIST_combine_shelf is too low.")
            raise Exception("ERROR, MINIST_combine_shelf is too low.")
        pos = tmp_length.index(max(tmp_length))
        self._tmp_low_efficiency_order = [i2 for i2 in _tmp_more if i2 not in _tmp_less]
        self._tmp_low_efficiency_shelf = tmp_shelfids[pos]
        intbatch = tmp_length[pos] // self._batch_num
        leftbatch = tmp_length[pos] % self._batch_num
        # 加入批次
        if intbatch > 0:
            # print("111")
            dealorders = tmp_ordersout[pos][0:intbatch * self._batch_num]
            order_batchs, batchbench = self._realsuborder(dealorders, tmp_shelfids[pos])
        elif leftbatch >= self._batch_num / (len(tmp_shelfids[pos]) + 2):
            # print("222")
            dealorders = tmp_ordersout[pos][0:leftbatch]
            order_batchs, batchbench = self._realsuborder(dealorders, tmp_shelfids[pos])
        elif leftbatch < self._batch_num / (len(tmp_shelfids[pos]) + 2) and leftbatch > 0 and len(self._order_info) == \
                tmp_length[pos]:
            # print("333")
            dealorders = tmp_ordersout[pos][0:leftbatch]
            print(leftbatch)
            print(len(dealorders), self._batch_num, len(tmp_shelfids[pos]), tmp_length[pos])
            order_batchs, batchbench = self._realsuborder(dealorders, tmp_shelfids[pos])
        elif leftbatch < self._batch_num / (len(tmp_shelfids[pos]) + 2) and leftbatch > 0 and len(self._order_info) != \
                tmp_length[pos]:
            # print("444")
            batch_discrete.append({"shelfs": tmp_shelfids[pos], "orders": tmp_ordersout[pos][0:leftbatch]})
            discretebatch_sig = 1
        elif 0 == tmp_length[pos]:
            pass
        # print("know distr:")
        # print(order_batchs, discretebatch_sig, batch_discrete)
        if 0 == len(order_batchs) and 1 == discretebatch_sig:
            lenthlist = [len(i1["orders"]) / len(i1["shelfs"]) for i1 in batch_discrete]
            pos2 = lenthlist.index(max(lenthlist))
            dealorders = batch_discrete[pos2]["orders"]
            order_batchs, batchbench = self._realsuborder(dealorders, batch_discrete[pos2]["shelfs"])
        elif 0 == len(order_batchs) and 1 != discretebatch_sig:
            logger1.info("low_efficiency_order %s" % self._tmp_low_efficiency_order)
            sanddans = [i1["orderID"] for i1 in order_list if i1 not in self._tmp_low_efficiency_order]
            sanddans2 = [i1["orderID"] for i1 in order_list if i1 in self._tmp_low_efficiency_order]
            sanddans.extend(sanddans2)
            logger1.info("sanddans: %s %s" % (len(sanddans), sanddans))
            # self._replenish_list45 = self._virtualreplenish45(sanddans)
            order_batchs, batchbench, dealorders = self._ordermatch(sanddans)
        # 已经更新可用数量
        restorders = [i1 for i1 in self._order_info if i1["orderID"] not in dealorders]
        restorderskey = [i1["orderID"] for i1 in restorders]
        self._replenish_list45, short4_json = self._virtualreplenish45(restorderskey)
        # print("已经更新可用数量")
        # print("deal orders %s" % dealorders)
        # print("rest orders length %s" % len(restorders))
        # print(restorders)
        # print(order_batchs)
        restorders.sort(key=lambda x: x["submitDate"], reverse=True)
        replenish_lists = self._virtualreplenish()
        # print(short4_json)

        return order_batchs, batchbench, replenish_lists, short4_json

    def _virtualsuborder(self, shelfinfo, orders):
        # 该种货架sku组合，待处理订单，待处理订单细节。
        # 返回该货架的订单列表。
        resorder_detail = set()
        # print("shelfinfo00000")
        # print(shelfinfo)
        # print(orderlf)
        # print("------")
        for i1 in orders:
            flag_in = 1
            ttt_shelfinfo = copy.deepcopy(shelfinfo)
            # print("i1", i1["orderID"])
            for i2 in i1["orderDetails"]:
                commstr = str(i2["commodityID"])
                # print(i2["orderID"], i2["commodityID"], i2["amount"])
                # 判断种类
                if commstr not in ttt_shelfinfo:
                    flag_in = 0
                    break
                ttt_shelfinfo[commstr] -= i2["amount"]
                if ttt_shelfinfo[commstr] < 0:
                    flag_in = 0
                    break
            if 1 == flag_in:
                # print("ttt_shelfinfo")
                # print(ttt_shelfinfo)
                shelfinfo = ttt_shelfinfo
                resorder_detail.add(i1["orderID"])
        resorder_info = list(resorder_detail)
        # print(resorder_info)
        return resorder_info

    def _realsuborder(self, orders, shelfids):
        # 返回该货架的订单列表。
        # print("_realsuborder")
        # print(shelfids)
        order_detail = [i1 for i1 in self._order_detail if i1["orderID"] in orders]
        batchorders = []
        shelfn_info = [i1 for i1 in self._shelf_info if str(i1["shelfID"]) + "-" + str(i1["shelfSide"]) in shelfids]
        # print(orders)
        # print(self._order_detail)
        # print(order_detail)
        # print(shelfn_info)
        for i1 in order_detail:
            for i2 in shelfn_info:
                tmpjson = {}
                available = i2["availableAmount"]
                if i1["commodityID"] == i2["commodityID"] and i1["amount"] > 0 and available > 0:
                    tmpjson["orderID"] = i1["orderID"]
                    tmpjson["commodityID"] = i1["commodityID"]
                    tmpjson["productBatchID"] = i1["productBatchID"]
                    tmpjson["stockType"] = i1["stockType"]
                    tmpjson["shelfID"] = i2["shelfID"]
                    tmpjson["posID"] = i2["posID"]
                    tmpjson["shelfSide"] = i2["shelfSide"]
                    if i1["amount"] > available:
                        tmpjson["amount"] = available
                        i1["amount"] -= available
                    else:
                        tmpjson["amount"] = i1["amount"]
                        i1["amount"] -= tmpjson["amount"]
                    batchorders.append(tmpjson)
        # 根据订单列表 加波次号
        for i1, i2 in enumerate(orders):
            tmpbatch_maxid = self._batch_maxid + i1 // self._batch_num + 1
            for i3 in batchorders:
                if i3["orderID"] == i2:
                    i3["batchID"] = tmpbatch_maxid
        self._batch_maxid += math.ceil(len(orders) / self._batch_num)
        batchbench = self._putbench(batchorders)
        self._order_info = [i1 for i1 in self._order_info if i1["orderID"] not in orders]
        self._order_detail = [i1 for i1 in self._order_detail if i1["orderID"] not in orders]
        return batchorders, batchbench

    def _ordermatch(self, sanddans):
        # 按顺序遍历
        for orderone in sanddans:
            order_detail = [copy.copy(i1) for i1 in self._order_detail if i1["orderID"] == orderone]
            batchorders = []
            shelfn_info = copy.copy(self._shelf_info)
            oksig = 0
            for i1 in order_detail:
                for i2 in shelfn_info:
                    tmpjson = {}
                    available = i2["availableAmount"]
                    if i1["commodityID"] == i2["commodityID"] and i1["amount"] > 0 and available > 0:
                        tmpjson["orderID"] = i1["orderID"]
                        tmpjson["commodityID"] = i1["commodityID"]
                        tmpjson["productBatchID"] = i1["productBatchID"]
                        tmpjson["stockType"] = i1["stockType"]
                        tmpjson["shelfID"] = i2["shelfID"]
                        tmpjson["shelfSide"] = i2["shelfSide"]
                        tmpjson["posID"] = i2["posID"]
                        if i1["amount"] > available:
                            tmpjson["amount"] = available
                            i1["amount"] -= available
                        else:
                            tmpjson["amount"] = i1["amount"]
                            i1["amount"] -= tmpjson["amount"]
                        batchorders.append(tmpjson)
                # 量不足查下一个订单
                if i1["amount"] > 0:
                    break
                else:
                    oksig = 1
            if 1 == oksig:
                dealorders = [orderone]
                self._batch_maxid += 1
                for i1 in batchorders:
                    i1["batchID"] = self._batch_maxid
                batchbench = self._putbench(batchorders)
                # print("sandan error!")
                # print(batchorders)
                # print(batchbench)
                self._order_info = [i1 for i1 in self._order_info if i1["orderID"] not in dealorders]
                self._order_detail = [i1 for i1 in self._order_detail if i1["orderID"] not in dealorders]
                return batchorders, batchbench, dealorders
        logger1.info("nomatch sandan!")
        batchorders = []
        batchbench = {}
        dealorders = []
        return batchorders, batchbench, dealorders

    def _putbench(self, batchorders):
        # 判断是否需要修改工作台个数，选一个最小的队列
        benchs_length = []
        for i1 in self._bench_queue:
            benchs_length.append(len(set([i2["batchID"] for i2 in self._bench_queue[str(i1)]])))
        all_length = sum(benchs_length)
        if all_length / self._bench_usenum > self._bench_queue_up:
            self._bench_usenum += 1
        elif all_length / self._bench_usenum < self._bench_queue_down:
            if self._bench_usenum > 1:
                self._bench_usenum -= 1
        available_list = [i1 for ind, i1 in enumerate(benchs_length) if ind < self._bench_usenum]
        batchbench = {}
        # print("_putbench:")
        # print(batchorders)
        # print(batchbench)
        if 0 != len(available_list):
            batchbench[str(available_list.index(min(available_list)))] = list(
                set([i1["batchID"] for i1 in batchorders]))
        return batchbench

    # 静态上架策略
    def static_put_strategy(self, order_data, shelf_type, shelf_desc, commodity_map):
        # 1. 用数据统计的结果初始化的每个货架的货品分布，作为遗传因子，将遗传因子代入历史订单池；
        # 2. 遗传变量为，每个储位的商品类型，每个储位存放数量；
        # 3. 重复具体货架组合n次，覆盖历史订单，利用率最高的就是最终的静态上架策略；
        replenish_lists = {}
        return replenish_lists

    #  补货部分
    def _virtualreplenish45(self, sanddans):
        # 缺货4
        # 新品5
        # 简单货架匹配不到
        # print("_virtualreplenish45")
        order_detail = [i1 for i1 in self._order_detail if i1["orderID"] in sanddans]
        order_waite = {}
        for i1 in order_detail:
            if i1["commodityID"] not in order_waite:
                order_waite.__setitem__(i1["commodityID"], 0)
            order_waite[i1["commodityID"]] += i1["amount"]
        shelf_waite = {}
        for i1 in self._shelf_info:
            if i1["commodityID"] not in shelf_waite:
                shelf_waite.__setitem__(i1["commodityID"], 0)
            shelf_waite[i1["commodityID"]] += i1["onAmount"]
        short5_list = [i1 for i1 in order_waite if i1 not in shelf_waite]
        short5_json = {i1: order_waite[i1] for i1 in order_waite if i1 in short5_list}
        if len(short5_list) != 0:
            # todo: 异步更新策略
            # 需要 1.各种储位对应各种商品的最大容量。2.现有储位skus。3.待更新skus。4.历史订单。5.限制变量，分拣台总数。
            logger1.info("strategy need update!")
            logger1.info("commodity is not having, json:%s list:%s" % (short5_json, short5_list))
        short4_json = {str(i1): (order_waite[i1] - shelf_waite[i2]) for i1 in order_waite for i2 in shelf_waite if
                       i1 == i2 and order_waite[i1] > shelf_waite[i2]}
        shelfn_info = copy.copy(self._shelf_info)
        # print(order_detail)
        # print(order_waite)
        # print(shelf_waite)
        # print(short5_json)
        # print(short4_json)
        replenish_lists = []
        for i1 in short4_json:
            for i2 in shelfn_info:
                tmpjson = {}
                if int(i1) == i2["commodityID"]:
                    tmpjson["commodityID"] = int(i1)
                    tmpjson["productBatchID"] = i2["productBatchID"]
                    tmpjson["stockType"] = i2["stockType"]
                    tmpjson["shelfID"] = i2["shelfID"]
                    tmpjson["shelfSide"] = i2["shelfSide"]
                    tmpjson["posID"] = i2["posID"]
                    tmpjson["priority"] = 4
                    maxnum = [i3["maxAmount"] for i3 in self._shelf_pos1_num if i3["posID"] == i2["posID"]][0]
                    if short4_json[i1] <= maxnum - i2["onAmount"]:
                        tmpjson["amount"] = short4_json[i1]
                        # tmpjson["shouldbenum"] = i2["onAmount"] + short4_json[i1]
                        i2["onAmount"] = i2["onAmount"] + short4_json[i1]
                        short4_json[i1] = 0
                    else:
                        tmpjson["amount"] = maxnum - i2["onAmount"]
                        # tmpjson["shouldbenum"] = maxnum
                        i2["onAmount"] = maxnum
                        short4_json[i1] -= tmpjson["amount"]
                    replenish_lists.append(tmpjson)
        # print(short4_json)
        # print(pd.DataFrame(replenish_lists))
        for i1 in short4_json:
            if short4_json[i1] > 0:
                print("even load full shelf, commodity is not enough, to much orders SKU:%s NUM:%s" % (
                    i1, short4_json[i1]))
        return replenish_lists, short4_json

    def _virtualreplenish(self):
        # 任意补货1
        # 阈值补货2
        # 低效补货3
        # 缺货4
        # 新品5
        low_efficiency_detail = set(
            [i1["commodityID"] for i1 in self._order_detail if i1["orderID"] in self._tmp_low_efficiency_order])
        # 遍历每一个货架找出超过低效补货3的。
        replenish_detail3 = []
        # print("low_efficiency_detail")
        tmp_pos_shelflist = [[i1["shelfID"], i1["posID"], i1["commodityID"]] for i1 in self._shelf_pos1_num if
                             i1["shelfID"] in self._tmp_low_efficiency_shelf and i1[
                                 "commodityID"] in low_efficiency_detail]
        for i1 in tmp_pos_shelflist:
            for i2 in self._shelf_pos1_num:
                if i2["posID"] == i1:
                    tmp_replenish = {}
                    realnum, stocktype, productbatchid = \
                        [[i3["amount"], i3["stockType"], i3["productBatchID"]] for i3 in self._shelf_info if
                         i3["posID"] == i1][0]
                    tmp_replenish["shelfID"] = i1[0]
                    tmp_replenish["amount"] = i2["maxAmount"] - realnum
                    tmp_replenish["productBatchID"] = productbatchid
                    tmp_replenish["stockType"] = stocktype
                    tmp_replenish["shelfID"] = i1[0]
                    tmp_replenish["shelfSide"] = i2["shelfSide"]
                    tmp_replenish["posID"] = i1[1]
                    tmp_replenish["commodityID"] = i1[2]
                    # tmp_replenish["shouldbenum"] = tmp_replenish["amount"] + realnum
                    tmp_replenish["priority"] = 3
                    replenish_detail3.append(tmp_replenish)
        replenish_shelf3 = list(set([str(i1["shelfID"]) + "-" + str(i1["shelfSide"]) for i1 in replenish_detail3]))
        # 遍历每一个货架找出超过阈值2的。
        replenish_detail2 = []
        for i1 in self._shelf_info:
            for i2 in self._shelf_pos1_num:
                if i2["posID"] == i1["posID"] and i2["commodityID"] == i1["commodityID"]:
                    tmp_replenish = {}
                    # print(i1, i2, i1["availableAmount"], i2["maxAmount"])
                    if i2["maxAmount"] != 0 and self._cell_num_thresh > i1["onAmount"] / i2["maxAmount"]:
                        tmp_replenish["shelfID"] = i1["shelfID"]
                        tmp_replenish["productBatchID"] = i1["productBatchID"]
                        tmp_replenish["stockType"] = i1["stockType"]
                        tmp_replenish["amount"] = i2["maxAmount"] - i1["onAmount"]
                        tmp_replenish["posID"] = i1["posID"]
                        tmp_replenish["shelfSide"] = i1["shelfSide"]
                        tmp_replenish["commodityID"] = i1["commodityID"]
                        # tmp_replenish["shouldbenum"] = tmp_replenish["amount"] + i1["onAmount"]
                        tmp_replenish["priority"] = 2
                        replenish_detail2.append(tmp_replenish)
        replenish_shelf2 = list(set([str(i1["shelfID"]) + "-" + str(i1["shelfSide"]) for i1 in replenish_detail2]))
        # print(replenish_shelf2)
        replenish_shelf45 = list(
            set([str(i1["shelfID"]) + "-" + str(i1["shelfSide"]) for i1 in self._replenish_list45]))
        # 普通优先级1设置，有任务pos的货架，未到阈值的
        replenish_shelf0 = []
        replenish_shelf0.extend(replenish_shelf2)
        replenish_shelf0.extend(replenish_shelf3)
        replenish_shelf0.extend(replenish_shelf45)
        replenish_detail0 = []
        replenish_detail0.extend(replenish_detail2)
        replenish_detail0.extend(replenish_detail3)
        replenish_detail0.extend(self._replenish_list45)
        posjson = {}
        for i1 in replenish_detail0:
            if i1["posID"] not in posjson:
                posjson.__setitem__(i1["posID"], i1["priority"])
            else:
                if posjson[i1["posID"]] < i1["priority"]:
                    posjson[i1["posID"]] = i1["priority"]
        replenish_detail0 = [i1 for i1 in replenish_detail0 for i2 in posjson if
                             i1["posID"] == i2 and posjson[i2] == i1["priority"]]
        replenish_detail_pos = [i1["posID"] for i1 in replenish_detail0]
        replenish_detail1 = []
        for i1 in replenish_shelf0:
            for i2 in self._shelf_pos1_num:
                if str(i2["shelfID"]) + "-" + str(i2["shelfSide"]) == i1:
                    if i2["posID"] not in replenish_detail_pos:
                        posnum, stocktype, productbatchid = \
                            [[i3["onAmount"], i3["stockType"], i3["productBatchID"]] for i3 in self._shelf_info if
                             i3["posID"] == i2["posID"]][0]
                        tmp_replenish = {}
                        tmp_replenish["shelfID"] = i2["shelfID"]
                        tmp_replenish["shelfSide"] = i2["shelfSide"]
                        tmp_replenish["amount"] = i2["maxAmount"] - posnum
                        tmp_replenish["posID"] = i2["posID"]
                        tmp_replenish["productBatchID"] = productbatchid
                        tmp_replenish["stockType"] = stocktype
                        tmp_replenish["commodityID"] = i2["commodityID"]
                        # tmp_replenish["shouldbenum"] = tmp_replenish["amount"] + posnum
                        tmp_replenish["priority"] = 1
                        replenish_detail1.append(tmp_replenish)
        replenish_detail0.extend(replenish_detail1)
        # 只取最小的productBatchID
        produ_json = {}
        for i1 in replenish_detail0:
            if i1["posID"] not in produ_json:
                produ_json[i1["posID"]] = [i1["productBatchID"]]
            else:
                produ_json[i1["posID"]].append(i1["productBatchID"])
        produ_json = {i1: min(produ_json[i1]) for i1 in produ_json}
        replenish_detail0 = [i1 for i1 in replenish_detail0 if
                             i1["posID"] in produ_json and i1["productBatchID"] == produ_json[i1["posID"]]]
        tmplist = [str(i1["shelfID"]) + "-" + str(i1["shelfSide"]) for i1 in replenish_detail0]
        for i1 in tmplist:
            self._replenish_maxid += 1
            for i2 in replenish_detail0:
                if i1 == str(i2["shelfID"]) + "-" + str(i2["shelfSide"]):
                    i2["replenishID"] = self._replenish_maxid
        return replenish_detail0

    def _reformat(self, res, cancelorderlist):
        batnew = res[0]
        bennew = res[1]
        repnew = res[2]
        batchids = list(set(i1["batchID"] for i1 in batnew))
        antijson = {}
        for i1 in bennew:
            for i2 in bennew[i1]:
                antijson[str(i2)] = i1
        batchnews = []
        # print("_reformat")
        # print(antijson)
        # print(bennew)
        # print(batchids)
        # print(batnew)
        for i1 in batchids:
            tmpjson = {}
            tmporderids = []
            for i2 in batnew:
                if i2["batchID"] == i1:
                    tmpjson = {}
                    tmporderids.append(i2["orderID"])
                    tmpjson["batchID"] = i1
                    tmpjson["workPlaceID"] = int(antijson[str(i1)])
                    tmpjson["priority"] = 0
                    tmpjson["orderIDs"] = list(set(tmporderids))
            batchnews.append(tmpjson)
        for i1 in batchnews:
            tmplist = []
            for i2 in batnew:
                if i2["orderID"] in i1["orderIDs"]:
                    ttjson = {}
                    ttjson["orderID"] = i2["orderID"]
                    ttjson["commodityID"] = i2["commodityID"]
                    ttjson["productBatchID"] = i2["productBatchID"]
                    ttjson["stockType"] = i2["stockType"]
                    ttjson["amount"] = i2["amount"]
                    ttjson["shelfID"] = i2["shelfID"]
                    ttjson["shelfSide"] = i2["shelfSide"]
                    ttjson["posID"] = i2["posID"]
                    tmplist.append(ttjson)
            i1["orderPickDetails"] = tmplist
        # ReplenishInfo
        repids = list(set(i1["replenishID"] for i1 in repnew))
        replenews = []
        for i1 in repids:
            tmpjson = {}
            tmplist = []
            for i2 in repnew:
                if i2["replenishID"] == i1:
                    tmpjson["replenishID"] = i2["replenishID"]
                    tmpjson["shelfID"] = i2["shelfID"]
                    tmpjson["shelfSide"] = i2["shelfSide"]
                    ttjson = {}
                    ttjson["replenishID"] = i2["replenishID"]
                    ttjson["commodityID"] = i2["commodityID"]
                    ttjson["posID"] = i2["posID"]
                    ttjson["stockType"] = i2["stockType"]
                    ttjson["productBatchID"] = i2["productBatchID"]
                    ttjson["amount"] = i2["amount"]
                    ttjson["priority"] = i2["priority"]
                    tmplist.append(ttjson)
            tmpjson["replenishDetails"] = tmplist
            replenews.append(tmpjson)
        # print("batchnews_length: ", len(batchnews), batchnews)
        params2 = {"batchInfos": batchnews, "replenishInfos": replenews, "occupiedAmounts": res[3],
                   "canceledOrders": cancelorderlist, "totalNeed": res[4]}
        return params2

    # 策略更新
    def strategy_update(self, order_data, shelf_status):
        # 变更时只考虑未占用货架，且设置货架状态待变更3
        # 订单历史信息表（带时间参数，取之后左右的包含新单未处理的），货架类型表，商品信息表，储位sku可容纳数量映射表
        pass


if '__main__' == __name__:
    pass
