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

import pandas as pd
import simplejson

pd.set_option('display.max_columns', None)


class SimuStrategy(object):
    def __init__(self, conf, model_json):
        self._conf = conf
        self._model_json = model_json
        # // 单分拣台队列货物挤压数量上阈值(开新线用)
        self._betch_queue_up = self._conf["betch_queue_up"]
        # // 单分拣台队列货物挤压数量下阈值(停线用)
        self._betch_queue_down = self._conf["betch_queue_down"]
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
        self._order_max = self._model_json["order_max"]
        self._batch_num = self._conf["batch_num"]
        self._para_path = self._model_json["para_path"]
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
        self._bench_queue = {"1": [{"BatchID": 3}, {"BatchID": 4}], "2": [{"BatchID": 5}, {"BatchID": 6}]}
        if os.getenv('prtest') is None:
            # "MaxOrderNum": -1 给全部的
            paratmp = {"RequestStatus": 1, "MaxOrderNum": -1, "GetOrderStatus": 1}
            params = {"Parameter": json.dumps(paratmp)}
            endata = bytes(json.dumps(params), "utf-8")
            request_headers = {"content-type": "application/json"}
            req = librequest.Request(url=self._wcsinit, data=endata, method='POST', headers=request_headers)
            with librequest.urlopen(req) as response:
                ori_page = response.read().decode('utf-8')
                the_page0 = simplejson.loads(ori_page)
                the_page = the_page0["data"][0]
        else:
            getstr = {"success": "", "message": "Succ",
                      "data":
                          {
                              "AlgOrderInfo": [
                                  {
                                      "OrderID": 3,
                                      "SubmitDate": "2019-01-01 11:11:11",
                                      "SaleDate": "2019-01-01 11:11:11",
                                      "DeliveryPlanTime": "2019-01-01 11:11:11",
                                      "OrderDetail": [
                                          {"OrderID": 3, "CommodityID": 2, "ProductBatchID": 3, "StockType": 1,
                                           "Amount": 6},
                                          {"OrderID": 3, "CommodityID": 3, "ProductBatchID": 3, "StockType": 1,
                                           "Amount": 6}
                                      ]
                                  },
                                  {
                                      "OrderID": 4,
                                      "SubmitDate": "2018-01-01 11:11:11",
                                      "SaleDate": "2018-01-01 11:11:11",
                                      "DeliveryPlanTime": "2018-01-01 11:11:11",
                                      "OrderDetail": [
                                          {"OrderID": 4, "CommodityID": 2, "ProductBatchID": 3, "StockType": 1,
                                           "Amount": 6},
                                          {"OrderID": 4, "CommodityID": 3, "ProductBatchID": 3, "StockType": 1,
                                           "Amount": 6}
                                      ]
                                  }
                              ],
                              "TaskBaseInfo": {"ReplenishMaxID": 11, "BatchMaxID": 22},
                              "StorageCapacity": [
                                  {"CommodityID": 1, "ShelfID": 3, "PosID": 2, "MaxNum": 3},
                                  {"CommodityID": 4, "ShelfID": 3, "PosID": 7, "MaxNum": 4},
                                  {"CommodityID": 4, "ShelfID": 9, "PosID": 5, "MaxNum": 6},
                                  {"CommodityID": 9, "ShelfID": 9, "PosID": 15, "MaxNum": 6},
                              ]
                          },
                      "total": 10}
            the_page = getstr["data"]
        for i1 in the_page["AlgOrderInfo"]:
            i1["cold_counter"] = 0
            self._order_info.append(i1)
        self._order_num = len(self._order_info)
        self._batch_maxid = the_page["TaskBaseInfo"]["BatchMaxID"]
        self._replenish_maxid = the_page["TaskBaseInfo"]["ReplenishMaxID"]
        self._shelf_pos1_num = the_page["StorageCapacity"]
        # self._shelf_pos_num = the_page["shelf_pos_num"]

    def dump_paras(self, parafile):
        tmpjson = {
            "betch_queue_up": self._betch_queue_up,
            "betch_queue_down": self._betch_queue_down,
            "replenish_queue_up": self._replenish_queue_up,
            "replenish_queue_down": self._replenish_queue_down,
        }
        simplejson.dumps(tmpjson, open(parafile, mode='w'))

    # 策略常态
    def pick_replenish_strategy(self):
        # 补货优先级，0. 不需要，1. 缓存区有就补，没有就算了，2.正常按需求补，3.低效补，4.缺货补，5.新品
        while True:
            # 1. 数据请求
            starttime = time.time()
            paratmp = {"RequestStatus": 2, "MaxOrderNum": self._order_max - self._order_num, "GetOrderStatus": 0}
            params = {"Parameter": json.dumps(paratmp)}
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
                    gddd = {"success": "", "message": "Succ",
                            "data": [
                                {
                                    "AlgOrderInfo": [
                                        {"OrderID": 1, "SubmitDate": "2019-01-01 11:11:11",
                                         "SaleDate": "2019-01-01 11:11:11",
                                         "DeliveryPlanTime": "2019-01-01 11:11:11", "OrderDetail": [
                                            {"OrderID": 1, "CommodityID": 4, "ProductBatchID": 3, "StockType": 1,
                                             "Amount": 2}]},
                                        {"OrderID": 2, "SubmitDate": "2018-01-01 11:11:11",
                                         "SaleDate": "2018-01-01 11:11:11",
                                         "DeliveryPlanTime": "2018-01-01 11:11:11", "OrderDetail": [
                                            {"OrderID": 2, "CommodityID": 1, "ProductBatchID": 3, "StockType": 1,
                                             "Amount": 2},
                                            {"OrderID": 2, "CommodityID": 4, "ProductBatchID": 3, "StockType": 1,
                                             "Amount": 6}]}
                                    ],
                                    "CancelingOrder": [3],
                                    "AvailableStorageInfo": [
                                        {"PosID": 7, "ShelfID": 3, "CommodityID": 4, "ProductBatchID": 5,
                                         "StockType": 6, "AvailableAmount": 3, "Amount": 3},
                                        {"PosID": 2, "ShelfID": 3, "CommodityID": 1, "ProductBatchID": 5,
                                         "StockType": 6, "AvailableAmount": 2, "Amount": 2},
                                        {"PosID": 5, "ShelfID": 9, "CommodityID": 4, "ProductBatchID": 5,
                                         "StockType": 6, "AvailableAmount": 5, "Amount": 5},
                                        {"PosID": 15, "ShelfID": 9, "CommodityID": 9, "ProductBatchID": 5,
                                         "StockType": 6, "AvailableAmount": 4, "Amount": 4}
                                    ]
                                }
                            ], "total": 10
                            }
                    the_page = gddd["data"][0]
            except Exception as e:
                print("error: when get order_info at %s. %s" % (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), e))
            res = self.get_data_judge_run(the_page["AlgOrderInfo"], the_page["CancelingOrder"],
                                          the_page["AvailableStorageInfo"], self._bench_queue,
                                          betchusenum=self._bench_usenum)
            if 0 == res:
                pass
            else:
                batnew = res[0]
                bennew = res[1]
                repnew = res[2]
                batchids = list(itertools.chain(*[bennew[i1] for i1 in bennew]))
                antijson = {}
                for i1 in bennew:
                    for i2 in bennew[i1]:
                        antijson[str(i2)] = i1
                batchnews = []
                for i1 in batchids:
                    shelfids = []
                    for i2 in batnew:
                        if i2["BatchID"] == i1 and i2["ShelfID"] not in shelfids:
                            tmpjson = {}
                            shelfids.append(i2["ShelfID"])
                            tmpjson["BatchID"] = i1
                            tmpjson["ShelfID"] = i2["ShelfID"]
                            tmpjson["WorkPlaceID"] = int(antijson[str(i1)])
                            tmpjson["Priority"] = 0
                            batchnews.append(tmpjson)
                for i1 in batchnews:
                    tmplist = []
                    for i2 in batnew:
                        if i2["BatchID"] == i1["BatchID"] and i2["ShelfID"] == i1["ShelfID"]:
                            ttjson = {}
                            tmpjson["BatchID"] = i2["BatchID"]
                            tmpjson["CommodityID"] = i2["CommodityID"]
                            tmpjson["OrderID"] = i2["OrderID"]
                            tmpjson["PosID"] = i2["PosID"]
                            tmpjson["StockType"] = i2["StockType"]
                            tmpjson["ProductBatchID"] = i2["ProductBatchID"]
                            tmpjson["Num"] = i2["Num"]
                            tmplist.append(ttjson)
                    i1["AlgOrderPickDetail"] = tmplist
                # ReplenishInfo
                repids = list(set(i1["ReplenishID"] for i1 in repnew))
                replenews = []
                for i1 in repids:
                    tmpjson = {}
                    tmplist = []
                    for i2 in repnew:
                        if i2["ReplenishID"] == i1:
                            tmpjson["ReplenishID"] = i2["ReplenishID"]
                            tmpjson["ShelfID"] = i2["ShelfID"]
                            ttjson = {}
                            ttjson["ReplenishID"] = i2["ReplenishID"]
                            ttjson["CommodityID"] = i2["CommodityID"]
                            ttjson["PosID"] = i2["PosID"]
                            ttjson["StockType"] = i2["StockType"]
                            ttjson["ProductBatchID"] = i2["ProductBatchID"]
                            ttjson["Amount"] = i2["Amount"]
                            ttjson["Priority"] = i2["Priority"]
                            tmplist.append(ttjson)
                    tmpjson["ReplenishDetail"] = tmplist
                    replenews.append(tmpjson)
                params2 = {"BatchInfo": batchnews, "ReplenishInfo": replenews, "OccupiedCommodity": res[3],
                           "CanceledOrder": the_page["CancelingOrder"]}
                endata2 = bytes(json.dumps(params2), "utf-8")
                # print(endata2)
                request_headers2 = {"content-type": "application/json"}
                req2 = librequest.Request(url=self._wcsput, data=endata2, method='POST', headers=request_headers2)
                tmp_times = 1
                if os.getenv('prtest') is None:
                    while True:
                        try:
                            with librequest.urlopen(req2) as response:
                                the_page2 = response.read().decode('utf-8')
                                if "ok" == the_page2:
                                    break
                        except Exception as e:
                            print(e)
                            tmp_times += 1
                            if tmp_times > self._error_retry:
                                break
            tmptime = time.time() - starttime
            print("usetime: %s" % tmptime)
            if os.getenv('prtest') is not None:
                break
            if self._read_interval < tmptime:
                continue
            else:
                if os.getenv('prtest') is not None:
                    continue
                time.sleep(self._read_interval - tmptime)

    # 订单合并初步判断
    def get_data_judge_run(self, order_new, cancel_list, shelf_info, bench_queue, betchusenum=1):
        # 读取wcs库新增订单的总表。每隔半小时刷进新单，不满100单(参数) 不开工，除非16点半之后。
        tmplist = [i1["OrderID"] for i1 in self._order_info]
        for i1 in order_new:
            if i1["OrderID"] in tmplist:
                print("error: new order %s have the same orderid with old's." % i1["OrderID"])
            else:
                i1["cold_counter"] = 0
                self._order_info.append(i1)
        self._order_info = [i1 for i1 in self._order_info if i1["OrderID"] not in cancel_list]
        self._order_detail = list(itertools.chain(*[i1["OrderDetail"] for i1 in self._order_info]))
        self._order_num = len(self._order_info)
        # 判断执行
        standt = time.strftime("%Y-%m-%d", time.localtime(time.time())) + " " + self._class_over
        nowt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        if self._order_buff_low > self._order_num and nowt < standt:
            return 0
        else:
            # 读入货品总量,内含已占用量。 目前模拟不读正在补货正在排队补货的。真实情况都读。
            self._shelf_info = shelf_info
            self._bench_queue = bench_queue
            print("get_data_judge_run:")
            print("_order_s:")
            print(list(i1["OrderID"] for i1 in self._order_info))
            print("_order_detail:")
            print(pd.DataFrame(self._order_detail))
            print("_shelf_info:")
            print(pd.DataFrame(self._shelf_info))
            print("_shelf_pos1_num:")
            print(pd.DataFrame(self._shelf_pos1_num))
            print("cell_num_thresh: ", self._cell_num_thresh)
            print("*************************************************************")
            order_batchs, batchbetch_all, replenish_lists, occupied_list = self.strategy_policy(betchusenum=betchusenum)
            # 更新列表
            order_batchs_idlist = list(set([i1["OrderID"] for i1 in order_batchs]))
            self._order_info = [i1 for i1 in self._order_info if i1["OrderID"] not in order_batchs_idlist]
            self._order_detail = [i1 for i1 in self._order_detail if i1["OrderID"] not in order_batchs_idlist]
            # 处理完后增加冷单计数
            for i1 in self._order_info:
                i1["cold_counter"] += 1
            print("*************************************************************")
            print("sending to server")
            print("_order_s:")
            print(list(i1["OrderID"] for i1 in self._order_info))
            print("order_batchs:")
            print(pd.DataFrame(order_batchs))
            print("replenish_lists:")
            print(pd.DataFrame(replenish_lists))
            print("occupied_list:")
            print(pd.DataFrame(occupied_list))
            return order_batchs, batchbetch_all, replenish_lists, occupied_list

    # 人工规则
    def strategy_policy(self, betchusenum=1):
        # 可直接调用，但需要预先设置
        # self._bench_queue
        # self._replenish_queue
        # 只考虑发送的优先级
        lists = []
        lists.append([i1 for i1 in self._order_info if i1["DeliveryPlanTime"] is None])
        lists.append([i1 for i1 in self._order_info if i1["DeliveryPlanTime"] is not None])
        print("order_info length:", len(self._order_info))
        order_batchs_all = []
        batchbetch_all = {}
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
        # time.sleep(10)
        for i1 in lists:
            if 0 != len(i1):
                fadet = time.time() - self._cold_seconds
                cold_list = [i2 for i2 in i1 if i2["cold_counter"] > self._cold_round or fadet < time.mktime(
                    time.strptime(i2["SubmitDate"], "%Y-%m-%d %H:%M:%S"))]
                cold_ids = [i2["OrderID"] for i2 in cold_list]
                normal_list = [i2 for i2 in i1 if i2["OrderID"] not in cold_ids]
                lencold = len(cold_list)
                order_batchs1 = []
                batchbetch1 = []
                replenish_lists1 = []
                if 0 != lencold:
                    print("cold_list length:", lencold)
                    order_batchs1, batchbetch1, replenish_lists1 = self.static_get_strategy(cold_list,
                                                                                            betchusenum=betchusenum)
                order_batchs2, batchbetch2, replenish_lists2 = self.static_get_strategy(normal_list,
                                                                                        betchusenum=betchusenum)
                # print("out lists2")
                for i2 in order_batchs1:
                    order_batchs_all.append(i2)
                for i2 in order_batchs2:
                    order_batchs_all.append(i2)
                batchbetch_all.update(batchbetch1)
                batchbetch_all.update(batchbetch2)
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
            posstr = str(i1["PosID"])
            if posstr not in repljson:
                repljson[posstr] = i1["Num"]
            else:
                repljson[posstr] += i1["Num"]
        for i1 in repljson:
            occupied_list.append({"PosID": i1, "HoldNum": repljson[i1]})
        return order_batchs_all, batchbetch_all, replenish_list_all, occupied_list

    # 静态下架策略
    def static_get_strategy(self, order_list, betchusenum=1):
        print("static_get_strategy")
        print("deal list length: %s" % len(order_list))
        betchusenum += 1
        # 仅对输入的订单做补货任务和下架
        # 3. 组合设计，不同策略对分拣速度的影响。
        # 每隔n批订单后调一次货架位置的优先级。
        # 每单的时间消耗。
        # 4. 根据的策略的结果，评判最优的策略。
        # 5. 根据最优策略和输入的批量订单，返回具体分批方式。
        shelf_list = list(set(i1["ShelfID"] for i1 in self._shelf_info))
        shelf_lenth = len(shelf_list)
        order_batchs = []
        batchbetch = {}
        dealorders = []

        def combikeynow(listobj):
            res = {}
            for i1 in listobj:
                ttstr = str(i1["CommodityID"])
                if ttstr not in res:
                    res.__setitem__(ttstr, 0)
                res[ttstr] += i1["AvailableAmount"]
            return res

        def combikeymax(listobj):
            res = {}
            for i1 in listobj:
                ttstr = str(i1["CommodityID"])
                if ttstr not in res:
                    res.__setitem__(ttstr, 0)
                res[ttstr] += i1["MaxNum"]
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
        for i1 in range(shelf_lenth):
            tmp_ordersin = copy.deepcopy(order_list)
            comblist = list(itertools.combinations(shelf_list, i1 + 1))
            # print("遍历每种货架个数的组合: %s" % i1)
            if self._combine_shelf < len(comblist):
                break
            for i2 in comblist:
                tmp_shelfids.append(i2)
                tmp_shelfinfo.append(combikeynow([i3 for i3 in self._shelf_info if i3["ShelfID"] in i2]))
                tmp_shelf_pos1_num.append(combikeymax([i3 for i3 in self._shelf_pos1_num if i3["ShelfID"] in i2]))
                resinfo = self._virtualsuborder(tmp_shelfinfo[len(tmp_shelfids) - 1], tmp_ordersin)
                resinfoall = self._virtualsuborder(tmp_shelf_pos1_num[len(tmp_shelfids) - 1], tmp_ordersin)
                tmp_ordersoutall.append(resinfoall)
                tmp_ordersout.append(resinfo)
        _tmp_less = list(set(sum(tmp_ordersout, [])))
        _tmp_more = list(set(sum(tmp_ordersoutall, [])))
        tmp_length = [len(i2) for i2 in tmp_ordersout]
        # print("multshelf")
        # print(shelf_list)
        # print(tmp_shelfids)
        # print(tmp_ordersout)
        # print(tmp_ordersoutall)
        # print(_tmp_less)
        # print(_tmp_more)
        # print(tmp_length)
        pos = tmp_length.index(max(tmp_length))
        self._tmp_low_efficiency_order = [i2 for i2 in _tmp_more if i2 not in _tmp_less]
        self._tmp_low_efficiency_shelf = tmp_shelfids[pos]
        intbatch = tmp_length[pos] // self._batch_num
        leftbatch = tmp_length[pos] % self._batch_num
        # 加入批次
        if intbatch > 0:
            print("111")
            dealorders = tmp_ordersout[pos][0:intbatch * self._batch_num]
            order_batchs, batchbetch = self._realsuborder(dealorders, tmp_shelfids[pos])
        elif leftbatch >= self._batch_num / (i1 + 2):
            print("222")
            dealorders = tmp_ordersout[pos][0:leftbatch]
            order_batchs, batchbetch = self._realsuborder(dealorders, tmp_shelfids[pos])
        elif leftbatch < self._batch_num / (i1 + 2) and leftbatch > 0 and len(self._order_info) == tmp_length[pos]:
            print("333")
            dealorders = tmp_ordersout[pos][0:leftbatch]
            order_batchs, batchbetch = self._realsuborder(dealorders, tmp_shelfids[pos])
        elif leftbatch < self._batch_num / (i1 + 2) and leftbatch > 0 and len(self._order_info) != tmp_length[pos]:
            print("444")
            # print(tmp_shelfids[pos], tmp_ordersout[pos])
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
            order_batchs, batchbetch = self._realsuborder(dealorders, batch_discrete[pos2]["shelfs"])
        elif 0 == len(order_batchs) and 1 != discretebatch_sig:
            # 散单预判
            print("散单预判")
            # print(self._tmp_low_efficiency_order)
            sanddans = [i1["OrderID"] for i1 in order_list if i1 not in self._tmp_low_efficiency_order]
            self._replenish_list45 = self._virtualreplenish45(sanddans)
            if 0 != len(sanddans):
                orderone = sanddans[0]
            else:
                orderone = self._tmp_low_efficiency_order[0]
            dealorders = [orderone]
            order_batchs, batchbench = self._ordermatch(orderone)
        # 已经更新可用数量
        restorders = [i1 for i1 in self._order_info if i1["OrderID"] not in dealorders]
        # print("已经更新可用数量")
        # print("deal orders %s" % dealorders)
        # print("rest orders length %s" % len(restorders))
        # print(restorders)
        # print(order_batchs)
        restorders.sort(key=lambda x: x["SubmitDate"], reverse=True)
        replenish_lists = self._virtualreplenish()
        return order_batchs, batchbetch, replenish_lists

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
            # print("i1", i1["OrderID"])
            for i2 in i1["OrderDetail"]:
                commstr = str(i2["CommodityID"])
                # print(i2["OrderID"], i2["CommodityID"], i2["Amount"])
                # 判断种类
                if commstr not in ttt_shelfinfo:
                    flag_in = 0
                    break
                ttt_shelfinfo[commstr] -= i2["Amount"]
                if ttt_shelfinfo[commstr] < 0:
                    flag_in = 0
                    break
            if 1 == flag_in:
                # print("ttt_shelfinfo")
                # print(ttt_shelfinfo)
                shelfinfo = ttt_shelfinfo
                resorder_detail.add(i1["OrderID"])
        resorder_info = list(resorder_detail)
        # print(resorder_info)
        return resorder_info

    def _realsuborder(self, orders, shelfids):
        # 返回该货架的订单列表。
        # print("_realsuborder")
        order_detail = [i1 for i1 in self._order_detail if i1["OrderID"] in orders]
        batchorders = []
        shelfn_info = [i1 for i1 in self._shelf_info if i1["ShelfID"] in shelfids]
        # print(orders)
        # print(self._order_detail)
        # print(order_detail)
        # print(shelfn_info)
        for i1 in order_detail:
            for i2 in shelfn_info:
                tmpjson = {}
                available = i2["AvailableAmount"]
                if i1["CommodityID"] == i2["CommodityID"] and i1["Amount"] > 0 and available > 0:
                    tmpjson["OrderID"] = i1["OrderID"]
                    tmpjson["CommodityID"] = i1["CommodityID"]
                    tmpjson["ProductBatchID"] = i1["ProductBatchID"]
                    tmpjson["StockType"] = i1["StockType"]
                    tmpjson["ShelfID"] = i2["ShelfID"]
                    tmpjson["PosID"] = i2["PosID"]
                    if i1["Amount"] > available:
                        tmpjson["Num"] = available
                        i1["Amount"] -= available
                    else:
                        tmpjson["Num"] = i1["Amount"]
                        i1["Amount"] -= tmpjson["Num"]
                    batchorders.append(tmpjson)
        # 根据订单列表 加波次号
        for i1, i2 in enumerate(orders):
            tmpbatch_maxid = self._batch_maxid + i1 // self._batch_num + 1
            for i3 in batchorders:
                if i3["OrderID"] == i2:
                    i3["BatchID"] = tmpbatch_maxid
        # print(batchorders)
        batchbench = self._putbench(batchorders)
        # print(batchbench)
        self._order_info = [i1 for i1 in self._order_info if i1["OrderID"] not in orders]
        self._order_detail = [i1 for i1 in self._order_detail if i1["OrderID"] not in orders]
        return batchorders, batchbench

    def _ordermatch(self, orderone):
        order_detail = [i1 for i1 in self._order_detail if i1["OrderID"] == orderone]
        batchorders = []
        shelfn_info = copy.copy(self._shelf_info)
        for i1 in order_detail:
            for i2 in shelfn_info:
                tmpjson = {}
                available = i2["AvailableAmount"]
                if i1["CommodityID"] == i2["CommodityID"] and available > 0:
                    tmpjson["OrderID"] = i1["OrderID"]
                    tmpjson["CommodityID"] = i1["CommodityID"]
                    tmpjson["ProductBatchID"] = i1["ProductBatchID"]
                    tmpjson["StockType"] = i1["StockType"]
                    tmpjson["ShelfID"] = i2["ShelfID"]
                    tmpjson["PosID"] = i2["PosID"]
                    if i1["Amount"] > available:
                        tmpjson["Num"] = available
                        i1["Amount"] -= available
                    else:
                        tmpjson["Num"] = i1["Amount"]
                        i1["Amount"] = 0
                    batchorders.append(tmpjson)
        batchbench = self._putbench(batchorders)
        return batchorders, batchbench

    def _putbench(self, batchorders):
        # 判断是否需要修改工作台个数，选一个最小的队列
        benchs_length = []
        for i1 in self._bench_queue:
            benchs_length.append(len(set([i2["BatchID"] for i2 in self._bench_queue[str(i1)]])))
        all_length = sum(benchs_length)
        if all_length / self._bench_usenum > self._betch_queue_up:
            self._bench_usenum += 1
        elif all_length / self._bench_usenum < self._betch_queue_down:
            if self._bench_usenum > 1:
                self._bench_usenum -= 1
        available_list = [i1 for ind, i1 in enumerate(benchs_length) if ind < self._bench_usenum]
        batchbench = {}
        if 0 != len(available_list):
            batchbench[str(available_list.index(min(available_list)))] = list(
                set([i1["BatchID"] for i1 in batchorders]))
        # print("_putbench:")
        # print(batchbench)
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
        order_detail = [i1 for i1 in self._order_detail if i1["OrderID"] in sanddans]
        order_waite = {}
        for i1 in order_detail:
            if i1["CommodityID"] not in order_waite:
                order_waite.__setitem__(i1["CommodityID"], 0)
            order_waite[i1["CommodityID"]] += i1["Amount"]
        shelf_waite = {}
        for i1 in self._shelf_info:
            if i1["CommodityID"] not in shelf_waite:
                shelf_waite.__setitem__(i1["CommodityID"], 0)
            shelf_waite[i1["CommodityID"]] += i1["Amount"]
        short5_list = [i1 for i1 in order_waite if i1 not in shelf_waite]
        short5_json = {i1: order_waite[i1] for i1 in order_waite if i1 in short5_list}
        if short5_list != 0:
            # todo: 异步更新策略
            print("strategy need update!")
        short4_json = {i1: (order_waite[i1] - shelf_waite[i2]) for i1 in order_waite for i2 in shelf_waite if
                       i1 == i2 and order_waite[i1] > shelf_waite[i2]}
        shelfn_info = copy.copy(self._shelf_info)
        replenish_lists = []
        for i1 in short4_json:
            for i2 in shelfn_info:
                tmpjson = {}
                if i1 == i2["CommodityID"]:
                    tmpjson["CommodityID"] = i1
                    tmpjson["ProductBatchID"] = i2["ProductBatchID"]
                    tmpjson["StockType"] = i2["StockType"]
                    tmpjson["ShelfID"] = i2["ShelfID"]
                    tmpjson["PosID"] = i2["PosID"]
                    tmpjson["Priority"] = 4
                    maxnum = [i3["MaxNum"] for i3 in self._shelf_pos1_num if i3["PosID"] == i2["PosID"]][0]
                    if short4_json[i1] <= maxnum - i2["Amount"]:
                        tmpjson["Amount"] = short4_json[i1]
                        # tmpjson["shouldbenum"] = i2["Amount"] + short4_json[i1]
                        i2["Amount"] = i2["Amount"] + short4_json[i1]
                        short4_json[i1] = 0
                    else:
                        tmpjson["Amount"] = maxnum - i2["Amount"]
                        # tmpjson["shouldbenum"] = maxnum
                        i2["Amount"] = maxnum
                        short4_json[i1] -= tmpjson["Amount"]
                    replenish_lists.append(tmpjson)
        for i1 in short4_json:
            if short4_json[i1] > 0:
                print("commodity is not enough, to much orders %s %s" % (i1, short4_json[i1]))
        return replenish_lists

    def _virtualreplenish(self):
        # 任意补货1
        # 阈值补货2
        # 低效补货3
        # 缺货4
        # 新品5
        # print("******************************************************")
        # print("_virtualreplenish")
        low_efficiency_detail = set(
            [i1["CommodityID"] for i1 in self._order_detail if i1["OrderID"] in self._tmp_low_efficiency_order])
        # 遍历每一个货架找出超过低效补货3的。
        replenish_detail3 = []
        # print("low_efficiency_detail")
        tmp_pos_shelflist = [[i1["ShelfID"], i1["PosID"], i1["CommodityID"]] for i1 in self._shelf_pos1_num if
                             i1["ShelfID"] in self._tmp_low_efficiency_shelf and i1[
                                 "CommodityID"] in low_efficiency_detail]
        for i1 in tmp_pos_shelflist:
            for i2 in self._shelf_pos1_num:
                if i2["PosID"] == i1:
                    tmp_replenish = {}
                    realnum, stocktype, productbatchid = \
                        [[i3["Amount"], i3["StockType"], i3["ProductBatchID"]] for i3 in self._shelf_info if
                         i3["PosID"] == i1][0]
                    tmp_replenish["ShelfID"] = i1[0]
                    tmp_replenish["Amount"] = i2["MaxNum"] - realnum
                    tmp_replenish["ProductBatchID"] = productbatchid
                    tmp_replenish["StockType"] = stocktype
                    tmp_replenish["ShelfID"] = i1[0]
                    tmp_replenish["PosID"] = i1[1]
                    tmp_replenish["CommodityID"] = i1[2]
                    # tmp_replenish["shouldbenum"] = tmp_replenish["Amount"] + realnum
                    tmp_replenish["Priority"] = 3
                    replenish_detail3.append(tmp_replenish)
        replenish_shelf3 = list(set([i1["ShelfID"] for i1 in replenish_detail3]))
        # 遍历每一个货架找出超过阈值2的。
        replenish_detail2 = []
        for i1 in self._shelf_info:
            for i2 in self._shelf_pos1_num:
                if i2["PosID"] == i1["PosID"] and i2["CommodityID"] == i1["CommodityID"]:
                    tmp_replenish = {}
                    # print(i1, i2, i1["AvailableAmount"], i2["MaxNum"])
                    if self._cell_num_thresh > i1["Amount"] / i2["MaxNum"]:
                        tmp_replenish["ShelfID"] = i1["ShelfID"]
                        tmp_replenish["ProductBatchID"] = i1["ProductBatchID"]
                        tmp_replenish["StockType"] = i1["StockType"]
                        tmp_replenish["Amount"] = i2["MaxNum"] - i1["Amount"]
                        tmp_replenish["PosID"] = i1["PosID"]
                        tmp_replenish["CommodityID"] = i1["CommodityID"]
                        # tmp_replenish["shouldbenum"] = tmp_replenish["Amount"] + i1["Amount"]
                        tmp_replenish["Priority"] = 2
                        replenish_detail2.append(tmp_replenish)
        replenish_shelf2 = list(set([i1["ShelfID"] for i1 in replenish_detail2]))
        # print(replenish_shelf2)
        # 普通优先级1设置，有任务pos的货架，未到阈值的
        replenish_shelf0 = []
        replenish_shelf0.extend(replenish_shelf2)
        replenish_shelf0.extend(replenish_shelf3)
        replenish_detail0 = []
        replenish_detail0.extend(replenish_detail2)
        replenish_detail0.extend(replenish_detail3)
        replenish_detail0.extend(self._replenish_list45)
        posjson = {}
        for i1 in replenish_detail0:
            if i1["PosID"] not in posjson:
                posjson.__setitem__(i1["PosID"], i1["Priority"])
            else:
                if posjson[i1["PosID"]] < i1["Priority"]:
                    posjson[i1["PosID"]] = i1["Priority"]
        replenish_detail0 = [i1 for i1 in replenish_detail0 for i2 in posjson if
                             i1["PosID"] == i2 and posjson[i2] == i1["Priority"]]
        # print(replenish_detail0)
        replenish_detail_pos = [i1["PosID"] for i1 in replenish_detail0]
        replenish_detail1 = []
        for i1 in replenish_shelf0:
            for i2 in self._shelf_pos1_num:
                if i2["ShelfID"] == i1:
                    if i2["PosID"] not in replenish_detail_pos:
                        posnum, stocktype, productbatchid = \
                            [[i3["Amount"], i3["StockType"], i3["ProductBatchID"]] for i3 in self._shelf_info if
                             i3["PosID"] == i2["PosID"]][0]
                        # posnum = [i3["Amount"] for i3 in self._shelf_info if i3["PosID"] == i2["PosID"]][0]
                        tmp_replenish = {}
                        tmp_replenish["ShelfID"] = i1
                        tmp_replenish["Amount"] = i2["MaxNum"] - posnum
                        tmp_replenish["PosID"] = i2["PosID"]
                        tmp_replenish["ProductBatchID"] = productbatchid
                        tmp_replenish["StockType"] = stocktype
                        tmp_replenish["CommodityID"] = i2["CommodityID"]
                        # tmp_replenish["shouldbenum"] = tmp_replenish["Amount"] + posnum
                        tmp_replenish["Priority"] = 1
                        replenish_detail1.append(tmp_replenish)
        replenish_detail0.extend(replenish_detail1)
        tmplist = [i1["ShelfID"] for i1 in replenish_detail0]
        for i1 in tmplist:
            self._replenish_maxid += 1
            for i2 in replenish_detail0:
                if i1 == i2["ShelfID"]:
                    i2["ReplenishID"] = self._replenish_maxid
        return replenish_detail0

    # 策略更新
    def strategy_update(self, order_data, shelf_status):
        # 变更时只考虑未占用货架，且设置货架状态待变更3
        # 订单历史信息表（带时间参数，取之后左右的包含新单未处理的），货架类型表，商品信息表，储位sku可容纳数量映射表
        pass


if '__main__' == __name__:
    pass
