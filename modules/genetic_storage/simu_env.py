# @Time : 2019/4/22 10:54
# @Author : YingbinQiu
# @Site :
# @File : simu_env.py

import math
import time
import pandas as pd
import os

MINTIMEERROR = 1.e-3


class ShelfStatus(object):
    def __init__(self, injson):
        self.shelfid = injson["shelfid"]
        self.shelftypeid = injson["shelftypeid"]
        self.shelfstatus = injson["shelfstatus"]
        self.workerid = injson["workerid"]
        self.workertype = injson["workertype"]


class ShelfInfo(object):
    def __init__(self, injson):
        self.shelfid = injson["shelfid"]
        self.cellid = injson["cellid"]
        self.commodityid = injson["commodityid"]
        self.shouldbenum = injson["shouldbenum"]
        self.occupynum = injson["occupynum"]
        self.shortcommodity = injson["shortcommodity"]

    def change_cell_info(self, upjson):
        self.shouldbenum = upjson["shouldbenum"]
        self.occupynum = upjson["occupynum"]
        self.shortcommodity = upjson["shortcommodity"]
        self.commodityid = upjson["commodityid"]


class ShelfType(object):
    def __init__(self, injson):
        self.shelftypeid = injson["shelftypeid"]
        self.shelflength = injson["shelflength"]
        self.shelfwidth = injson["shelfwidth"]
        self.shelfhight = injson["shelfhight"]
        self.layernum = injson["layernum"]
        self.maxcell = injson["maxcell"]


class OrderInfo(object):
    def __init__(self, injson):
        self.orderid = injson["orderid"]
        self.commodityid = injson["urgent"]
        self.commodityid = injson["indate"]
        self.num = injson["num"]


class OrderDetail(object):
    def __init__(self, injson):
        self.orderid = injson["orderid"]
        self.commodityid = injson["commodityid"]
        self.num = injson["num"]


class SimuEnv(object):
    def __init__(self, conf, model_json):
        self._conf = conf
        self._model_json = model_json
        # 参数转内部
        self._cancel_random = self._conf["cancel_random"]
        self.simu_flag = self._model_json["simu_flag"]
        self.bench_maxnum = self._conf["bench_maxnum"]
        self.replenish_maxnum = self._conf["replenish_maxnum"]
        # shelf_list
        self.shelf_list = []
        # 分拣台可用数(订单因素)
        self.operater_line_max = 0
        # 虚拟现实时间戳
        self._virtual_current_stamp = 0
        # 虚拟现实时间
        self._virtual_current_time = 0
        # 虚拟订单
        self._virtual_current_order = 0
        # 虚拟货架信息
        self._virtual_current_shelf = 0
        # 虚拟工作台状态初始化
        self._virtual_bench = {}
        # self._virtual_bench_init()
        # 虚拟补货台状态初始化
        self._virtual_replenish = {}
        # self._virtual_replenish_init()

    # 初始化货架
    def _shelf_init(self, shelfjson):
        for i2 in shelfjson:
            self.shelf_list.append(ShelfInfo(i2))

    # 订单引擎
    def _order_generater(self):
        while True:
            a = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(self._virtual_order_start)))
            b = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(self._virtual_current_stamp)))
            print(a, b)
            newadd_orders = self._order_info[a:b]
            print(newadd_orders)
            newadd_orders = self._order_info.iloc[a:b, :]
            print(newadd_orders)
            self._virtual_calcu_order = newadd_orders
            yield True

    def _get_batch_length(self):
        return sum([len(self._virtual_bench[i1].batch_waiting) for i1 in self._virtual_bench if
                    0 != len(self._virtual_bench[i1].batch_waiting)]) / self._betch_use_num

    def _get_batch_doing_length(self):
        return sum([1 for i1 in self._virtual_bench if
                    0 != len(self._virtual_bench[i1].batch_mission)]) / self._betch_use_num

    def _get_replenish_length(self):
        return sum([len(self._virtual_replenish[i1].batch_waiting) for i1 in self._virtual_replenish if
                    0 != len(self._virtual_replenish[i1].batch_waiting)]) / self._replenish_use_num

    def _get_replenish_doing_length(self):
        return sum([1 for i1 in self._virtual_replenish if
                    0 != len(self._virtual_replenish[i1].batch_mission)]) / self._replenish_use_num

    # 尝试加入补货任务队列,并改变标记
    def _try_add_replenish(self, replenish_lists):
        lenth = self._get_replenish_length()
        if lenth > self._replenish_qeueu_up:
            self._replenish_use_num += 1
        elif lenth < self._replenish_qeueu_down:
            if lenth > 1:
                self._replenish_use_num -= 1
        # 2为正补货
        for betchid in self._virtual_replenish:
            for replenish1 in replenish_lists:
                # 更新缺货数量，补货阈值判断，更新状态。
                if self._virtual_replenish[betchid].status != 2 and self._virtual_replenish[betchid].shelfid == \
                        replenish1["shelfid"]:
                    for i1 in self.shelf_list:
                        if i1.shelfid == replenish1["shelfid"] and i1.cellid == replenish1["cellid"]:
                            i1.change_cell_info(replenish1)

    # 加入正在执行任务
    def _add2batch(self):
        for i1 in self._virtual_bench:
            if 0 == len(self._virtual_bench[i1].batch_mission):
                if 0 != len(self._virtual_bench[i1].batch_waiting):
                    tmpid = self._virtual_bench[i1].batch_waiting[0].batchid
                    self._virtual_bench[i1].batch_mission = [i2 for i2 in self._virtual_bench[i1].batch_waiting if
                                                             i2.batchid == tmpid]
                    self._virtual_bench[i1].batch_waiting = [i2 for i2 in self._virtual_bench[i1].batch_waiting if
                                                             i2.batchid != tmpid]

    # 加入正在执行任务
    def _add2replenish(self):
        for i1 in self._virtual_replenish:
            if 0 == len(self._virtual_replenish[i1].batch_mission):
                if 0 != len(self._virtual_replenish[i1].batch_waiting):
                    tmpid = self._virtual_replenish[i1].batch_waiting[0].batchid
                    self._virtual_replenish[i1].batch_mission = [i2 for i2 in self._virtual_replenish[i1].batch_waiting
                                                                 if i2.batchid == tmpid]
                    self._virtual_replenish[i1].batch_waiting = [i2 for i2 in self._virtual_replenish[i1].batch_waiting
                                                                 if i2.batchid != tmpid]

    # 策略(流程规则，环境互动 - 订单缓存表的动态变化，)
    def strategy_flow(self, indata, simustrategy, start=None, to=None, threadid=1):
        # 订单信息，取消订单信息，货架货物表
        self._read_interval = simustrategy._read_interval
        self._betch_qeueu_down = simustrategy._betch_qeueu_down
        self._betch_qeueu_up = simustrategy._betch_qeueu_up
        self._replenish_qeueu_down = simustrategy._replenish_qeueu_down
        self._replenish_qeueu_up = simustrategy._replenish_qeueu_up
        self._betch_use_num = 1
        self._replenish_use_num = 1
        # 数据转化
        self._order_info = indata["order_info"]
        self._order_detail = indata["order_detail"]
        self._commodity_info = indata["commodity_info"]
        self._shelf_info = indata["shelf_info"]
        # todo: 运货时间模拟
        # todo: 增加基本货物信息
        # self._shelf_init(self._shelf_info)
        # 新订单起时间
        self._virtual_order_start = 0
        # 虚拟计算订单
        self._virtual_calcu_order = None
        # 虚拟分好订单
        self._virtual_prepare_order = None
        # 虚拟分拣订单
        self._virtual_doing_order = None
        self._min_step_time = 0

        # 0. 获得虚拟时间戳
        timestamp_list = [time.mktime(time.strptime(i1, "%Y-%m-%d %H:%M:%S")) for i1 in list(self._order_info.index)]
        self._virtual_current_stamp = timestamp_list[0]
        # 结束时间
        finishtime = timestamp_list[-1]
        # finishtime = time.mktime(time.strptime("2019-03-17 19:50:30", "%Y-%m-%d %H:%M:%S"))
        print(finishtime)
        print(self._virtual_current_stamp)
        print(self._order_info.iloc[0:2])
        self._virtual_order_start = self._virtual_current_stamp
        calc_next = 1
        while True:
            if 1 == calc_next:
                print("add order range:")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(self._virtual_order_start))))
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(self._virtual_current_stamp))))
                self._order_generater()
                self._virtual_order_start = self._virtual_current_stamp
                # self._cancel_order_simu(time_range)
                time_start = time.time()
                # 输入self._virtual_calcu_order，货架信息已经扣除了占用量，返回波次订单信息，随机撤销订单
                simustrategy.pick_replenish_strategy()
                order_batchs, batchbetch_all, replenish_lists = simustrategy.strategy_policy(self._virtual_calcu_order,
                                                                                            self._virtual_current_shelf,
                                                                                            self._betch_use_num, )
                self._calc_time_delay = time.time() - time_start
                if self._calc_time_delay < self._read_interval:
                    self._timecalu_left = self._read_interval
                else:
                    self._timecalu_left = self._calc_time_delay
                # self._try_add_batch(order_batchs)
                self._try_add_replenish(replenish_lists)
                self._add2batch()
                self._add2replenish()
            # 更新系统时间, 根据触发时间进入下一个步骤
            calc_next = self._min_step_events()
            print("calc_next", calc_next)
            # 模拟是否结束，1. 订单始终无法执行 死循环，2. 虚拟时间大于历史订单的最后一个，且无预备执行波次 和 正在执行波次。
            if finishtime < self._virtual_current_stamp and 0 == self._get_batch_length() + self._get_batch_doing_length():
                print("finish condition.")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(finishtime))))
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(self._virtual_current_stamp))))
                print(self._get_batch_length(), self._get_batch_doing_length())
                simustrategy._betch_qeueu_down = self._betch_qeueu_down
                simustrategy._betch_qeueu_up = self._betch_qeueu_up
                simustrategy._replenish_qeueu_down = self._replenish_qeueu_down
                simustrategy._replenish_qeueu_up = self._replenish_qeueu_up
                print("统计各种信息：")
                # 多机版要改到数据库
                print("finishing！")
                simustrategy.dump_paras(
                    os.path.join(simustrategy._para_path, "gene_%s_%s.json" % (threadid, self.cost_fun())))
                return 0

    # 策略更新(流程规则，环境互动 - 订单缓存表的动态变化，)
    def strategy_update(self, indata, simustrategy, start=None, to=None):
        # 变更时只考虑未占用货架，且设置货架状态待变更3
        # 订单历史信息表（带时间参数，取之后左右的包含新单未处理的），货架类型表，商品信息表，储位sku可容纳数量映射表
        simustrategy.strategy_update()
        pass

    # 订单取消(随机模拟，正在执行的忽略，分好波次的删除，否则动态删除缓存)
    def _cancel_order_simu(self, time_range):
        # cancel_num = int(len(self._virtual_calcu_order) * self._cancel_random * time_range / 100)
        # label_index = random.sample(self._virtual_calcu_order, cancel_num)
        return True

    # 最小步长事件
    def _min_step_events(self):
        # 1. 对比最小事件时间，取最短。
        print("_timecalu_left")
        timelist = [self._timecalu_left]
        # autotimelist = []
        min_replenish = self._strategy_replenish_count()
        min_order_finish = self._strategy_order_finish_count()
        if min_replenish is not None:
            timelist.append(min_replenish)
        if min_order_finish is not None:
            timelist.append(min_order_finish)
        self._min_step_time = min(timelist)
        # 2. 按最小事件时间，更新虚拟时间及各时间的进程。
        print(self._timecalu_left)
        self._strategy_replenish_sync()
        print(self._timecalu_left)
        self._strategy_order_finish_sync()
        print(self._timecalu_left)
        real_action = self._calc_sync()
        self._virtual_current_stamp += self._min_step_time
        self._min_step_time = 0
        return real_action

    # 上架计时()
    def _strategy_replenish_count(self):
        # 遍历每一个最短事件，返回该值
        timlist = [self._virtual_replenish[i1].deal_time() for i1 in self._virtual_replenish if
                   0 != len(self._virtual_replenish[i1].batch_mission)]
        if 0 != len(timlist):
            return min(timlist)
        else:
            return None

    # 下架计时()
    def _strategy_order_finish_count(self):
        # 遍历每一个最短事件，返回该值
        timlist = [self._virtual_bench[i1].deal_time() for i1 in self._virtual_bench if
                   0 != len(self._virtual_bench[i1].batch_mission)]
        if 0 != len(timlist):
            return min(timlist)
        else:
            return None

    # 上架同步()
    def _calc_sync(self):
        print("_timecalu_left")
        print(self._timecalu_left)
        self._timecalu_left -= self._min_step_time
        print(self._timecalu_left)
        # 同步该任务时间
        if abs(self._timecalu_left) < MINTIMEERROR:
            return 1
        elif self._timecalu_left < -MINTIMEERROR:
            raise Exception("error in _calc_sync")
        else:
            return 0

    # 上架同步()
    def _strategy_replenish_sync(self):
        shelf_list = []
        for i1 in self._virtual_replenish:
            if abs(self._virtual_replenish[i1].time4finish_mission) < MINTIMEERROR:
                # 同步该任务时间
                self._virtual_replenish[i1].mission_finish()
                # 修改该时间完成任务的货架状态
                # shelf_list.append(shelfid)
            elif self._virtual_replenish[i1].time4finish_mission > self._min_step_time:
                self._virtual_replenish[i1].time4finish_mission -= self._min_step_time
            elif self._virtual_replenish[i1].time4finish_mission < self._min_step_time:
                print(self._virtual_replenish[i1].time4finish_mission, self._min_step_time)
                raise Exception("error in _strategy_replenish_sync.")
            else:
                pass
        return shelf_list

    # 下架同步()
    def _strategy_order_finish_sync(self):
        shelf_list = []
        for i1 in self._virtual_bench:
            if abs(self._virtual_bench[i1].time4finish_mission) < MINTIMEERROR:
                # 同步该任务时间
                self._virtual_bench[i1].mission_finish()
                # 修改该时间完成任务的货架状态
                # return shelfid
            elif self._virtual_bench[i1].time4finish_mission > self._min_step_time:
                self._virtual_bench[i1].time4finish_mission -= self._min_step_time
            elif self._virtual_bench[i1].time4finish_mission < self._min_step_time:
                print(self._virtual_bench[i1].time4finish_mission, self._min_step_time)
                raise Exception("error in _strategy_order_finish_sync")
            else:
                pass
        return shelf_list

    # 扫码补货通知(仅消息传递)
    def _replenish_simu(self):
        # 获取下一个补货任务
        return True

    # 扫码满箱通知(仅消息传递)
    def _order_finish_simu(self):
        # 获取下一个订单任务
        return True

    # 单轮成本(人工成本，面积成本，设备成本)
    def cost_fun(self):
        return "115"


if '__main__' == __name__:
    pass
