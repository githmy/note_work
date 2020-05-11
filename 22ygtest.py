"""
凌晨前运行程序 ，生成第二天的月嫂排班信息，不会预生成之后的，如果预定某月嫂对某订单，要走请假的流程单算。
如果未来月嫂数量不足，必须在招募计划表里设置一个计划招募数量，否则无法生成新订单。

公式：
订单价格 = 产品原价 × 产品折扣
实际工资 = 原价工资 × 工资折扣
销售进账 = 订单价格 + 月嫂使用费 + 订单保证金
结算支出 = 实际工资 + 产品原价 × 平台提成 + 订单保证金
          月嫂部分     创始人部分         返还部分

双头补金额 = 产品原价 × (1 - 产品折扣) + 原价工资 × (工资折扣 - 1)
当天订单售退金额 = 当天原价订单售退金额 - 当天双头补金额
当天平台内开销 = 当天订单售退金额 - 当天月嫂工资 - 当天创始人结算 + 当天平台通信费 - 当天平台开销
当天资金增量 = 当天平台内开销 - 平台借贷资金

待服务订单数 = 未到期订单数 + 月嫂的缺额
延期且未开订单数 = 月嫂的缺额
服务中订单数 = 服务中月嫂数
未开发订单数 = 未利用月嫂数

月嫂预留数 = 服务期内订单数 × 预留百分数

问题:
优先分配 延时加长的订单 还是正常订单？
服务期不确定性大，就要预留更多的月嫂，否则延时会冲掉后期的新增订单，引起资金累积震荡，增长速度很慢
服务期

本次修改:
增加了资金显示的修改
增加了变更日期的规则

待做:
1. 得出统计参数
2. 基于统计参数 得出 规则常数
"""
import os
import itertools
import cv2
import numpy as np
import pytesseract
from PIL import Image
import csv
import re
import json
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib as mpl
import datetime, time
import copy
import random

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False


def plot_curve(x, ys, titles):
    yins = [np.array(y) for y in ys]
    xin = np.arange(0, len(ys[0]))
    nums = len(ys)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (nums // 7 + 1)
    # 长 宽 背景颜色
    plt.figure(figsize=(12, 6), facecolor='w')
    # plt.figure(facecolor='w')
    for n in range(nums):
        plt.plot(xin, yins[n], color=colors[n], linestyle='-', linewidth=1.2, marker="", markersize=7,
                 markerfacecolor='b', markeredgecolor='g', label=titles[n])
        plt.legend(loc='upper right', frameon=False)
    # plt.plot(xin, yin, color='r', linestyle='-', linewidth=1.2, marker="*", markersize=7, markerfacecolor='b',
    #          markeredgecolor='g')
    plt.xlabel("x", verticalalignment="top")
    plt.ylabel("y", rotation=0, horizontalalignment="right")
    # xticks = ["今天", "周五", "周六", "周日", "周一"]
    # show_inte = 30
    show_inte = 7
    s_xin = [i1 for i1 in xin if i1 % show_inte == 0]
    s_x = [i1 for id1, i1 in enumerate(x) if id1 % show_inte == 0]
    plt.xticks(s_xin, s_x, rotation=90, fontsize=10)
    # plt.xticks(xin, x, rotation=90, fontsize=5)
    # yticks = np.arange(0, 500, 10)
    # plt.yticks(yticks)
    # plt.title(title)
    # plt.grid(b=True)
    # plt.savefig('../cap.png')
    plt.show()


def int2date(num):
    return datetime.datetime.now() + datetime.timedelta(days=num)
    # return datetime.datetime.now() + datetime.timedelta(days=num)).strftime("%Y-%m-%d %H:%M:%S")


def datediff(datestr1, datestr2):
    t1 = time.mktime(time.strptime(datestr1, "%Y-%m-%d"))
    t2 = time.mktime(time.strptime(datestr2, "%Y-%m-%d"))
    return int(round((t2 - t1) / 86400))


def date2green(datet):
    return time.mktime(datet.timetuple())


def str2green(datestr):
    return time.mktime(time.strptime(datestr, "%Y-%m-%d"))


def str_num2datestr(datestr, num):
    tst = (datetime.datetime.strptime(datestr, "%Y-%m-%d") + datetime.timedelta(days=num)).timetuple()
    return time.strftime("%Y-%m-%d", tst)


class FutureShow(object):
    # todo: 星级拆分
    def __init__(self):
        """已知:可查询数据如下"""
        self.datajson = {
            "月嫂日增期望": 10,  # todo: 统计 星级 地域 当前工资 报名加入日期(含培训期) 当天工资分享数量（可得2周内）
            "月嫂日增标准差": 3,  # todo: 统计 星级 地域 当前工资 报名加入日期(含培训期) 当天工资分享数量（可得2周内）
            # "月嫂初始数": 2000,
            "月嫂初始数": 200,
            "月嫂预留百分数": 0.1,  # 未排班的订单为基数
            "订单销售速度期望": 10000,  # 昨天的倍数 1 相当于持平  # todo: 统计 订单详情(星级 地域 服务等待期 当天订单价格 预售期日期 等待期日期) 订单销售速度 销售排行状态
            "订单销售速度标准差": 0,  # 浮动百分数         # todo: 统计 订单详情(星级 地域 服务等待期 当天订单价格 预售期日期 等待期日期) 订单销售速度 销售排行状态
            "原价订单": 8000,
            # "订单折扣": 0.7,
            "订单折扣": 1,
            "订单保证金": 500,
            "等待期": 60,
            "服务期": 28,
            "结算期": 7,
            "原价工资": 5000,
            # "工资折扣": 1.2,
            "工资折扣": 1,
            "月嫂使用费": 500,
            "订单提成": 0.1,
            "总时长": 365,
            # "总时长": 180,
            "初始资金": 0,
            "初始借贷": 0,
            "平台通讯费": 0,
            "平台开销": 0,
            # "日期偏移标准差": 0,
            # "日期偏移期望": 0,
            "日期偏移标准差": 4,  # todo: 统计订单的 初始日期 和 最后落实日期
            "日期偏移期望": -3,  # todo: 统计订单的 初始日期 和 最后落实日期
            # "日期偏移标准差": 5,
            # "日期偏移期望": -4,
            "活期利率": 0.02,
            "服务取消费率": 3,  # todo: 给客户3倍的活期利率 默认不走此流程，所以模拟过程不使用该项
            "禁止变更提前期": 7,  # 小于7天不允许
            "变更服务费率": [
                [[-3, 3], 0],
                [[-30, -4], 0.2],
                [[4, 60], 0.2],
                [[61, 180], 0.25],
            ],
            "服务逾期补偿": [0.25, 3],  # 给客户 3倍的活期利率 或 25%延时服务
        }
        self.datajson["订单价格"] = self.datajson["原价订单"] * self.datajson["订单折扣"]
        self.datajson["实际工资"] = self.datajson["原价工资"] * self.datajson["工资折扣"]
        self.order_today_alllist = []
        self.sao_tomorrow_freelist = []

    def gene_fakeori_date(self):
        # 1. 月嫂id表
        self.sao_nowlist = [str(i1) for i1 in range(self.datajson["月嫂初始数"])]
        # 2. 月嫂计划增量表，日期相同后自动删除记录
        # self.sao_today_addlist = [{"add_date": "2020-05-17", "num": 2}]
        self.sao_today_addlist = []
        # 3. 月嫂请假表
        # self.sao_vacation_list = [{"saoid": "3", "start": "2020-06-07", "end": "2099-06-07"}]
        self.sao_vacation_list = []
        # 4. 服务未关闭，订单列表，包含 待服务 服务中 待结算"status": "servicing,waiting,calc"
        self.order_pass_list = []
        # 5. 资金借贷列表
        self.borrow_list = [
            ["2020-06-18", 10000],
            ["2020-07-18", -5000],
            ["2020-07-18", -10000],
        ]

    def get_employ_std(self, ordertype):
        """统计 星级 地域 当前工资 报名加入日期(含培训期) 当天工资分享数量（可得2周内）"""
        # todo: 函数参数待统计
        employ_expect = self.datajson["月嫂日增期望"]
        employ_std = self.datajson["月嫂日增标准差"]
        return employ_expect, employ_std

    def get_sale_std(self, ordertype):
        """统计 订单详情(星级 地域 服务等待期 当天订单价格 预售期日期 等待期日期) 订单销售速度 销售排行状态"""
        # todo: 函数参数待统计
        sale_expect = self.datajson["订单销售速度期望"]
        sale_std = self.datajson["订单销售速度标准差"]
        return sale_expect, sale_std

    def get_service_shift_std(self, ordertype):
        """统计订单的 初始日期 和 最后落实日期"""
        # todo: 函数参数待统计
        service_shift_expect = self.datajson["日期偏移期望"]
        service_shift_std = self.datajson["日期偏移标准差"]
        return service_shift_expect, service_shift_std

    def get_day_borrow(self, dateid):
        datestr = int2date(dateid).strftime("%Y-%m-%d")
        borrow_num = 0
        for bn in self.borrow_list:
            if bn[0] == datestr:
                borrow_num += bn[1]
        return borrow_num

    def get_day_sao_list(self, today_date):
        # todo: 一次性生成字典，随时取
        # 1. 每天请假月嫂列表
        dellist = [i1["saoid"] for i1 in self.sao_vacation_list if self.sao_vacation_list if
                   today_date >= i1["start"] and today_date <= i1["end"]]
        # print("dellist", dellist)
        # 2. 每天的预增月嫂
        # sao_addnum = sum([i1["num"] for i1 in self.sao_today_addlist if today_date >= i1["add_date"]])
        employ_expect, employ_std = self.get_employ_std(ordertype=None)
        diffdate = datediff(self.today_date, today_date)
        sao_addnum = int(round(sum(np.random.normal(loc=employ_expect, scale=employ_std, size=diffdate))))
        # print("addnum", sao_addnum)
        if sao_addnum > 0:
            sao_add_list = ["{}_{}".format(today_date, i1) for i1 in range(sao_addnum)]
        else:
            dellist += random.sample(self.sao_nowlist, -sao_addnum)
            sao_add_list = []
        # 3. 每天可用的月嫂列表
        sao_today_totallist = []
        add_length = len(sao_add_list)
        counter = 0
        # 清空时 先用新增的补缺额
        # for i1 in self.sao_nowlist:
        #     if i1 not in dellist:
        #         sao_today_totallist.append(i1)
        #     else:
        #         for i2 in self.order_today_alllist:
        #             if i2["saoid"] == i1:
        #                 if counter < add_length:
        #                     i2["saoid"] = sao_add_list[counter]
        #                     counter += 1
        #                 else:
        #                     i2["saoid"] = ""
        #                 break
        sao_today_totallist = [i1 for i1 in self.sao_nowlist if i1 not in dellist]
        sao_today_totallist += sao_add_list
        return sao_today_totallist

    def every_update_infor(self):
        emptylenth = len(self.sao_tomorrow_freelist)
        destrib_counter = 0
        for order in self.order_today_alllist:
            if order["calcdate"] < self.tomorrow_str:
                # 如果明天超时，结算
                self.cap_num -= self.datajson["实际工资"] + \
                                self.datajson["原价订单"] * self.datajson["订单提成"] + self.datajson["订单保证金"]
                self.subsidy_num += self.datajson["原价工资"] * (self.datajson["工资折扣"] - 1)
                self.salary_num += self.datajson["实际工资"]
                self.creater_num += self.datajson["原价订单"] * self.datajson["订单提成"] + self.datajson["订单保证金"]
                order["status"] = "done"
            elif order["end"] < self.tomorrow_str:
                order["status"] = "calc"
            elif order["start"] > self.tomorrow_str:
                order["status"] = "waiting"
            else:
                # 本应 servicing 的订单
                if order["status"] == "waiting":
                    # 明天是服务期，请假了重分配 未分配则分配
                    if order["saoid"] == "":
                        # # 第一天只修改月嫂分配状态，否则
                        # if 0 == datediff(order["start"], self.tomorrow_str):
                        destrib_counter += 1
                        if destrib_counter > emptylenth:
                            # print(destrib_counter, emptylenth)
                            # 月嫂余量不足 一直变到 有月嫂
                            order["start"] = self.tomorrow_str
                            order["end"] = self.future_addend_str
                            order["calc"] = self.future_addcalc_str
                            order["status"] = "waiting"
                            destrib_counter -= 1
                        else:
                            order["saoid"] = self.sao_tomorrow_freelist[destrib_counter - 1]
                            order["status"] = "servicing"
                    else:
                        order["status"] = "servicing"
                else:
                    order["status"] = "servicing"
        # 清空过期的订单列表
        # print()
        self.order_today_alllist = [orderold for orderold in self.order_today_alllist if orderold["status"] != "done"]
        return self.order_today_alllist

    def every_new_orders(self, dateid):
        """创建 新订单： 预留月嫂，受销售数量上限"""
        #  待服务 服务中 待结算 : "servicing,waiting,calc,done"
        # 未来空闲月嫂数 可能已被占用 只是未分配，不能简单求空闲的长度 作为新增的长度
        # 1. 预留月嫂数 sao_future_free_lenth<len(sao_future_freelist)
        sao_future_free_lenth = len(self.sao_future_totallist) - int(
            len(self.sao_future_servicinglist) * (1 + self.datajson["月嫂预留百分数"]))
        # sao_future_free_lenth = int(len(self.sao_future_totallist) * (1 - self.datajson["月嫂预留百分数"])) - len(
        #     self.sao_future_servicinglist)
        sale_expect, sale_std = self.get_sale_std(ordertype=None)
        sale_num = int(round(np.random.normal(loc=sale_expect, scale=sale_std, size=1)[0]))
        sale_final = sale_num if sale_num < sao_future_free_lenth else sao_future_free_lenth
        # 2. 预留
        self.order_today_sale_list = []
        if sale_final > 0:
            # 6.1. 直接模拟卖出了 未来等待期的 原始订单。未来变更，直接现在修改
            service_shift_expect, service_shift_std = self.get_service_shift_std(ordertype=None)
            randshifts = np.random.normal(loc=service_shift_expect, scale=service_shift_std, size=sale_final)
            shiftsstart = [int(round(rn) + self.futurestart_int) for rn in randshifts]
            shiftsstart = [rn if rn > 0 else 1 for rn in shiftsstart]
            shiftsend = [rn + self.datajson["服务期"] for rn in shiftsstart]
            shiftscalc = [rn + self.datajson["结算期"] for rn in shiftsend]
            self.order_today_sale_list = [{"orderid": saoid + "_" + self.tomorrow_str,  # saoid 只为防止重复占位，跟绑定月嫂无关。
                                           "waitdate": self.tomorrow_str,
                                           "oristart": self.futurestart_str,
                                           "start": int2date(shiftsstart[idn]).strftime("%Y-%m-%d"),
                                           "end": int2date(shiftsend[idn]).strftime("%Y-%m-%d"),
                                           "calcdate": int2date(shiftscalc[idn]).strftime("%Y-%m-%d"),
                                           "saoid": "",
                                           "status": "waiting"} for idn, saoid in enumerate(self.sao_future_freelist)
                                          if idn < sale_final]
            # 6.2. 模拟原始售出资金变动，未来变更，在排班服务时修正
            self.cap_num += (self.datajson["订单价格"] + self.datajson["月嫂使用费"] + self.datajson[
                "订单保证金"]) * sale_final
            self.subsidy_num += (self.datajson["原价订单"] * (1 - self.datajson["订单折扣"])) * sale_final
            self.creater_num -= self.datajson["订单保证金"] * sale_final
        return self.order_today_sale_list

    def every_status_orders(self):
        """更新 订单 服务状态，加入 变期 订单费用"""
        for order in self.order_today_alllist:
            # 当天服务中的 订单和对应月嫂。如果有月嫂 突然退出 需要外部程序清空该订单的月嫂id更改状态为 等待期 或其他约定字段
            if order["status"] == "servicing":
                self.order_today_servicinglist.append(order["orderid"])
                self.sao_today_servicinglist.append(order["saoid"])
            elif order["status"] == "waiting":
                # if today_date>order["start"]:
                self.order_today_waitinglist.append(order["orderid"])
                # 明天为已排班 切为起始日，切 变更过起始日期，则修改资金收入
                if order["start"] == self.tomorrow_str and order["start"] != order["oristart"] and order["saoid"] != "":
                    diffdays = datediff(order["oristart"], order["start"])
                    for fee in self.datajson["变更服务费率"]:
                        if diffdays <= fee[0][1] and diffdays >= fee[0][0]:
                            self.cap_num += fee[1] * self.datajson["订单价格"]
                            break
                if order["start"] <= self.today_date and order["end"] >= self.today_date and order["saoid"] == "":
                    # if order["saoid"] == "":
                    self.order_today_delaylist.append(order["orderid"])
            # 明天已排班的数量 即使未分配，也占一个位置
            if order["start"] <= self.tomorrow_str and order["end"] >= self.tomorrow_str:
                self.sao_tomorrow_servicinglist.append(order["saoid"])
            # 新订单已排班的数量 即使未分配，也占一个位置
            if order["start"] <= self.futurestart_str and order["end"] >= self.futurestart_str:
                self.sao_future_servicinglist.append(order["saoid"])
        # print("self.order_today_servicinglist", len(self.order_today_servicinglist))
        # print("self.sao_today_servicinglist", len(self.sao_today_servicinglist))
        # print("self.order_today_waitinglist", len(self.order_today_waitinglist))
        return self.order_today_alllist

    def gene_full_service(self):
        self.y_capital = []  # 资金池 历史累计余额
        self.y_subsidy = []  # 双头补 历史累计余额
        self.y_borrow = []  # 借贷资金 历史累计余额 正为借出 负为贷入
        self.y_salary = []  # 原价工资累积
        self.y_creater = []  # 创始人收益累积
        self.y_capital_change = []  # 每日资金变化
        self.y_subsidy_change = []  # 双头补资金变化
        self.y_borrow_change = []  # 借贷资金变化
        self.y_salary_change = []  # 原价工资变化
        self.y_creater_change = []  # 创始人收益变化
        self.y_sao_total = []  # 月嫂总量 是 订单总容量 的上限
        self.y_sao_servicing = []  # 服务中月嫂数 = 服务中订单数
        self.y_sao_short = []  # 延期且未开订单数 = 月嫂的缺额
        self.y_sao_free = []  # 未利用月嫂数 = 未开发订单数
        self.y_order_total = []  # 订单总容量 包含待服务的 不含未开发的，上限为 当天月嫂总量
        self.y_order_waiting = []  # 待服务订单数 = 未到期订单数 + 月嫂的缺额
        self.y_order_delay = []  # 延期且未开订单数 = 月嫂的缺额
        self.y_order_servicing = []  # 服务中订单数 = 服务中月嫂数
        self.y_order_free = []  # 未开发订单数 = 未利用月嫂数
        self.x = [i1 for i1 in range(self.datajson["总时长"])]
        self.x_label = [int2date(i1).strftime("%Y-%m-%d") for i1 in range(self.datajson["总时长"])]
        saoid_list = [i1 for i1 in range(self.datajson["月嫂初始数"])]
        addday = int(round(self.datajson["服务逾期补偿"][0] * self.datajson["等待期"]))
        # 根据月嫂数生成对应的订单，初始化时 是否利用等待期的空置月嫂？ 尽量短的
        # 4. 服务未关闭，订单列表，包含 待服务 服务中 待结算"status": "servicing,waiting,calc,done"
        self.order_today_alllist = copy.deepcopy(self.order_pass_list)
        for dateid, today_date in enumerate(self.x_label):
            print("dateid:", dateid)
            self.today_date = today_date
            self.futurestart_int = dateid + self.datajson["等待期"]
            self.futureend_int = self.futurestart_int + self.datajson["服务期"]
            self.futurecalc_int = self.futureend_int + self.datajson["结算期"]
            self.tomorrow_str = int2date(dateid + 1).strftime("%Y-%m-%d")
            self.futurestart_str = int2date(self.futurestart_int).strftime("%Y-%m-%d")
            self.futureend_str = int2date(self.futureend_int).strftime("%Y-%m-%d")
            self.futurecalc_str = int2date(self.futurecalc_int).strftime("%Y-%m-%d")
            self.future_addend_str = int2date(self.futureend_int + addday).strftime("%Y-%m-%d")
            self.future_addcalc_str = int2date(self.futurecalc_int + addday).strftime("%Y-%m-%d")
            self.cap_num = 0  # 资金初始
            self.subsidy_num = 0  # 双头补 分 订单和工资两部分
            self.salary_num = 0  # 工资
            self.creater_num = 0  # 创始人结算
            # 1. 每天可用的月嫂列表=已排班的+未排班的
            sao_today_totallist = self.get_day_sao_list(today_date)
            sao_tomorrow_totallist = self.get_day_sao_list(self.tomorrow_str)
            self.sao_future_totallist = self.get_day_sao_list(self.futurestart_str)
            # 2. 每天 服务未完成，已排服务列表 = self.sao_today_servicinglist
            self.order_today_servicinglist = []
            self.order_today_delaylist = []
            self.order_today_waitinglist = []
            # 3. 月嫂已排班记录 = 已排服务列表, 每天 明天和未来的 因为可能有月嫂缺额，未必相等
            self.sao_today_servicinglist = []
            self.sao_tomorrow_servicinglist = []
            self.sao_future_servicinglist = []  # 未来待服务列表 = 对应区间的月嫂数 即 self.order_future_waitinglist
            # 4. 每天 未排班服务列表
            # 最近空闲列表，未分配 但可能已占位，可用长度要用月嫂总数减去分配长度
            self.sao_tomorrow_freelist = []
            self.sao_future_freelist = []
            # 生成订单统计数据
            self.order_today_alllist = self.every_status_orders()
            order_today_length = len(self.order_today_alllist)
            sao_today_freelist = [saoid for saoid in sao_today_totallist if saoid not in self.sao_today_servicinglist]
            self.sao_tomorrow_freelist = [saoid for saoid in sao_tomorrow_totallist if
                                          saoid not in self.sao_tomorrow_servicinglist]
            self.sao_future_freelist = [saoid for saoid in self.sao_future_totallist if
                                        saoid not in self.sao_future_servicinglist]
            # 6. 每天 空闲服务列表 直接分配 按日期生成订单号,
            self.order_today_sale_list = self.every_new_orders(dateid)
            # 7. 更新 跟新订单合并
            self.order_today_alllist += self.order_today_sale_list
            # 8. 更新明天每单的状态, 同时 输出排班信息，结算
            self.order_today_alllist = self.every_update_infor()
            # 9. 更新 月嫂池 订单池 和资金池
            self.cap_num += self.datajson["平台通讯费"]
            self.cap_num -= self.datajson["平台开销"]
            borrow_num = self.get_day_borrow(dateid)
            self.cap_num -= borrow_num
            self.y_capital_change.append(self.cap_num)  # 每日资金变化
            self.y_subsidy_change.append(self.subsidy_num)  # 原价工资
            self.y_borrow_change.append(borrow_num)  # 借贷资金变化
            self.y_salary_change.append(self.salary_num)  # 原价工资变化
            self.y_creater_change.append(self.creater_num)  # 创始人收益变化
            self.y_sao_free.append(len(sao_today_freelist))  # 未利用月嫂数 = 未开发订单数
            self.y_sao_short.append(len(self.order_today_delaylist))  # 月嫂缺额
            self.y_sao_total.append(len(sao_today_totallist))  # 月嫂总量 是 订单总容量 的上限
            self.y_sao_servicing.append(len(self.sao_today_servicinglist))  # 服务中月嫂数 = 服务中订单数
            self.y_order_total.append(order_today_length)  # 订单总容量 包含待服务的 不含未开发的，上限为 当天月嫂总量
            self.y_order_waiting.append(len(self.order_today_waitinglist))  # 待服务订单数
            self.y_order_delay.append(len(self.order_today_delaylist))  # 延迟订单数
            self.y_order_servicing.append(len(self.order_today_servicinglist))  # 服务中订单数 = 服务中月嫂数
            self.y_order_free.append(len(sao_today_freelist))  # 未开发订单数 = 未利用月嫂数
            # print(len(self.order_today_alllist))
            # print(len(self.order_today_waitinglist))
            # print(len(self.order_today_servicinglist))
            # print(len(sao_today_freelist))
        # 9. 累计值生成
        for sn, cap in enumerate(self.y_capital_change):  # 资金池 历史累计余额
            if sn == 0:
                self.y_capital.append(self.datajson["初始资金"] + cap)
                self.y_subsidy.append(self.y_subsidy_change[0])  # 原价工资
                self.y_borrow.append(self.datajson["初始借贷"] + self.y_borrow_change[0])  # 借贷累计
                self.y_salary.append(self.y_salary_change[0])  # 原价工资
                self.y_creater.append(self.y_creater_change[0])  # 创始人收益
            else:
                self.y_capital.append(self.y_capital[-1] + cap)
                self.y_subsidy.append(self.y_subsidy[-1] + self.y_subsidy_change[sn])  # 双头补金额
                self.y_borrow.append(self.y_borrow[-1] + self.y_borrow_change[sn])  # 借贷累计
                self.y_salary.append(self.y_salary[-1] + self.y_salary_change[sn])  # 原价工资
                self.y_creater.append(self.y_creater[-1] + self.y_creater_change[sn])  # 创始人收益

    def capital2orders(self, capital_num):
        # todo: 流程更改
        # 1. 根据资金，查找实现组合
        # 2. 每种组合 在预期条件下的实现概率
        # 3. 选择合理的订单
        pass


def main():
    # 1. 默认加载原数据
    fs_ins = FutureShow()
    stime = time.time()
    # 2. 生成数据
    fs_ins.gene_fakeori_date()
    # 3. 算结果
    fs_ins.gene_full_service()
    print("use time is {}s".format(time.time() - stime))
    # 4. 绘图
    # 每日资金 每日月嫂
    # print(fs_ins.y_capital_change)
    # print(fs_ins.y_salary_change)
    # print(fs_ins.y_creater_change)
    titles = ["资金池增量", "双头补金额增量", "借贷增量", "工资增量", "创始人收益增量"]
    ys = [fs_ins.y_capital_change, fs_ins.y_subsidy_change, fs_ins.y_borrow_change, fs_ins.y_salary_change,
          fs_ins.y_creater_change]
    plot_curve(fs_ins.x_label, ys, titles)
    # print(fs_ins.y_capital)
    # print(fs_ins.y_salary)
    # print(fs_ins.y_creater)
    titles = ["资金池累计", "双头补金额累积", "借贷累计", "工资累积", "创始人收益累积"]
    ys = [fs_ins.y_capital, fs_ins.y_subsidy, fs_ins.y_borrow, fs_ins.y_salary, fs_ins.y_creater]
    plot_curve(fs_ins.x_label, ys, titles)
    # print(fs_ins.y_order_total)
    # print(fs_ins.y_order_waiting)
    # print(fs_ins.y_order_servicing)
    # print(fs_ins.y_order_free)
    # # ind = [i1 for i1 in range(len(fs_ins.y_order_free))]
    # # print([i1 for i1 in zip(ind, fs_ins.y_order_free)])
    titles = ["订单总容量", "待服务订单数", "服务中订单数", "延迟订单数", "未开发订单数"]
    ys = [fs_ins.y_order_total, fs_ins.y_order_waiting, fs_ins.y_order_servicing, fs_ins.y_order_delay,
          fs_ins.y_order_free]
    plot_curve(fs_ins.x_label, ys, titles)
    # print(fs_ins.y_sao_total)
    # print(fs_ins.y_sao_servicing)
    # print(fs_ins.y_sao_free)
    titles = ["月嫂总量", "服务中月嫂数", "月嫂缺额数", "未利用月嫂数"]
    ys = [fs_ins.y_sao_total, fs_ins.y_sao_servicing, fs_ins.y_sao_short, fs_ins.y_sao_free]
    plot_curve(fs_ins.x_label, ys, titles)
    exit()


if __name__ == '__main__':
    np.random.seed(5)
    main()
