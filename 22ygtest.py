"""
凌晨前运行程序 ，生成第二天的月嫂排班信息，不会预生成之后的，如果预定某月嫂对某订单，要走请假的流程单算。
如果未来一个等待期后月嫂数量不足，必须在招募计划表里设置一个计划招募数量，否则无法生成新订单。

公式：
订单价格 = 产品原价 × 产品折扣
实际工资 = 原价工资 × 工资折扣
销售进账 = 订单价格 + 月嫂使用费 + 订单保证金
结算支出 = 实际工资 + 产品原价 × 平台提成 + 订单保证金
          月嫂部分   创始人部分           返还部分

双头补金额 = 产品原价 × (1 - 产品折扣) + 原价工资 × (工资折扣 - 1)
当天订单售退金额 = 当天原价订单售退金额 - 当天双头补金额
当天平台内开销 = 当天订单售退金额 - 当天月嫂工资 - 当天创始人结算 + 当天平台通信费 - 当天平台开销
当天资金增量 = 当天平台内开销 - 平台借贷资金

待服务订单数 = 未到期订单数 + 月嫂的缺额
延期且未开订单数 = 月嫂的缺额
服务中订单数 = 服务中月嫂数
未开发订单数 = 未利用月嫂数

月嫂预留数 = 服务期内订单数 × 预留百分数

规则
分配月嫂时，起始日期越早的，优先安排
排在最后的订单，如果平台预留的月嫂不足，可能导致其频繁中断，无限延期服务，这种情况是否可以间断累计25%的增值服务？
取消订单返给客户： 订单价格
取消订单返给创始人：无提成
取消订单撤回双头补的订单部分：原价订单 × (1 - 订单折扣)

问题:
服务期不确定性大，就要预留更多的月嫂，否则延时会冲掉后期的新增订单，引起资金累积震荡，增长速度很慢。
25% 的延时服务，会放大不确定性对资金池的影响，需要预留更多的月嫂数，才比较安全。所以是否应该采用折现方式？
优先分配 延时加长的订单 还是正常订单？
集中销售会导致集中逾期，不利于月嫂的低预留策略。
不论售卖订单或招募月嫂，越平均，对月嫂的利用率越高，也就是可以预留更少的月嫂作为储备。

30 50 20
每次1-2个月
100

本次修改:
大客户存在日期变动，但不存在撤单和违约金
延时的订单，全为散单
借贷只影响散单部分
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
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False


def bar3dplot(data):
    m = data
    X = np.array(range(len(m[0])))
    Y = np.array(range(len(m[1])))
    z = m[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 此处因为是要和其他的图像一起展示，用的add_subplot，如果只是展示一幅图的话，可以用subplot即可

    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    x, y = xx.ravel(), yy.ravel()  # 矩阵扁平化

    # 更改柱形图的颜色，这里没有导入第四维信息，可以用z来表示了
    xlength = len(X)
    ylength = len(Y)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (ylength // 7 + 1)
    C = [[colors[i1]] * xlength for i1 in range(ylength)]
    C = list(itertools.chain(*C))
    # C = []
    # for a in range(z):
    #     if a < 10:
    #         C.append('b')
    #     elif a < 20:
    #         C.append('c')
    #     elif a < 30:
    #         C.append('m')
    #     elif a < 40:
    #         C.append('pink')
    #     elif a > 39:
    #         C.append('r')

    # 此处dx，dy，dz是决定在3D柱形图中的柱形的长宽高三个变量
    dx = 0.6 * np.ones(len(x))
    dy = 0.2 * np.ones(len(y))
    dz = z
    z = np.zeros_like(z)

    # 设置三个维度的标签
    ax.set_xlabel('Xlabel')
    ax.set_ylabel('Ylabel')
    ax.set_zlabel('Amplitude')

    show_inte = 30
    s_xin = [i1 for i1 in X if i1 % show_inte == 0]
    s_x = [i1 for id1, i1 in enumerate(m[0]) if id1 % show_inte == 0]
    plt.xticks(s_xin, s_x, rotation=90, fontsize=10)
    plt.yticks(Y, m[1])

    ax.bar3d(x, y, z, dx, dy, dz, color=C, zsort='average', shade=True)
    plt.show()


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
            "月嫂日增期望值": 1,  # todo: 统计 星级 地域 当前工资 报名加入日期(含培训期) 当天工资分享数量（可得2周内）
            "月嫂日增标准差": 1,  # todo: 统计 星级 地域 当前工资 报名加入日期(含培训期) 当天工资分享数量（可得2周内）
            "月嫂专职数": 2000,
            "月嫂精英数": 20,
            "大客户订单百分数": 0.2,
            "月嫂预留百分数": 0.2,  # 未排班的订单为基数
            "订单销售速度期望值": 10000,  # todo: 统计 订单详情(星级 地域 服务等待期 当天订单价格 预售期日期 等待期日期) 订单销售速度 销售排行状态
            "订单销售速度标准差": 0,  # todo: 统计 订单详情(星级 地域 服务等待期 当天订单价格 预售期日期 等待期日期) 订单销售速度 销售排行状态
            "原价订单": 10000,
            "订单折扣": 0.7,
            # "订单折扣": 1,
            "订单保证金": 500,
            "等待期": 60,
            "服务期": 28,
            "结算期": 7,
            "原价工资": 10000,
            "工资折扣": 1.5,
            # "工资折扣": 1,
            "月嫂使用费": 500,
            "订单提成": 0.1,
            "总时长": 365,
            # "总时长": 180,
            "初始资金": 0,
            "初始大客户资金": 0,
            "初始散客资金": 0,
            "初始借贷": 0,
            "初始双头补资金": 0,
            "初始单头补资金": 0,
            "初始发放工资": 0,
            "初始创始人资金": 0,
            "平台通讯费": 0,
            "平台开销": 0,
            # "日期偏移标准差": 0,
            # "日期偏移期望值": 0,
            "日期偏移标准差": 2,  # todo: 统计订单的 初始日期 和 最后落实日期
            "日期偏移期望值": -1,  # todo: 统计订单的 初始日期 和 最后落实日期
            # "日期偏移标准差": 5,
            # "日期偏移期望值": -4,
            "活期利率": 0.02,
            "服务取消费率": 0,  # todo: 给客户0倍的活期利率
            "禁止变更提前期": 7,  # 小于7天不允许
            "变更服务费率": [
                [[-30, -4], 0.2],
                [[-3, 3], 0],
                [[4, 60], 0.2],
                [[61, 180], 0.25],
            ],
            "服务逾期补偿": [0.25, 3],  # 给客户 3倍的活期利率 或 25%延时服务
        }
        print("初始参数：{}".format(self.datajson))
        # self.datajson["订单价格"] = self.datajson["原价订单"] * self.datajson["订单折扣"]
        # self.datajson["实际工资"] = self.datajson["原价工资"] * self.datajson["工资折扣"]
        # print("订单价格 = 原价订单 × 订单折扣")
        # print("{} = {} × {}".format(self.datajson["订单价格"], self.datajson["原价订单"], self.datajson["订单折扣"]))
        # print("实际工资 = 原价工资 × 工资折扣")
        # print("{} = {} × {}".format(self.datajson["实际工资"], self.datajson["原价工资"], self.datajson["工资折扣"]))

    def gene_fakeori_date(self):
        # 1. 当天的月嫂数，如果月嫂在服务期，没有备用月嫂不能减少。即消单才退。
        self.vocation_list = [
            ["2020-06-08", 2],
            ["2020-07-05", 5],
            ["2020-08-18", 7],
        ]
        print("模拟月嫂未来请假列表：{}".format(self.vocation_list))
        employ_expect, employ_std = self.get_employ_std(ordertype=None)
        sao_addlist = np.random.normal(loc=employ_expect, scale=employ_std,
                                       size=self.datajson["总时长"] + self.datajson["等待期"] + self.datajson["服务期"])
        self.sao_day_nums = [self.datajson["月嫂专职数"] + self.datajson["月嫂精英数"]]
        print("月嫂初始数量 = 月嫂专职数 + 月嫂精英数")
        print("{} = {} + {}".format(self.sao_day_nums[0], self.datajson["月嫂专职数"], self.datajson["月嫂精英数"]))
        for i1 in sao_addlist:
            self.sao_day_nums.append(self.sao_day_nums[-1] + int(round(i1)))
        for i1 in self.vocation_list:
            tn = datediff(int2date(0).strftime("%Y-%m-%d"), i1[0])
            self.sao_day_nums[tn] -= i1[1]
        # 2. 折扣变动
        nday = 90
        change_every = (1 - self.datajson["订单折扣"]) / nday
        self.order_small_discount_day = [1 for _ in range(self.datajson["总时长"])]
        self.order_small_discount_day[0:nday] = [self.datajson["订单折扣"] + i1 * change_every for i1 in range(nday)]
        self.order_small_price_day = np.array(self.order_small_discount_day) * self.datajson["原价订单"]
        self.order_big_discount_day = [1 for _ in range(self.datajson["总时长"])]
        self.order_big_discount_day[0:nday] = [self.datajson["订单折扣"] + i1 * change_every for i1 in range(nday)]
        self.order_big_price_day = np.array(self.order_big_discount_day) * self.datajson["原价订单"]
        change_every = (1 - self.datajson["工资折扣"]) / nday
        self.salary_discount_day = [1 for _ in range(self.datajson["总时长"])]
        self.salary_discount_day[0:nday] = [self.datajson["工资折扣"] + i1 * change_every for i1 in range(nday)]
        self.salary_price_day = np.array(self.salary_discount_day) * self.datajson["原价工资"]
        # 4. 服务未关闭，订单列表，包含 待服务 服务中 待结算"status": "servicing,waiting,calc"
        self.order_small_pass_list = []
        self.order_big_pass_list = []
        # 5. 资金借贷列表
        self.borrow_list = [
            ["2020-10-18", 5000000],
            ["2020-11-18", -1000000],
            ["2020-12-18", -1000000],
        ]
        print("模拟未来借贷列表：{}".format(self.borrow_list))

    def get_employ_std(self, ordertype):
        """统计 星级 地域 当前工资 报名加入日期(含培训期) 当天工资分享数量（可得2周内）"""
        # todo: 函数参数待统计
        employ_expect = self.datajson["月嫂日增期望值"]
        employ_std = self.datajson["月嫂日增标准差"]
        return employ_expect, employ_std

    def get_sale_std(self, ordertype):
        """统计 订单详情(星级 地域 服务等待期 当天订单价格 预售期日期 等待期日期) 订单销售速度 销售排行状态"""
        # todo: 函数参数待统计
        sale_expect = self.datajson["订单销售速度期望值"]
        sale_std = self.datajson["订单销售速度标准差"]
        return sale_expect, sale_std

    def get_service_shift_std(self, ordertype):
        """统计订单的 初始日期 和 最后落实日期"""
        # todo: 函数参数待统计
        service_shift_expect = self.datajson["日期偏移期望值"]
        service_shift_std = self.datajson["日期偏移标准差"]
        return service_shift_expect, service_shift_std

    def get_day_borrow(self, dateid):
        datestr = int2date(dateid).strftime("%Y-%m-%d")
        borrow_num = 0
        for bn in self.borrow_list:
            if bn[0] == datestr:
                borrow_num += bn[1]
        return borrow_num

    def get_day_sao_num(self, dateid, today_date):
        # 1. 当天可用月嫂数
        return self.sao_day_nums[dateid]
        # sao_daynum = self.sao_day_nums[dateid]
        # order_small_daynum = len(
        #     ["" for i1 in self.order_small_today_alllist if i1["start"] <= today_date and i1["end"] >= today_date])
        # return sao_daynum if sao_daynum > order_small_daynum else order_small_daynum

    def every_update_infor(self):
        availablelenth = self.sao_tomorrow_totalnum if self.sao_tomorrow_totalnum < self.sao_tomorrow_planservicing_num else self.sao_tomorrow_planservicing_num
        destrib_counter = 0
        # 安起始日期排序
        self.order_small_today_alllist = sorted(self.order_small_today_alllist, key=lambda x: x["start"])
        for order in self.order_small_today_alllist:
            if order["calcdate"] < self.tomorrow_str:
                # 如果明天超时，结算
                if order["calcdate"] < order["start"]:
                    # 结算取消订单
                    self.cap2_num -= order["orderprice"] + self.datajson["订单保证金"]
                    self.creater_num += self.datajson["订单保证金"]
                    self.subsidy2_num -= self.datajson["原价订单"] - order["orderprice"]
                    print("        取消订单结算")
                    print("        本单工资发放 = {}".format(order["salaryprice"]))
                    print("        本单双头补订单部分退回 = 原价订单 - 订单价格")
                    print("        {} = {} - {} )".format(self.datajson["原价订单"] - order["orderprice"],
                                                          self.datajson["原价订单"], order["orderprice"]))
                    print("        本单平台结算金额 = 订单价格 + 订单保证金")
                    print("        {} = {} + {}".format(order["orderprice"] + self.datajson["订单保证金"],
                                                        order["orderprice"], self.datajson["订单保证金"]))
                else:
                    self.cap2_num -= order["salaryprice"] + \
                                     self.datajson["原价订单"] * self.datajson["订单提成"] + self.datajson["订单保证金"]
                    self.subsidy2_num += order["salaryprice"] - self.datajson["原价工资"]
                    self.salary_num += order["salaryprice"]
                    self.creater_num += self.datajson["原价订单"] * self.datajson["订单提成"] + self.datajson["订单保证金"]
                    print("        正常订单结算")
                    print("        本单工资发放 = {}".format(order["salaryprice"]))
                    print("        本单双头补工资部分 = 实际工资 - 原价工资")
                    print("        {} = {} - {} )".format(order["salaryprice"] - self.datajson["原价工资"],
                                                          order["salaryprice"], self.datajson["原价工资"]))
                    print("        本单创始人金额 = 原价订单 * 订单提成 + 订单保证金")
                    print("        {} = {} * {} + {}".format(
                        self.datajson["原价订单"] * self.datajson["订单提成"] + self.datajson["订单保证金"],
                        self.datajson["原价订单"], self.datajson["订单提成"], self.datajson["订单保证金"]))
                    print("        平台结算金额 = 实际工资 + 原价订单 * 订单提成 + 订单保证金")
                    print("        {} = {} + {} * {} + {}".format(order["salaryprice"] + self.datajson["原价订单"] *
                                                                  self.datajson["订单提成"] + self.datajson["订单保证金"],
                                                                  order["salaryprice"], self.datajson["原价订单"],
                                                                  self.datajson["订单提成"], self.datajson["订单保证金"]
                                                                  ))
                order["status"] = "done"
            elif order["end"] < self.tomorrow_str:
                order["status"] = "calc"
            elif order["start"] > self.tomorrow_str:
                order["status"] = "waiting"
            else:
                # 本应 servicing 的订单
                destrib_counter += 1
                if destrib_counter > availablelenth:
                    # print(destrib_counter, emptylenth)
                    # 月嫂余量不足 一直变到 有月嫂
                    order["saoid"] = ""
                    order["start"] = self.tomorrow_str
                    order["end"] = self.future_addend_str
                    order["calc"] = self.future_addcalc_str
                    order["status"] = "waiting"
                    destrib_counter -= 1
                else:
                    order["saoid"] = "saoidxn"
                    order["status"] = "servicing"

        self.order_big_today_alllist = sorted(self.order_big_today_alllist, key=lambda x: x["start"])
        for order in self.order_big_today_alllist:
            if order["calcdate"] < self.tomorrow_str:
                # 如果明天超时，结算
                if order["calcdate"] < order["start"]:
                    raise Exception("        大客户不存在取消订单")
                else:
                    self.cap1_num -= order["salaryprice"] + \
                                     self.datajson["原价订单"] * self.datajson["订单提成"] + self.datajson["订单保证金"]
                    self.subsidy1_num += order["salaryprice"] - self.datajson["原价工资"]
                    self.salary_num += order["salaryprice"]
                    self.creater_num += self.datajson["原价订单"] * self.datajson["订单提成"] + self.datajson["订单保证金"]
                    print("        正常订单结算")
                    print("        本单工资发放 = {}".format(order["salaryprice"]))
                    print("        本单双头补工资部分 = 实际工资 - 原价工资")
                    print("        {} = {} - {} )".format(order["salaryprice"] - self.datajson["原价工资"],
                                                          order["salaryprice"], self.datajson["原价工资"]))
                    print("        本单创始人金额 = 原价订单 * 订单提成 + 订单保证金")
                    print("        {} = {} * {} + {}".format(
                        self.datajson["原价订单"] * self.datajson["订单提成"] + self.datajson["订单保证金"],
                        self.datajson["原价订单"], self.datajson["订单提成"], self.datajson["订单保证金"]))
                    print("        平台结算金额 = 实际工资 + 原价订单 * 订单提成 + 订单保证金")
                    print("        {} = {} + {} * {} + {}".format(order["salaryprice"] + self.datajson["原价订单"] *
                                                                  self.datajson["订单提成"] + self.datajson["订单保证金"],
                                                                  order["salaryprice"], self.datajson["原价订单"],
                                                                  self.datajson["订单提成"], self.datajson["订单保证金"]
                                                                  ))
                order["status"] = "done"
            elif order["end"] < self.tomorrow_str:
                order["status"] = "calc"
            elif order["start"] > self.tomorrow_str:
                order["status"] = "waiting"
            else:
                # 本应 servicing 的订单
                destrib_counter += 1
                if destrib_counter > availablelenth:
                    # print(destrib_counter, emptylenth)
                    # 月嫂余量不足 一直变到 有月嫂
                    order["saoid"] = ""
                    order["start"] = self.tomorrow_str
                    order["end"] = self.futureend_str
                    order["calc"] = self.futurecalc_str
                    order["status"] = "waiting"
                    destrib_counter -= 1
                else:
                    order["saoid"] = "saoidxn"
                    order["status"] = "servicing"

        # 清空过期的订单列表
        print("    未来一个等待期后月嫂总数量")
        print("    {}".format(self.sao_future_totalnum))
        print("    清空过期的订单列表")
        self.order_small_today_alllist = [orderold for orderold in self.order_small_today_alllist if
                                          orderold["status"] != "done"]
        self.order_big_today_alllist = [orderold for orderold in self.order_big_today_alllist if
                                        orderold["status"] != "done"]
        return self.order_small_today_alllist, self.order_big_today_alllist

    def every_new_orders(self, dateid):
        """创建 新订单： 预留月嫂，受销售数量上限"""
        #  待服务 服务中 待结算 : "servicing,waiting,calc,done"
        # 未来一个等待期后空闲月嫂数 可能已被占用 只是未分配，不能简单求空闲的长度 作为新增的长度
        # 1. 预留月嫂数 sao_future_freenum<len(sao_future_freelist)
        sao_future_freenum = self.sao_future_totalnum - int(
            self.sao_future_planservicing_num * (1 + self.datajson["月嫂预留百分数"]))
        print("    未来一个等待期后空闲月嫂数 = 未来一个等待期后月嫂总数 - 未来一个等待期后月嫂服务中数量 * ( 1 + 月嫂预留百分数 )")
        print("    {} = {} - {} * ( 1 + {} )".format(sao_future_freenum, self.sao_future_totalnum,
                                                     self.sao_future_planservicing_num, self.datajson["月嫂预留百分数"]))
        sale_expect, sale_std = self.get_sale_std(ordertype=None)
        sale_num = int(round(np.random.normal(loc=sale_expect, scale=sale_std, size=1)[0]))
        sale_final = sale_num if sale_num < sao_future_freenum else sao_future_freenum
        print("    当天订单出售数量 = 最小值 (订单售出上限，未来一个等待期后空闲月嫂数量) 如果是负值，不售出。")
        print("    {} = 最小值 ( {}，{} )".format(sale_final, sale_num, sao_future_freenum))
        # 2. 预留
        sale_final_big = int(sale_final * self.datajson["大客户订单百分数"])
        print("    当天大客户订单出售数量 = 当天订单出售数量 * 大客户订单百分数 ")
        print("    {} = {} * {}".format(sale_final_big, sale_final, self.datajson["大客户订单百分数"]))
        sale_final_small = sale_final - sale_final_big
        print("    当天散单订单出售数量 = 当天订单出售数量 - 当天大客户订单出售数量 ")
        print("    {} = {} * {}".format(sale_final_small, sale_final, sale_final_big))
        self.order_small_today_sale_list = []
        self.order_big_today_sale_list = []
        # self.datajson["大客户订单百分数"]
        if sale_final > 0:
            # 6.1. 直接模拟卖出了 未来等待期的 原始订单。未来变更，直接现在修改
            service_shift_expect, service_shift_std = self.get_service_shift_std(ordertype=None)
            randshifts = np.random.normal(loc=service_shift_expect, scale=service_shift_std, size=sale_final_small)
            randshifts = [
                1000 if rn < self.datajson["变更服务费率"][0][0][0] or rn > self.datajson["变更服务费率"][-1][0][1] else rn for rn
                in randshifts]
            shiftsstart = [int(round(rn) + self.futurestart_int) for rn in randshifts]
            shiftsend = [rn + self.datajson["服务期"] for rn in shiftsstart]
            shiftscalc = [rn + self.datajson["结算期"] for rn in shiftsend]
            self.order_small_today_sale_list = [{"orderid": "saoidx_" + self.tomorrow_str,  # saoid 只为防止重复占位，跟绑定月嫂无关。
                                                 "waitdate": self.tomorrow_str,
                                                 "oristart": self.futurestart_str,
                                                 "start": int2date(shiftsstart[idn]).strftime("%Y-%m-%d"),
                                                 "end": int2date(shiftsend[idn]).strftime("%Y-%m-%d"),
                                                 "calcdate": int2date(shiftscalc[idn]).strftime("%Y-%m-%d"),
                                                 "saoid": "",
                                                 "orderprice": self.order_small_price_day[dateid],
                                                 "salaryprice": self.salary_price_day[dateid],
                                                 "status": "waiting"} for idn in range(sale_final_small) if
                                                idn < sale_final]
            # 大客户只考虑偏移，不考虑费用
            randshifts_big = np.random.normal(loc=service_shift_expect, scale=service_shift_std, size=sale_final_big)
            # shiftsstart_big = [rn if rn > 0 else -rn for rn in randshifts_big]
            randshifts_big = [rn if rn > 0 else -rn for rn in randshifts_big]
            shiftsstart_big = [int(round(rn) + self.futurestart_int) for rn in randshifts_big]
            shiftsend_big = [rn + self.datajson["服务期"] for rn in shiftsstart_big]
            shiftscalc_big = [rn + self.datajson["结算期"] for rn in shiftsend_big]
            self.order_big_today_sale_list = [{"bigorderid": "saoidx_" + self.tomorrow_str,  # saoid 只为防止重复占位，跟绑定月嫂无关。
                                               "waitdate": self.tomorrow_str,
                                               "oristart": self.futurestart_str,
                                               "start": int2date(shiftsstart_big[idn]).strftime("%Y-%m-%d"),
                                               "end": int2date(shiftsend_big[idn]).strftime("%Y-%m-%d"),
                                               "calcdate": int2date(shiftscalc_big[idn]).strftime("%Y-%m-%d"),
                                               "saoid": "",
                                               "orderprice": self.order_big_price_day[dateid],
                                               "salaryprice": self.salary_price_day[dateid],
                                               "status": "waiting"} for idn in range(sale_final_big) if
                                              idn < sale_final]
            print("    生成随机服务日期变更的散单订单，变更超出费率区间表的直接设为取消")
            print("    生成随机服务日期变更的大客户订单，不考虑变更问题")
            # 6.2. 模拟原始售出资金变动，未来变更，在排班服务时修正
            self.cap2_num += (self.order_small_price_day[dateid] + self.datajson["月嫂使用费"] + self.datajson[
                "订单保证金"]) * sale_final_small
            self.subsidy2_num += (
                                     self.datajson["原价订单"] * (
                                         1 - self.order_small_discount_day[dateid])) * sale_final_small
            print("    当天散客订单出售金额 = ( 订单价格 + 月嫂使用费 + 订单保证金) * 散客售出数量")
            print("    {} = ( {} + {} + {} ) * {}".format(self.cap2_num, self.order_small_price_day[dateid],
                                                          self.datajson["月嫂使用费"], self.datajson["订单保证金"],
                                                          sale_final_small))
            print("    当天双头补订单部分 = 原价订单 * ( 1 - 订单折扣 ) * 散客售出数量")
            print("    {} = {} * ( 1 - {} ) * {}".format(
                (self.datajson["原价订单"] * (1 - self.order_small_discount_day[dateid])) *
                sale_final_small, self.datajson["原价订单"], self.order_small_discount_day[dateid], sale_final_small))
            self.cap1_num += (self.order_big_price_day[dateid] + self.datajson["月嫂使用费"] + self.datajson[
                "订单保证金"]) * sale_final_big
            self.subsidy1_num += (self.datajson["原价订单"] * (1 - self.order_big_discount_day[dateid])) * sale_final_big
            print("    当天大客户订单出售金额 = ( 订单价格 + 月嫂使用费 + 订单保证金) * 大客户售出数量")
            print("    {} = ( {} + {} + {} ) * {}".format(self.cap1_num, self.order_big_price_day[dateid],
                                                          self.datajson["月嫂使用费"], self.datajson["订单保证金"],
                                                          sale_final_big))
            print("    当天单头补订单部分 = 原价订单 * ( 1 - 订单折扣 ) * 大客户售出数量")
            print("    {} = {} * ( 1 - {} ) * {}".format(
                (self.datajson["原价订单"] * (1 - self.order_big_discount_day[dateid])) *
                sale_final_big, self.datajson["原价订单"], self.order_big_discount_day[dateid], sale_final_big))
            print("    当天创始人订单保证金额= 订单保证金 * 总售出数量 ) ")
            print("    {} = {} * {}".format(self.datajson["订单保证金"] * sale_final, self.datajson["订单保证金"], sale_final))
            self.creater_num -= self.datajson["订单保证金"] * sale_final
        return self.order_small_today_sale_list, self.order_big_today_sale_list

    def every_status_orders(self):
        """更新 订单 服务状态，加入 变期 订单费用"""
        for order in self.order_big_today_alllist:
            # 当天服务中的 订单和对应月嫂。如果有月嫂 突然退出 需要外部程序清空该订单的月嫂id更改状态为 等待期 或其他约定字段
            if order["status"] == "servicing":
                self.order_big_today_servicingnum += 1
                self.sao_today_servicingnum += 1
            elif order["status"] == "waiting":
                # if today_date>order["start"]:
                self.order_big_today_waitingnum += 1
                # 明天为已排班 切为起始日，切 变更过起始日期，则修改资金收入
                if order["start"] == self.tomorrow_str and order["start"] != order["oristart"] and order["saoid"] != "":
                    # 大客户无违约金相关费用
                    print("        当天服务中订单数量")
                if order["start"] <= self.today_date and order["end"] >= self.today_date and order["saoid"] == "":
                    self.order_big_today_delaynum += 1
            # 明天订单在服务期的数量 即使未分配月嫂，也占一个位置
            if order["start"] <= self.tomorrow_str and order["end"] >= self.tomorrow_str:
                self.sao_tomorrow_planservicing_num += 1
            # 新订单在服务期的数量 即使未分配月嫂，也占一个位置
            if order["start"] <= self.futurestart_str and order["end"] >= self.futurestart_str:
                self.sao_future_planservicing_num += 1
                # print(order["start"], order["end"], self.futurestart_str)
        for order in self.order_small_today_alllist:
            # 当天服务中的 订单和对应月嫂。如果有月嫂 突然退出 需要外部程序清空该订单的月嫂id更改状态为 等待期 或其他约定字段
            if order["status"] == "servicing":
                self.order_small_today_servicingnum += 1
                self.sao_today_servicingnum += 1
            elif order["status"] == "waiting":
                # if today_date>order["start"]:
                self.order_small_today_waitingnum += 1
                # 明天为已排班 切为起始日，切 变更过起始日期，则修改资金收入
                if order["start"] == self.tomorrow_str and order["start"] != order["oristart"] and order["saoid"] != "":
                    diffdays = datediff(order["oristart"], order["start"])
                    for fee in self.datajson["变更服务费率"]:
                        if diffdays <= fee[0][1] and diffdays >= fee[0][0]:
                            self.cap2_num += fee[1] * order["orderprice"]
                            print("        订单变更信息，原始服务起始日期, 实际服务起始日期")
                            print("                    {}, {}".format(order["oristart"], order["start"]))
                            print("        变更服务费率区间，上限日期, 下限日期, 征收金额=订单价格*对应费率")
                            print("                        {}, {}, {} = {} * {}".format(fee[0][0], fee[0][1],
                                                                                        fee[1] * order["orderprice"],
                                                                                        order["orderprice"], fee[1]))
                            break
                if order["start"] <= self.today_date and order["end"] >= self.today_date and order["saoid"] == "":
                    self.order_small_today_delaynum += 1
            # 明天订单在服务期的数量 即使未分配月嫂，也占一个位置
            if order["start"] <= self.tomorrow_str and order["end"] >= self.tomorrow_str:
                self.sao_tomorrow_planservicing_num += 1
            # 新订单在服务期的数量 即使未分配月嫂，也占一个位置
            if order["start"] <= self.futurestart_str and order["end"] >= self.futurestart_str:
                self.sao_future_planservicing_num += 1
                # print(order["start"], order["end"], self.futurestart_str)
        print("    当天散单服务中订单数量")
        print("    {}".format(self.order_small_today_servicingnum))
        print("    当天散单等待服务中订单数量")
        print("    {}".format(self.order_small_today_waitingnum))
        print("    当天散单延迟服务订单数量")
        print("    {}".format(self.order_small_today_delaynum))
        print("    当天大客户服务中订单数量")
        print("    {}".format(self.order_big_today_servicingnum))
        print("    当天大客户等待服务中订单数量")
        print("    {}".format(self.order_big_today_waitingnum))
        print("    当天大客户延迟服务订单数量")
        print("    {}".format(self.order_big_today_delaynum))
        print("    明天订单在服务期的数量 即使未分配月嫂，也占一个位置")
        print("    {}".format(self.sao_tomorrow_planservicing_num))
        print("    新订单在服务期的数量 即使未分配月嫂，也占一个位置")
        print("    {}".format(self.sao_future_planservicing_num))
        print("    当天订单开始服务的日期变更费用=所有订单变更的日期的费用和")
        print("    {}".format(self.cap2_num))
        return self.order_small_today_alllist, self.order_big_today_alllist

    def gene_full_service(self):
        self.y_capital = []  # 资金池 历史累计余额
        self.y_1_capital = []  # 资金池大客户部分 历史累计余额
        self.y_2_capital = []  # 资金池散客户部分 历史累计余额
        self.y_2subsidy = []  # 双头补散客 历史累计余额
        self.y_1subsidy = []  # 双头补大客户 历史累计余额
        self.y_borrow = []  # 借贷资金 历史累计余额 正为借出 负为贷入
        self.y_salary = []  # 原价工资累积
        self.y_creater = []  # 创始人收益累积
        self.y_capital_change = []  # 每日资金变化
        self.y_1_capital_change = []  # 每日资金变化
        self.y_2_capital_change = []  # 每日资金变化
        self.y_2subsidy_change = []  # 双头补散客 资金变化
        self.y_1subsidy_change = []  # 双头补大客户 资金变化
        self.y_borrow_change = []  # 借贷资金变化
        self.y_salary_change = []  # 原价工资变化
        self.y_creater_change = []  # 创始人收益变化
        self.y_sao_total = []  # 月嫂总量 是 订单总容量 的上限
        self.y_sao_servicing = []  # 服务中月嫂数 = 服务中订单数
        self.y_sao_short = []  # 延期且未开订单数 = 月嫂的缺额
        self.y_sao_free = []  # 未利用月嫂数 = 未开发订单数
        self.y_order_total = []  # 订单总容量 包含待服务的 不含未开发的，上限为 当天月嫂总量
        self.y_order_waiting = []  # 待服务订单数 = 未到期订单数 + 月嫂的缺额
        self.y_order_servicing = []  # 服务中订单数 = 服务中月嫂数
        self.y_order_delay = []  # 延期且未开订单数 = 月嫂的缺额
        self.y_order_small_total = []  # 订单总容量 包含待服务的 不含未开发的，上限为 当天月嫂总量
        self.y_order_small_waiting = []  # 待服务订单数 = 未到期订单数 + 月嫂的缺额
        self.y_order_small_servicing = []  # 服务中订单数 = 服务中月嫂数
        self.y_order_small_delay = []  # 延期且未开订单数 = 月嫂的缺额
        self.y_order_big_total = []  # 订单总容量 包含待服务的 不含未开发的，上限为 当天月嫂总量
        self.y_order_big_waiting = []  # 待服务订单数 = 未到期订单数 + 月嫂的缺额
        self.y_order_big_servicing = []  # 服务中订单数 = 服务中月嫂数
        self.y_order_big_delay = []  # 延期且未开订单数 = 月嫂的缺额
        self.y_order_free = []  # 未开发订单数 = 未利用月嫂数
        self.x = [i1 for i1 in range(self.datajson["总时长"])]
        self.x_label = [int2date(i1).strftime("%Y-%m-%d") for i1 in range(self.datajson["总时长"])]
        saoid_list = [i1 for i1 in range(self.datajson["月嫂专职数"] + self.datajson["月嫂精英数"])]
        addday = int(round(self.datajson["服务逾期补偿"][0] * self.datajson["等待期"]))
        # 根据月嫂数生成对应的订单，初始化时 是否利用等待期的空置月嫂？ 尽量短的
        # 4. 服务未关闭，订单列表，包含 待服务 服务中 待结算"status": "servicing,waiting,calc,done"
        self.order_small_today_alllist = copy.deepcopy(self.order_small_pass_list)
        self.order_big_today_alllist = copy.deepcopy(self.order_big_pass_list)
        for dateid, today_date in enumerate(self.x_label):
            # print("dateid:", dateid)
            print("当天日期:", today_date)
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
            self.cap1_num = 0  # 资金初始
            self.cap2_num = 0  # 资金初始
            self.subsidy2_num = 0  # 双头补 分 订单和工资两部分
            self.subsidy1_num = 0  # 双头补 分 关系费和工资两部分
            self.salary_num = 0  # 工资
            self.creater_num = 0  # 创始人结算
            # 1. 当天可用的月嫂列表=已排班的+未排班的
            sao_today_totalnum = self.get_day_sao_num(dateid, today_date)
            self.sao_tomorrow_totalnum = self.get_day_sao_num(dateid + 1, self.tomorrow_str)
            self.sao_future_totalnum = self.get_day_sao_num(self.futurestart_int, self.futurestart_str)
            print("    当天月嫂总数量")
            print("    {}".format(sao_today_totalnum))
            # 2. 当天 服务未完成，已排服务列表 = self.sao_today_servicingnum
            self.order_small_today_servicingnum = 0
            self.order_small_today_waitingnum = 0
            self.order_small_today_delaynum = 0
            self.order_big_today_servicingnum = 0
            self.order_big_today_waitingnum = 0
            self.order_big_today_delaynum = 0
            # 3. 月嫂已排班记录 = 已排服务列表, 当天 明天和未来的 因为可能有月嫂缺额，未必相等
            self.sao_today_servicingnum = 0
            self.sao_tomorrow_planservicing_num = 0
            self.sao_future_planservicing_num = 0  # 未来一个等待期后待服务列表 = 对应区间的月嫂数 即 self.order_small_future_waitinglist
            # 4. 当天 未排班服务列表
            # 最近空闲列表，未分配 但可能已占位，可用长度要用月嫂总数减去分配长度
            self.sao_tomorrow_freenum = 0
            # 生成订单统计数据
            self.order_small_today_alllist, self.order_big_today_alllist = self.every_status_orders()
            order_small_today_length = len(self.order_small_today_alllist)
            order_big_today_length = len(self.order_big_today_alllist)
            sao_today_freenum = sao_today_totalnum - self.sao_today_servicingnum
            self.sao_tomorrow_freenum = self.sao_tomorrow_totalnum - self.sao_tomorrow_planservicing_num
            # 6. 当天 空闲服务列表 直接分配 按日期生成订单号,
            self.order_small_today_sale_list, self.order_big_today_sale_list = self.every_new_orders(dateid)
            # 7. 更新 跟新订单合并
            self.order_small_today_alllist += self.order_small_today_sale_list
            self.order_big_today_alllist += self.order_big_today_sale_list
            # 8. 更新明天每单的状态, 同时 输出排班信息，结算
            self.order_small_today_alllist, self.order_big_today_alllist = self.every_update_infor()

            # 9. 当天 非订单开销结算
            print("    当天平台开销")
            print("    {}".format(self.datajson["平台开销"]))
            if len(self.y_2_capital) == 0 or self.y_2_capital[-1] >= self.datajson["平台开销"]:
                self.cap2_num -= self.datajson["平台开销"]
            else:
                print(len(self.y_2_capital))
                print(self.y_2_capital[-1])
                print(self.datajson["平台开销"])
                raise Exception("没有声明散客资金池不足开销的情况")
            borrow_num = self.get_day_borrow(dateid)
            print("    当天计划借贷金额")
            print("    {}".format(borrow_num))
            if len(self.y_2_capital) == 0 or self.y_2_capital[-1] >= self.datajson["平台开销"] + borrow_num:
                print("    当天实际借贷金额")
                print("    {}".format(borrow_num))
                self.cap2_num -= borrow_num
            else:
                print(len(self.y_2_capital))
                print(self.y_2_capital[-1])
                print(self.datajson["平台开销"])
                print(borrow_num)
                raise Exception("没有声明散客资金池不足借贷的情况")
            print("    散客资金量扣除开销变化后金额")
            print("    {}".format(self.cap2_num))
            # 资金汇总
            self.cap_num = self.cap1_num + self.cap2_num
            captmp = self.cap_num
            print("    当天订单结算总额：{}".format(captmp))
            self.cap_num += self.datajson["平台通讯费"]
            print("    当天平台通讯费")
            print("    {}".format(self.datajson["平台通讯费"]))
            print("    当天资金变化 =  当天订单结算总额 + 当天平台通讯费")
            print("    {} = {} + {}".format(self.cap_num, captmp, self.datajson["平台通讯费"]))
            self.y_capital_change.append(self.cap_num)  # 每日资金变化
            self.y_1_capital_change.append(self.cap1_num)  # 每日资金变化
            self.y_2_capital_change.append(self.cap2_num)  # 每日资金变化
            self.y_1subsidy_change.append(self.subsidy1_num)  # 双头补金额
            self.y_2subsidy_change.append(self.subsidy2_num)  # 双头补金额
            self.y_borrow_change.append(borrow_num)  # 借贷资金变化
            self.y_salary_change.append(self.salary_num)  # 原价工资变化
            self.y_creater_change.append(self.creater_num)  # 创始人收益变化
            self.y_sao_free.append(sao_today_freenum)  # 未利用月嫂数 = 未开发订单数
            self.y_sao_short.append(self.order_small_today_delaynum + self.order_big_today_delaynum)  # 月嫂缺额
            self.y_sao_total.append(sao_today_totalnum)  # 月嫂总量
            self.y_sao_servicing.append(self.sao_today_servicingnum)  # 服务中月嫂数 = 服务中订单数
            self.y_order_small_total.append(order_small_today_length)  # 订单总容量 包含待服务的 和 服务中的。不含未开发的
            self.y_order_small_waiting.append(self.order_small_today_waitingnum)  # 待服务订单数
            self.y_order_small_servicing.append(self.order_small_today_servicingnum)  # 服务中订单数 = 服务中月嫂数
            self.y_order_small_delay.append(self.order_small_today_delaynum)  # 延迟订单数
            self.y_order_big_total.append(order_big_today_length)  # 订单总容量 包含待服务的 和 服务中的。不含未开发的
            self.y_order_big_waiting.append(self.order_big_today_waitingnum)  # 待服务订单数
            self.y_order_big_servicing.append(self.order_big_today_servicingnum)  # 服务中订单数 = 服务中月嫂数
            self.y_order_big_delay.append(self.order_big_today_delaynum)  # 延迟订单数
            self.y_order_servicing.append(
                self.order_small_today_servicingnum + self.order_big_today_servicingnum)  # 订单总容量 包含待服务的 和 服务中的。不含未开发的
            self.y_order_waiting.append(
                self.order_small_today_waitingnum + self.order_big_today_waitingnum)  # 订单总容量 包含待服务的 和 服务中的。不含未开发的
            self.y_order_total.append(order_small_today_length + order_big_today_length)  # 订单总容量 包含待服务的 和 服务中的。不含未开发的
            self.y_order_delay.append(self.order_small_today_delaynum + self.order_big_today_delaynum)  # 延迟订单数
            self.y_order_free.append(sao_today_freenum)  # 未开发订单数 = 未利用月嫂数
            # print(len(self.order_small_today_alllist))
            # print("sao_today_totalnum", sao_today_totalnum)
            # print(self.order_small_today_waitingnum)
            # print(self.order_small_today_servicingnum)
            # 9. 累计值生成
            # print(self.sao_day_nums)
            if dateid == 0:
                self.y_capital.append(self.datajson["初始资金"] + self.y_capital_change[dateid])
                self.y_1_capital.append(self.datajson["初始大客户资金"] + self.y_1_capital_change[dateid])
                self.y_2_capital.append(self.datajson["初始散客资金"] + self.y_2_capital_change[dateid])
                self.y_1subsidy.append(self.datajson["初始单头补资金"] + self.y_1subsidy_change[dateid])  # 双头补金额累计
                self.y_2subsidy.append(self.datajson["初始双头补资金"] + self.y_2subsidy_change[dateid])  # 双头补金额累计
                self.y_salary.append(self.datajson["初始发放工资"] + self.y_salary_change[dateid])  # 工资累计
                self.y_creater.append(self.datajson["初始创始人资金"] + self.y_creater_change[dateid])  # 创始人收益累计
                self.y_borrow.append(self.datajson["初始借贷"] + self.y_borrow_change[dateid])  # 借贷累计
            else:
                self.y_capital.append(self.y_capital[-1] + self.y_capital_change[dateid])  # 资金池
                self.y_1_capital.append(self.y_1_capital[-1] + self.y_1_capital_change[dateid])
                self.y_2_capital.append(self.y_2_capital[-1] + self.y_2_capital_change[dateid])
                self.y_borrow.append(self.y_borrow[-1] + self.y_borrow_change[dateid])  # 借贷累计
                self.y_1subsidy.append(self.y_1subsidy[-1] + self.y_1subsidy_change[dateid])  # 双头补金额累计
                self.y_2subsidy.append(self.y_2subsidy[-1] + self.y_2subsidy_change[dateid])  # 双头补金额累计
                self.y_salary.append(self.y_salary[-1] + self.y_salary_change[dateid])  # 工资累计
                self.y_creater.append(self.y_creater[-1] + self.y_creater_change[dateid])  # 创始人收益累计
            print("    当天历史资金累计")
            print("    {}".format(self.y_capital[-1]))
            print("    当天历史大客户资金累计")
            print("    {}".format(self.y_1_capital[-1]))
            print("    当天历史散客资金累计")
            print("    {}".format(self.y_2_capital[-1]))
            print("    当天历史散客双头补资金累计")
            print("    {}".format(self.y_2subsidy[-1]))
            print("    当天历史大客户单头补开销资金累计")
            print("    {}".format(self.y_1subsidy[-1]))
            print("    当天历史借贷累计")
            print("    {}".format(self.y_borrow[-1]))
            print("    当天历史工资累计")
            print("    {}".format(self.y_salary[-1]))
            print("    当天历史创始人资金累计")
            print("    {}".format(self.y_creater[-1]))

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
    # fs_ins.x_label.insert(0, int2date(-1).strftime("%Y-%m-%d"))
    # exit()
    print("use time is {}s".format(time.time() - stime))
    # 4. 绘图
    # titles = ["月嫂总量", "服务中月嫂数", "月嫂缺额数", "未利用月嫂数",
    #           "总资金池累计", "散单资金池累计", "大客户资金池累计", "双头补金额累积", "单头补金额累积", "工资累积", "创始人收益累积",
    #           "订单总容量", "订单散客容量", "订单大客户容量", "待服务订单数", "服务中订单数", "延迟订单数", "未开发订单数"]
    # ys = [np.array(fs_ins.y_sao_total) / max(fs_ins.y_sao_total),
    #       np.array(fs_ins.y_sao_servicing) / max(fs_ins.y_sao_servicing),
    #       np.array(fs_ins.y_sao_short) / max(fs_ins.y_sao_short),
    #       np.array(fs_ins.y_sao_free) / max(fs_ins.y_sao_free),
    #       np.array(fs_ins.y_capital) / max(fs_ins.y_capital),
    #       np.array(fs_ins.y_2_capital) / max(fs_ins.y_2_capital),
    #       np.array(fs_ins.y_1_capital) / max(fs_ins.y_1_capital),
    #       np.array(fs_ins.y_2subsidy) / max(fs_ins.y_2subsidy),
    #       np.array(fs_ins.y_1subsidy) / max(fs_ins.y_1subsidy),
    #       np.array(fs_ins.y_salary) / max(fs_ins.y_salary),
    #       np.array(fs_ins.y_creater) / max(fs_ins.y_creater),
    #       np.array(fs_ins.y_order_small_total) / max(fs_ins.y_order_small_total),
    #       np.array(fs_ins.y_order_small_waiting) / max(fs_ins.y_order_small_waiting),
    #       np.array(fs_ins.y_order_small_servicing) / max(fs_ins.y_order_small_servicing),
    #       np.array(fs_ins.y_order_delay) / max(fs_ins.y_order_delay),
    #       np.array(fs_ins.y_order_free) / max(fs_ins.y_order_free),
    #       ]
    # plot_curve(fs_ins.x_label, ys, titles)
    # bar3dplot([fs_ins.x_label, titles, list(itertools.chain(*ys))])
    # exit()
    titles = ["工资价格", "订单价格"]
    ys = [fs_ins.salary_price_day, fs_ins.order_small_price_day]
    plot_curve(fs_ins.x_label, ys, titles)

    titles = ["月嫂总量", "服务中月嫂数", "月嫂缺额数", "未利用月嫂数"]
    ys = [fs_ins.y_sao_total, fs_ins.y_sao_servicing, fs_ins.y_sao_short, fs_ins.y_sao_free]
    plot_curve(fs_ins.x_label, ys, titles)
    # 每日资金 每日月嫂
    # print(fs_ins.y_capital_change)
    # print(fs_ins.y_salary_change)
    # print(fs_ins.y_creater_change)
    titles = ["总资金池增量", "大客户资金池增量", "散单资金池增量", "双头补金额增量", "单补金额增量", "借贷增量", "工资增量", "创始人收益增量"]
    ys = [fs_ins.y_capital_change, fs_ins.y_1_capital_change, fs_ins.y_2_capital_change, fs_ins.y_2subsidy_change,
          fs_ins.y_1subsidy_change, fs_ins.y_borrow_change, fs_ins.y_salary_change, fs_ins.y_creater_change]
    plot_curve(fs_ins.x_label, ys, titles)
    # print(fs_ins.y_capital)
    # print(fs_ins.y_salary)
    # print(fs_ins.y_creater)
    # titles = ["总资金池累计", "大客户资金池累计", "散单资金池累计", "双头补金额累积", "单头补金额累积", "借贷累计", "工资累积", "创始人收益累积"]
    # ys = [fs_ins.y_capital, fs_ins.y_1_capital, fs_ins.y_2_capital, fs_ins.y_2subsidy, fs_ins.y_1subsidy,
    #       fs_ins.y_borrow, fs_ins.y_salary, fs_ins.y_creater]
    titles = ["总资金池累计", "大客户资金池累计", "散单资金池累计", "双头补金额累积", "单头补金额累积", "借贷累计", "创始人收益累积"]
    ys = [fs_ins.y_capital, fs_ins.y_1_capital, fs_ins.y_2_capital, fs_ins.y_2subsidy, fs_ins.y_1subsidy,
          fs_ins.y_borrow, fs_ins.y_creater]
    plot_curve(fs_ins.x_label, ys, titles)
    # print(fs_ins.y_order_small_total)
    # print(fs_ins.y_order_small_waiting)
    # print(fs_ins.y_order_small_servicing)
    # print(fs_ins.y_order_free)
    # # ind = [i1 for i1 in range(len(fs_ins.y_order_free))]
    # # print([i1 for i1 in zip(ind, fs_ins.y_order_free)])
    # titles = ["订单总容量", "订单散单容量", "订单大客户容量", "待服务总订单数", "待服务散客订单数", "待服务大客户订单数",
    #           "服务中总订单数", "服务中散客订单数", "服务中大客户订单数", "大客户延迟订单数", "散客延迟订单数", "总延迟订单数", "未开发订单数"]
    # ys = [fs_ins.y_order_total, fs_ins.y_order_small_total, fs_ins.y_order_big_total,
    #       fs_ins.y_order_waiting, fs_ins.y_order_small_waiting, fs_ins.y_order_big_waiting,
    #       fs_ins.y_order_servicing, fs_ins.y_order_small_servicing, fs_ins.y_order_big_servicing,
    #       fs_ins.y_order_big_delay, fs_ins.y_order_small_delay, fs_ins.y_order_delay,
    #       fs_ins.y_order_free]
    titles = ["订单总容量", "订单散单容量", "订单大客户容量", "待服务总订单数", "待服务散客订单数", "待服务大客户订单数"]
    ys = [fs_ins.y_order_total, fs_ins.y_order_small_total, fs_ins.y_order_big_total,
          fs_ins.y_order_waiting, fs_ins.y_order_small_waiting, fs_ins.y_order_big_waiting,
          ]
    plot_curve(fs_ins.x_label, ys, titles)
    titles = ["服务中总订单数", "服务中散客订单数", "服务中大客户订单数", "大客户延迟订单数", "散客延迟订单数", "总延迟订单数", "未开发订单数"]
    ys = [fs_ins.y_order_servicing, fs_ins.y_order_small_servicing, fs_ins.y_order_big_servicing,
          fs_ins.y_order_big_delay, fs_ins.y_order_small_delay, fs_ins.y_order_delay, fs_ins.y_order_free]
    plot_curve(fs_ins.x_label, ys, titles)


if __name__ == '__main__':
    np.random.seed(5)
    main()
    print("end")
