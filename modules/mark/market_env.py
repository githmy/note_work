from random import random
import numpy as np
import pandas as pd
import math
import time
import gym
from gym import spaces
from modules.stocks.stock_data import Stockdata
from modules.stocks.stock_chara import gene_1pd


class MarketEnv(gym.Env):
    PENALTY = 1  # 0.999756079

    def __init__(self, input_codes, start_date, end_date, scope=60, sudden_death=-1.,
                 cumulative_reward=False):
        self.startDate = pd.Timestamp(start_date)
        self.endDate = pd.Timestamp(end_date)
        self.scope = scope
        self.sudden_death = sudden_death
        self.cumulative_reward = cumulative_reward

        # 数据定义获取
        self.inputCodes = []
        self.targetCodes_all = []
        self.dataMap = {}
        print("data from {} to {}.".format(self.startDate, self.endDate))
        self._get_ori_data_list()
        # 定义行为
        self.actions = [
            "UP",
            "WAIT",
            "DOWN",
        ]
        # 定义操作状态空间
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(np.ones(scope * (len(input_codes) + 1)) * -1,
                                            np.ones(scope * (len(input_codes) + 1)))
        # 初始化
        self.reset = self._reset
        self.render = self._render
        self._seed()

    def _get_ori_data_list(self):
        # 获取原始数据 和 名称列表
        dclass = Stockdata()
        stocklist = dclass.data_stocklist()
        self.targetCodes_all = list(stocklist.reshape(-1))
        self.dataMap = dclass.data_stocklist_value("D", stocklist)

    def step_mem(self):
        if self.done:
            return self.state, self.reward, self.done, {}
        # 生成状态
        self.defineState()
        # 状态指标变更
        self.currentTargetIndex += 1
        if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[
            self.currentTargetIndex]:
            self.done = True
        return self.state, self.reward, self.done, {}

    def step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.reward = 0
        # 判断操作，正向时boughts +1，反向时清空boughts
        if self.actions[action] == "DOWN":
            if sum(self.boughts) < 0:
                for b in self.boughts:
                    self.reward += -(b + 1)
                if self.cumulative_reward:
                    self.reward = self.reward / max(1, len(self.boughts))
                if self.sudden_death * len(self.boughts) > self.reward:
                    self.done = True
                self.boughts = []
            self.boughts.append(1.0)
        elif self.actions[action] == "UP":
            if sum(self.boughts) > 0:
                for b in self.boughts:
                    self.reward += b - 1
                if self.cumulative_reward:
                    self.reward = self.reward / max(1, len(self.boughts))
                if self.sudden_death * len(self.boughts) > self.reward:
                    self.done = True
                self.boughts = []
            self.boughts.append(-1.0)
        elif self.actions[action] == "WAIT":
            pass
        else:
            pass
        vari = self.input_target[self.targetDates[self.currentTargetIndex]][2]
        self.cum_win = self.cum_win * (1 + vari)

        # 计算 一次交易的收益
        for i in range(len(self.boughts)):
            self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + vari * (-1 if sum(self.boughts) < 0 else 1))

        self.defineState()
        self.currentTargetIndex += 1
        if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[
            self.currentTargetIndex]:
            self.done = True

        # 本轮的奖励求和，按交易次数算均值
        if self.done:
            for b in self.boughts:
                self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1
            if self.cumulative_reward:
                self.reward = self.reward / max(1, len(self.boughts))
            self.boughts = []
        return self.state, self.reward, self.done, {
            "dt": self.targetDates[self.currentTargetIndex],
            "cum": self.cum_win,
            "code": self.targetCode}

    def _reset(self):
        # 重置股票组，初始状态
        self.targetCode = self.targetCodes_all[int(random() * len(self.targetCodes_all))]
        # self.input_target = self.dataMap[self.targetCode]
        parajson = {
            "avenlist": [5, 20, 60],
            # "labellist": [1, 2, 4, 8, 16, 32, 64, 128, 256],
            "labellist": [1, 2],
        }
        split_n = 0.8
        self.input_target_all = gene_1pd(self.dataMap[self.targetCode], parajson)
        charalist = self.input_target_all.columns
        charalist = [i for i in charalist if i not in ["open", "high", "close", "low", "volume", "price_change"]]
        charalist = [i for i in charalist if not (i.startswith("ma") or i.startswith("v_ma"))]
        charalist = [i for i in charalist if not (i.startswith("charao_") and i.endswith("_FI"))]
        charalist = [i for i in charalist if not (i.startswith("charaR_") and i.endswith("_volr"))]
        charalist = [i for i in charalist if not (i.startswith("charaR_") and i.endswith("_CCI"))]
        charalist = [i for i in charalist if not i.startswith("chara_")]
        # charalist = [i for i in charalist if not (i.startswith("chara_") and i.endswith("_EVM"))]
        # charalist = [i for i in charalist if not (i.startswith("chara_") and i.endswith("_EWMA"))]
        # charalist = [i for i in charalist if not (i.startswith("chara_") and i.endswith("_BBdown"))]
        # charalist = [i for i in charalist if not (i.startswith("chara_") and i.endswith("_BBup"))]
        # charalist = [i for i in charalist if not (i.startswith("chara_") and i.endswith("_SMA"))]
        self.input_target_all = self.input_target_all[charalist]
        self.input_target_all["p_change"] = self.input_target_all["p_change"] / 100
        self.input_target_all["turnover"] = self.input_target_all["turnover"] / 100
        charalist = self.input_target_all.columns
        charalist = [i for i in charalist if (i.startswith("charaR_") and i.endswith("_EWMAR"))]
        charalist = [i for i in charalist if (i.startswith("charaR_") and i.endswith("_SMAR"))]
        for i1 in charalist:
            self.input_target_all[i1] = self.input_target_all[i1] - 1

        # print(self.input_target_all.head(100))
        lenth0 = self.input_target_all.shape[0]
        self.train_to = int(split_n * lenth0)
        self.input_target = self.input_target_all[0:self.train_to]
        self.input_target_test = self.input_target_all[self.train_to - self.scope:]
        print("code: {} train from {} to {}, valid from {} to {}".format(
            self.targetCode, 0, self.train_to - 1, self.train_to, lenth0))
        self.inputlist = [i for i in self.input_target.columns if not i.startswith("ylabel_")]
        self.targetlist = ["ylabel_1_c"]
        self.chara_length = len(self.inputlist)
        self.target_length = len(self.targetlist)
        self.targetDates = sorted(self.input_target.index)
        self.currentTargetIndex = self.scope
        # 历史状态
        self.boughts = []
        # 赢值/持仓时间。。
        self.reward = 0
        # 赢值积累
        self.cum_win = 1.
        # 结束标志
        if self.train_to < 60:
            self.done = True
        else:
            self.done = False
        self.defineState()
        self._get_test_data()
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            return
        return self.state

    '''
    def _close(self):
        pass

    def _configure(self):
        pass
    '''

    def _seed(self):
        return int(random() * 100)

    def defineState(self):
        # 定义 此时 价位状态
        subject = []
        tarject = []
        try:
            subject = self.input_target[self.inputlist][self.currentTargetIndex - self.scope:self.currentTargetIndex]
            tarject = self.input_target[self.targetlist][self.currentTargetIndex - self.scope:self.currentTargetIndex]
        except Exception as e:
            print(self.targetCode, self.currentTargetIndex, len(self.targetDates))
            self.done = True
        # 已有单价成本，已有数量，建仓时间，建仓名称，历史准确率，
        self.state = (np.array(subject), np.array(tarject))

    def _get_test_data(self):
        # 定义 此时 价位状态
        subject = []
        tarject = []
        try:
            subject = self.input_target_test[self.inputlist]
            tarject = self.input_target_test[self.targetlist]
        except Exception as e:
            print("get data test error")
            self.done = True
        # 已有单价成本，已有数量，建仓时间，建仓名称，历史准确率，
        self.test_state = (np.array(subject), np.array(tarject))
