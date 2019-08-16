# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pandas as pd
from server import Delphis
from models.model_cnn import TextCNN
from utils.data_trans import data2js
from utils.log_tool import logger
from sklearn.cluster import KMeans
import os


class OutPutResult():
    def __init__(self):
        self.student_quality = [
            {"学生id": 1, "skill": 1, "thinking": 1, "hobby": 1, "school_type": 1, "student_point": 1}]
        self.point_info = [{"知识点": 1, "技能水平": 1, }]
        self.question_detial = [{"题目id": 1, "知识点": 1, "技能水平": 1, "建议时间": 1}]
        self.question_main = [{"题目id": 1, "思维水平": 1, "技能水平": 1}]
        self.quiz_detail = [{"学生id": 1, "题目id": 1, "测试时间": 1, "知识点id": 1, "答题时长": 1}]
        self.quiz_main = [{"测试id": 1, "学生id": 1, "试卷id": 1, "测试时间": 1, "skill": 1, "thinking": 1}]
        self.quiz_point = [
            {"测试id": 1, "知识点id": 1, "know": 1, "skill": 1, "thinking": 1, "答点时长": 1, "建议时长": 1.5, "是否超前": 1}]
        self.learing_info = [{"学生id": 1, "学习时间": 1, "知识点": 1, "学习时长": 1}]
        self.triple_info = [{"主知识点id": 1, "关系id": 1, "客知识点id": 1, "场景id": 1}]
        self.pd_student_quality = pd.DataFrame(self.student_quality)
        self.pd_point_info = pd.DataFrame(self.point_info)
        self.pd_question_detial = pd.DataFrame(self.question_detial)
        self.pd_question_main = pd.DataFrame(self.question_main)
        self.pd_quiz_detail = pd.DataFrame(self.quiz_detail)
        self.pd_quiz_main = pd.DataFrame(self.quiz_main)
        self.pd_quiz_point = pd.DataFrame(self.quiz_point)
        self.pd_learing_info = pd.DataFrame(self.learing_info)
        self.pd_triple_info = pd.DataFrame(self.triple_info)

    def update_point_skill_level(self):
        # 1. 输入：所有人 历次 学习 测试的平均时间消耗 和错误率，及相关知识点，输出：更新题目和知识点的技能难度等级。
        pass

    def points_teach_weigh(self):
        # 2. 输入：学习详情，历次测试，所属素质群体，输出：某素质群体，学习方式的比例。
        pass

    def points_file_weigh(self):
        # 3. 输入：学习详情，历次测试，所属素质群体，输出：知识点更新资料表的权重。（文件过多，可能不支持）
        pass

    def pre_student_quality(self):
        # 4. 根据，学校，业余安排(整合到素质里)，以及初始测试跟0点的对比，初始化学生的素质属性（0点有多高，需要知道学生是否提前预习过该学期的课程）。
        pass

    def _lines2rank(self):
        # 知识点列变为行
        self.pd_quiz_point["efficient_rate"] = self.pd_quiz_point["建议时长"] / self.pd_quiz_point["答点时长"]
        self.pd_quiz_point["S"] = self.pd_quiz_point["skill"] * self.pd_quiz_point["efficient_rate"]
        self.pd_quiz_point["T"] = self.pd_quiz_point["thinking"] * self.pd_quiz_point["efficient_rate"]
        print(self.pd_quiz_point)
        # groupby分组操作
        reslist = []
        for name, group in self.pd_quiz_point.groupby('测试id'):
            print('*' * 13 + name + '*' * 13 + '\n', group)
            print()
            tmpjson = {"quizid": name}
            for i2 in self.pd_point_info["知识点"]:
                if i2 in group["知识点id"]:
                    ind = group["知识点id"].index(i2)
                    tmpjson[i2 + "_K"] = group.iloc[ind, "K"]
                    tmpjson[i2 + "_T"] = group.iloc[ind, "T"]
                else:
                    tmpjson[i2 + "_K"] = 0
                    tmpjson[i2 + "_T"] = 0
            reslist.append(tmpjson)
        res_pd = pd.DataFrame(reslist)
        return res_pd

    def _single_lines2rank(self, pd_in):
        # 知识点列变为行
        reslist = []
        tmpjson = {"quizid": pd_in.iloc[0, "测试id"]}
        for i2 in self.pd_point_info["知识点"]:
            if i2 in pd_in["知识点id"]:
                ind = pd_in["知识点id"].index(i2)
                tmpjson[i2 + "_K"] = pd_in.iloc[ind, "K"]
                tmpjson[i2 + "_T"] = pd_in.iloc[ind, "T"]
            else:
                tmpjson[i2 + "_K"] = 0
                tmpjson[i2 + "_T"] = 0
        reslist.append(tmpjson)
        res_pd = pd.DataFrame(reslist)
        return res_pd

    def update_student_quality(self, single_pd_in):
        # 5. 输入：历次测试2个维度的知识点降维结果：更新学生的素质属性。（为了得出发展轨迹）
        lines_pd = self._lines2rank()
        # 1. 统计 同一科目 同一学段同一学期 的 各个报名时间 各个学生 的发展轨迹，做学习模式聚类，作为背景参照。
        cls = KMeans(n_clusters=4, init='k-means++')
        y_hat = cls.fit_predict(lines_pd)
        # 2. 选   某一科目 某一学段某一学期 的 某个报名时间 某个学生 的评测结果。
        single_pd = self._single_lines2rank(single_pd_in)
        single_y_hat = cls.fit(single_pd)
        # 3. 属于 查看该学生属于某个类
        # 4. 画出 该科目 该学段该学期 该聚类 的 均值轨迹和波动值，外加拟合延长线

    def update_point_weigh(self):
        # 6. 输入：题库的统计结果，输出：所有相关的知识点按权重排序（同权重的或全选或全不先，题目出现多就代表重要）。
        pass

    def _get_point_parent(self):
        # 关系为 词"属于"，词域为"哲学概念"的id。场景 为词"考试知识点"，词域为"人事"的id。
        # 1. 对学生做错误知识点的聚类，
        # 2. 取最近的10个人离散到知识图谱上，
        # 3. 以错误基点为基础倒推父级下的有标记知识点
        pass

    def recommand_wrong_points(self):
        # 7. 输入：知识点权重，评测结果，对比同类错误学生(未必是同素质学生，只考虑错点，不考虑数量，取10个测试样本，离散到知识图谱上，以错误基点为基础倒退父级下的有标记知识点)，输出：同样错误的知识点。
        pass

    def output_general_curve_paras(self):
        # 统计准确率保持在95%以上，否则要减小素质群体的分类
        # 8. 输入：不同素质群体，不同时间(规范化学期开始为时间0)综合降维知识点的掌握情况，输出：不同素质群体，2个维度的曲线的拟合参数方差。
        pass

    def output_curve(self):
        # 9. 输入：学生素质，曲线基本参数，起始学习时间，知识点评测，返回最合理的预测：当前学生表现的象限能力，输出发展曲线 及标准差。
        pass


def main():
    ins = OutPutResult()
    ins.update_student_quality()
    single_pd_in = pd.DataFrame()
    ins.update_student_quality(single_pd_in)
    # 由于题目的思维难度根据步骤来，是客观的，不需要人为干预（暂时没考虑不同解决方案的步骤不同，即一道题多个解法的情况）


if __name__ == '__main__':
    logger.info("".center(100, "*"))
    logger.info("welcome to Delphis".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    main()
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
