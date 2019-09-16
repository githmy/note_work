# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import itertools
import math
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.cluster import KMeans

from models.model_trend import TrendNN
from utils.connect_mongo import MongoDB
from utils.connect_mysql import MysqlDB
from utils.log_tool import logger


class PlotTool(object):
    def plot_line(self, ts):
        plt.figure()
        plt.grid()
        listlen = len(ts)
        colorbar = ["red", "yellow", "blue", "black"] * int(math.ceil(listlen / 4))
        for id1 in range(listlen):
            plt.plot(ts[id1][0], ts[id1][1], color=colorbar[id1], label='Original_{}'.format(id1))
        plt.legend(loc='best')
        plt.title('Mean & Standard Deviation')
        plt.show()

    def plot_line_ideal(self, ts):
        plt.figure()
        plt.grid()
        listlen = len(ts)
        colorbar = ["red", "yellow", "blue", "black"] * int(math.ceil(listlen / 4))
        for id1 in range(listlen):
            plt.plot(ts[id1][0], ts[id1][2], color=colorbar[id1], label='Original_{}'.format(id1))
        plt.legend(loc='best')
        plt.title('Mean & Standard Deviation')
        plt.show()


class OutPutResult(object):
    def __init__(self, ):
        # 1. 采集数据源
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

    def get_learn_info_multistudents(self, datalist):
        # 4. 输入：学生的表现详情。输出：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）
        curve_lists = []
        for student_s_list in datalist:
            # print(student_s_list)
            # print(student_s_list["data"])
            curve_lists.append({"datetime": student_s_list["datetime"], "studentid": student_s_list["studentid"],
                                "data": self.get_learn_info_student(student_s_list["data"])})
        return curve_lists

    def get_learn_info_student(self, student_s_list):
        single_curve = []
        for execise_s_list in student_s_list:
            single_curve.append(self.get_learn_info_examination(execise_s_list))
        return single_curve

    def get_learn_info_examination(self, examination_obj):
        # 0:具体课程号，1:相对时间 2:详情
        # 0:概念，1:技能，2:思维
        # 1 知识点维度，2 时间消耗，3 正确率
        point_base_obj = [examination_obj[1], {}]
        # 第一维 学习类型, 第二维 时间, 第三维 正确率
        way_base_obj = {"videinfo": [0, 0.0], "exampleinfo": [0, 0.0], "execiseinfo": [0, 0.0]}
        # 第一维 文件id, 第二维 时间, 第三维 知识点名, 第四维 正确率
        file_base_obj = {}
        # 象限图 按做题的 1. 次序号, 2. k, 3. S, 3. L
        curve_dim_obj = [examination_obj[1], 0, 0, 0, 0, 0, 0]
        # 线性图 按做题的 总分 1. 次序号，2. 得分，3. 总分
        score_p_curve = [examination_obj[1], 0, 0]
        # 三种学习形式
        for tmpobj in examination_obj[2]["videinfo"]:
            if tmpobj["PointCode"] not in point_base_obj[1]:
                point_base_obj[1][tmpobj["PointCode"]] = [0, 0.0, 0.0]
            if tmpobj["SpentTime"] is None:
                tmpobj["SpentTime"] = 0
            point_base_obj[1][tmpobj["PointCode"]][0] += tmpobj["SpentTime"]
            way_base_obj["videinfo"][0] += tmpobj["SpentTime"]
            if tmpobj["FileId"] not in file_base_obj:
                file_base_obj[tmpobj["FileId"]] = [tmpobj["SpentTime"], tmpobj["PointCode"], 0.0]
            file_base_obj[tmpobj["FileId"]][0] += tmpobj["SpentTime"]
        for tmpobj in examination_obj[2]["exampleinfo"]:
            if tmpobj["SpentTime"] is None:
                tmpobj["SpentTime"] = 0
            if tmpobj["PointCode"] not in point_base_obj[1]:
                point_base_obj[1][tmpobj["PointCode"]] = [0, 0.0, 0.0]
            point_base_obj[1][tmpobj["PointCode"]][0] += tmpobj["SpentTime"]
            way_base_obj["exampleinfo"][0] += tmpobj["SpentTime"]
            if tmpobj["ExampleId"] not in file_base_obj:
                file_base_obj[tmpobj["ExampleId"]] = [tmpobj["SpentTime"], tmpobj["PointCode"], 0.0]
            file_base_obj[tmpobj["ExampleId"]][0] += tmpobj["SpentTime"]
        if examination_obj[2]["execiseinfo"] is None:
            curve_dim_obj = None
            score_p_curve = None
        else:
            for tmpobj in examination_obj[2]["execiseinfo"]:
                if tmpobj["spentTime"] is None:
                    tmpobj["spentTime"] = 0
                if tmpobj["mainReviewPoints"][0] not in point_base_obj[1]:
                    point_base_obj[1][tmpobj["mainReviewPoints"][0]] = [0, 0.0, 0.0]
                point_base_obj[1][tmpobj["mainReviewPoints"][0]][0] += tmpobj["spentTime"]
                point_base_obj[1][tmpobj["mainReviewPoints"][0]][1] += tmpobj["actualScore"]
                point_base_obj[1][tmpobj["mainReviewPoints"][0]][2] += tmpobj["score"]
                way_base_obj["execiseinfo"][0] += tmpobj["spentTime"]
                if tmpobj["question"] not in file_base_obj:
                    file_base_obj[tmpobj["question"]] = [tmpobj["spentTime"], tmpobj["mainReviewPoints"][0], 0.0]
                file_base_obj[tmpobj["question"]][0] += tmpobj["spentTime"]
                if tmpobj["qCategory"] == "K":
                    curve_dim_obj[1] += tmpobj["actualScore"]
                    curve_dim_obj[2] += tmpobj["score"]
                elif tmpobj["qCategory"] == "S":
                    curve_dim_obj[3] += tmpobj["actualScore"]
                    curve_dim_obj[4] += tmpobj["score"]
                elif tmpobj["qCategory"] == "L":
                    curve_dim_obj[5] += tmpobj["actualScore"]
                    curve_dim_obj[6] += tmpobj["score"]
            score_p_curve[1] = curve_dim_obj[1] + curve_dim_obj[3] + curve_dim_obj[5]
            score_p_curve[2] = curve_dim_obj[2] + curve_dim_obj[4] + curve_dim_obj[6]
        return {"point_base_obj": point_base_obj, "way_base_obj": way_base_obj, "file_base_obj": file_base_obj,
                "curve_dim_obj": curve_dim_obj, "score_p_curve": score_p_curve}

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

    def cluster_student_quality(self, single_pd_in):
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
        return None

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

    def get_score_curve_data(self, data_reform):
        # 9. 输入：学生素质，曲线基本参数，起始学习时间，知识点评测，返回最合理的预测：当前学生表现的象限能力，输出发展曲线 及标准差。
        score_dim1_curves = []
        score_dim2_curves = []
        for tmpobj in data_reform:
            oneline = copy.deepcopy(tmpobj)
            oneline["data"] = []
            for i1 in tmpobj["data"]:
                tt = i1["score_p_curve"]
                if tt is not None:
                    oneline["data"].append([tt[0], 100 * (tt[1] / tt[2]) if tt[2] != 0 else 0.0])
            multline = copy.deepcopy(tmpobj)
            multline["data"] = []
            for i1 in tmpobj["data"]:
                tt = i1["curve_dim_obj"]
                if tt is not None:
                    if tt[6] == 0:
                        tt[6] = 1e10
                    if tt[2] == 0:
                        tt[2] = 1e10
                    if tt[4] == 0:
                        tt[4] = 1e10
                    multline["data"].append([tt[0], (tt[1] + tt[3]) / (tt[2] + tt[4]), tt[5] / tt[6]])
            score_dim1_curves.append(oneline)
            score_dim2_curves.append(multline)
        return score_dim1_curves, score_dim2_curves

    def get_nearest_pointids(self, origin_point, point_set, findnum=20):
        # 输入：原始点，待比较列表，最近书目  输出：最近点的索引
        pointnum = len(point_set)
        targetnum = findnum
        if findnum > pointnum:
            targetnum = pointnum
        # 1. 获取点的长度
        point_setnp = np.array(point_set)
        origin_pointnp = np.array(origin_point)
        distancelist = (origin_pointnp[0] - point_setnp[:, 0]) ** 2 + (origin_pointnp[1] - point_setnp[:, 1]) ** 2
        ordlist = np.argsort(distancelist)
        return ordlist[0:targetnum]

    # 1.1 生成素质数据
    def gene_quality(self, datalist):
        # 输入：二维数据序列 输出：素质点
        datalistnp = np.array(datalist)
        quality_list = []
        for id2, i2 in enumerate(datalist):
            if id2 != 0:
                dt = datalistnp[id2][0] - datalist[id2 - 1][0]
                quality_list.append([(datalistnp[id2][1] - datalistnp[id2 - 1][1]) / dt,
                                     (datalistnp[id2][2] - datalistnp[id2 - 1][2]) / dt])
        quality_nplist = np.array(quality_list)
        quality_point = np.mean(quality_nplist, axis=0)
        return quality_point

    def gene_curve_data(self, all_data):
        # 1. 转化数据
        def data2np(all_data, pad_lenth=5):
            alldata = copy.deepcopy(all_data)
            paralenth = len(alldata)
            lenlist = np.array([len(i1) for i1 in alldata])
            nplist = []
            nplenlist = []
            for i1 in range(paralenth):
                tmpdata = alldata[i1]
                nplist.append(np.array([np.vstack([np.array(tmpdata), np.ones(((pad_lenth - lenlist[i1]), 2))])]))
                nplenlist.append(np.hstack([np.ones(lenlist[i1]), np.zeros(pad_lenth - lenlist[i1])]))
            npobj = np.vstack(nplist)
            nplenth = np.vstack(nplenlist)
            return npobj, nplenth

        def transpose(matrix):
            return zip(*matrix)

        all_data_transpose = []
        for i1 in all_data:
            all_data_transpose.append(list(transpose(i1)))

        # npobj, len_list = datac2np(accum_curve)
        curvesobj, lenlist = data2np(all_data, 25)
        # 2. 生成模型参数
        model_json = {
            "num_epochs": 50000,
            "learning_rate": 1e-4,
            "evaluate_every": 1000000,
            "early_stop": 20000,
            "save_step": 2000,
        }
        print(len(curvesobj), type(curvesobj))
        print(len(curvesobj[0]), type(curvesobj[0]))
        print(len(curvesobj[0][0]), type(curvesobj[0][0]))
        insmodel = TrendNN("model_type", "model_name", model_json, curvesobj, lenlist)
        insmodel.build()
        insmodel.load_mode("")
        paras = insmodel.fit()

        def get_point_y(x, a, b, c, m):
            xt = x + m
            y = a * xt ** 2 + b * xt + c
            return y

        def get_curve_data(all_data_transpose, paras):
            new_data = []
            for id1, i1 in enumerate(all_data_transpose):
                ty = get_point_y(i1[0], paras[0], paras[1], paras[2], paras[3][id1])
                new_data.append([i1[0] + paras[3][id1], i1[1], ty])
            stand_curvex = [i1 for i1 in range(-100, 100)]
            stand_curvey = [paras[0] * i1 ** 2 + paras[1] * i1 + paras[2] for i1 in range(-100, 100)]
            stand_curve = [[stand_curvex, stand_curvey]]
            return new_data, stand_curve

        # 3. 重整曲线
        new_data, trend_curve = get_curve_data(all_data_transpose, paras)
        return new_data, trend_curve

    def gene_trend_models(self, score_dim2_curves):
        # 输入：知识点累计曲线 输出：模型列表信息
        # 1. 生成报课id 的素质集合,
        # 1.1 生成素质数据
        print(score_dim2_curves)
        quality_points = [self.gene_quality(i1["data"]) for i1 in score_dim2_curves]
        lenth_curves = len(score_dim2_curves)
        # 原索引，删后索引映射
        curve_point_mapid = []
        counter_p = 0
        quality_train_points = []
        for i1 in range(lenth_curves):
            if len(quality_points[i1].shape) != 0:
                curve_point_mapid.append([i1, counter_p])
                quality_train_points.append(copy.deepcopy(quality_points[i1]))
                counter_p += 1
        print(curve_point_mapid)
        quality_train_points = np.array([i1 for i1 in quality_train_points if len(i1.shape) != 0])
        print(quality_train_points)
        print(quality_train_points.shape)
        # 2. 筛选属性相似，图谱接近的 数据
        # model = KMeans(n_clusters=3, random_state=0)
        classn = 3
        n_clusters = quality_train_points.shape[0] // classn
        model = KMeans(n_clusters=n_clusters)
        model.fit(quality_train_points)
        predicted = model.predict(quality_train_points)
        print(predicted)
        # 2.3 获取拟合数据
        # 3. 输出对应数据的拟合曲线
        # 4. kmeans 结合的中心点
        a = None
        return a

    def get_trend_curve_data(self, accum_map, accum_curve, score_dim2_curves, studentid):
        # 输入：知识点累计图谱，知识点累计曲线，学生素质列表，目标学生当前素质,报课id 输出：统计成长曲线
        # print(accum_map)
        # print(accum_curve)
        # print(score_dim2_curves)
        # print(studentid)
        # 0. 临时假数据
        score_dim2_curves[0]["data"] = [[0, 2, 3.5], [3, 5.3, 7.5], [6, 10.2, 13.5]]
        # 1. 获取学生最近一次课程的素质特性
        # 1.1 生成素质数据
        quality_points = [self.gene_quality(i1["data"]) for i1 in score_dim2_curves]
        latestid = -1
        maxdata = "0"
        for id1, i1 in enumerate(score_dim2_curves):
            if i1["studentid"] == studentid and i1["datetime"] > maxdata:
                latestid = id1
        if latestid == -1:
            raise Exception("没有找到该学生{}的数据信息".format(studentid))
        quality_point = self.gene_quality(score_dim2_curves[latestid]["data"])
        # 2. 筛选属性相似，图谱接近的 数据
        # 2.1 筛选素质 fit_ids 为score_dim2_curves的自然序号
        fit_ids = self.get_nearest_pointids(quality_point, quality_points, findnum=10)
        # 2.2 筛选能力 暂时不考虑
        # ability_datas = list(itertools.chain(*[i1['data'] for id1, i1 in enumerate(score_dim2_curves) if id1 in fit_ids]))
        # 2.3 获取拟合数据
        curve_datas = [i1['data'] for id1, i1 in enumerate(score_dim2_curves) if id1 in fit_ids]
        curve1_datas = []
        curve2_datas = []
        for i1 in curve_datas:
            tmp1 = []
            tmp2 = []
            for i2 in i1:
                tmp1.append([i2[0], i2[1]])
                tmp2.append([i2[0], i2[2]])
            curve1_datas.append(tmp1)
            curve2_datas.append(tmp2)
        # 3. 输出对应数据的拟合曲线
        all_data = [
            [[0, 4.5], [2, 7.5]],
            [[0, 3.5], [3, 7.5], [6, 13.5]],
            [[0, 4.2], [7, 12.5]],
            [[0, 0], [4, 2.5]],
        ]
        pprint(all_data)
        pprint(curve1_datas)
        pprint(curve2_datas)
        # new_data, trend_curve = self.gene_curve_data(all_data)
        _, trend1_curve = self.gene_curve_data(curve1_datas)
        _, trend2_curve = self.gene_curve_data(curve2_datas)
        # print(new_data)
        # print(stand_curve)
        # # 4. 显示测试数据
        # insplot = PlotTool()
        # insplot.plot_line(all_data_transpose)
        # insplot.plot_line(stand_curve)
        # insplot.plot_line(new_data)
        # insplot.plot_line_ideal(new_data)
        return trend1_curve, trend2_curve

    def get_point_accum_data(self, data_reform, point_list):
        # 9. 输入：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）输出：按用户测评时间 知识点累计值 分值
        # 9.1 切片数据
        poinsmap = self.get_point_splice_data(data_reform)
        # 9.2 累计数据
        accummap = []
        accumcurve = []
        point_lenth = len(point_list)
        for tmpobj in poinsmap:
            oneline = copy.deepcopy(tmpobj)
            twoline = copy.deepcopy(tmpobj)
            for id1, i1 in enumerate(tmpobj["data"]):
                # 线上的各坐标点
                if id1 != 0:
                    # 上一个坐标点的累计知识点
                    tmpjson = copy.deepcopy(oneline["data"][id1 - 1][1])
                    # 每个坐标点的 每个知识点
                    for i2 in i1[1]:
                        tmpjson[i2] = i1[1][i2]
                    oneline["data"][id1][1] = tmpjson
                # print(oneline["data"][id1][1])
                for _ in oneline["data"][id1][1]:
                    # 不同的计分方式 目前只有累计值，未按权重计数，后期需要加上难度系数。
                    twoline["data"][id1][1] = len(oneline["data"][id1][1]) / point_lenth
                    # twoline["data"][id1][1] = sum(oneline["data"][id1][1].values()) / point_lenth
            accummap.append(oneline)
            accumcurve.append(twoline)
        return accummap, accumcurve

    def get_point_splice_data(self, data_reform):
        # 9. 输入：学生素质，曲线基本参数，起始学习时间，知识点评测，返回最合理的预测：当前学生表现的象限能力，输出发展曲线 及标准差。
        poinsmap = []
        # print(555)
        for tmpobj in data_reform:
            oneline = copy.deepcopy(tmpobj)
            oneline["data"] = []
            for i1 in tmpobj["data"]:
                tmpjson = {}
                for i2 in i1["point_base_obj"][1]:
                    if i1["point_base_obj"][1][i2][2] < 1e-5:
                        vlu = 0
                    else:
                        vlu = i1["point_base_obj"][1][i2][1] / i1["point_base_obj"][1][i2][2]
                    tmpjson[i2] = vlu
                    # print(i1["point_base_obj"][1][i2][1])
                oneline["data"].append([i1["point_base_obj"][0], tmpjson])
            poinsmap.append(oneline)
        return poinsmap


class GetData(object):
    def __init__(self):
        # mysql
        config_my = {
            'host': "192.168.1.252",
            'port': 3306,
            'user': "thinking",
            'password': "thinking2018",
            'database': "htdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
        }
        # config_my = {
        #     'host': "192.168.1.52",
        #     'port': 3306,
        #     'user': "thinking",
        #     'password': "thinking2019",
        #     'database': "htdb",
        #     'charset': 'utf8mb4',  # 支持1-4个字节字符
        # }
        self.insmysql = MysqlDB(config_my)
        # mongo
        config_mon = {
            'host': "192.168.1.252",
            'port': 27017,
            'database': "thinking2ht-test",
            'col': 'col',
            'charset': 'utf8mb4',  # 支持1-4个字节字符
        }
        # config_mon = {
        #     'host': "192.168.1.52",
        #     'port': 27017,
        #     'database': "thinking2ht",
        #     'col': 'col',
        #     'charset': 'utf8mb4',  # 支持1-4个字节字符
        # }
        self.insmongo = MongoDB(config_mon)

    # 2. 筛选课程详情编号
    def get_course_subject_stage_info(self, studentid, subjectid, sectionid, edition, timestart, timeend):
        # 教学相关
        # 课程id， 学习材料id，学生id, 学习日期, 学习形式（学练辅评）, 包含知识点的名称，技能等级，思维等级，学习时间， 建议学习时间
        # sectionid = "aaaa"
        # 试卷id， 题目id，学生id, 考试日期, 包含知识点的名称，技能等级，思维等级， 答题时间， 建议答题时间
        if studentid is None:
            sqls = """
            SELECT studentid, CourseInstanceId, Id as CourseInstanceDtlId, startTime FROM bus_course_instance_dtl WHERE finishtime is not NULL and CourseInstanceId in (
              select Id from bus_course_instance WHERE `Section`='{}' AND startTime>='{}' AND startTime<='{}') ORDER BY StartTime;
            """.format(sectionid, timestart, timeend)
        else:
            sqls = """
            SELECT studentid, CourseInstanceId, Id as CourseInstanceDtlId, startTime FROM bus_course_instance_dtl WHERE finishtime is not NULL and studentid='{}' 
              AND CourseInstanceId in (select Id from bus_course_instance WHERE `Subject`='{}' AND `Section`='{}' AND `Edition`='{}' AND startTime>='{}' AND startTime<='{}') ORDER BY StartTime;
            """.format(studentid, subjectid, sectionid, edition, timestart, timeend)
        # subjectid,
        # print(sqls)
        return self.insmysql.exec_sql(sqls)

    def get_course_video_info(self, coursedtlid):
        sqls = """
        SELECT FileId, PointCode, SpentTime FROM bus_course_file WHERE CourseDtlId='{}';""".format(coursedtlid)
        # print(sqls)
        return self.insmysql.exec_sql(sqls)

    def get_course_example_info(self, coursedtlid):
        sqls = """
        SELECT ExampleId, PointCode, SpentTime FROM bus_course_example WHERE CourseDtlId='{}';""".format(coursedtlid)
        return self.insmysql.exec_sql(sqls)

    def get_course_execise_info(self, coursedtlid):
        sqls = """
        SELECT CourseDtlId, QuestionId, PointCode FROM bus_course_question WHERE CourseDtlId='{}';""".format(
            coursedtlid)
        questioninfos = self.insmysql.exec_sql(sqls)
        return self.get_course_question_info(questioninfos)

    def get_course_question_info(self, questioninfos):
        # 1. 找出一次课程的所有习题id
        filter_dic = {"courseId": {"$in": [i1["CourseDtlId"] for i1 in questioninfos]}}
        show_dic = {"exercises": 1, "_id": 0}
        sql_tuple = (filter_dic, show_dic)
        colname = "studentassignments"
        execise_dtl = self.insmongo.exec_require(sql_tuple, colname)
        execise_dtl = list(execise_dtl)
        if len(execise_dtl) == 0:
            return None
        # 2. 找出题目详情
        colname = "studentexercises"
        filter_dic = {"_id": {"$in": list(itertools.chain(*[i1["exercises"] for i1 in execise_dtl]))}}
        show_dic = {"question": 1, "mainReviewPoints": 1, "reviewPoints": 1, "spentTime": 1, "mode": 1, "qCategory": 1,
                    "score": 1, "actualScore": 1, "_id": 0}
        sql_tuple = (filter_dic, show_dic)
        return list(self.insmongo.exec_require(sql_tuple, colname))

    def get_course_single_info(self, coursedtlid):
        # 2.1 找出测验id
        videinfo = self.get_course_video_info(coursedtlid)
        exampleinfo = self.get_course_example_info(coursedtlid)
        execiseinfo = self.get_course_execise_info(coursedtlid)
        return {"videinfo": videinfo, "exampleinfo": exampleinfo, "execiseinfo": execiseinfo}

    # 获取统计数据主要
    def get_course_result_info(self, courseinfo):
        # 筛选课程id 下的数据 整理成标准形式 [{起始时差:datatime, 课时id:xx, 知识点列表:[{知识点名称:xx, 学习时间:xx, 学习形式:xx, 成绩分值:xx, 成绩类型:xx}]}]
        # 1. 得出主列表Id
        cm_list = set([i1['CourseInstanceId'] for i1 in courseinfo])
        cd_dict = []
        for i1 in cm_list:
            tmp_cd_list = []
            tmptimelist = []
            tmppointinfo = []
            for i2 in courseinfo:
                if i2['CourseInstanceId'] == i1:
                    tmp_cd_list.append(i2['CourseInstanceDtlId'])
                    tmptimelist.append(i2['startTime'])
                    tmppointinfo.append(self.get_course_single_info(tmp_cd_list[-1]))
            tmptimelist = [(i2 - tmptimelist[0]).days for i2 in tmptimelist]
            cd_dict.append({"datetime": str(courseinfo[0]["startTime"]),
                            "studentid": courseinfo[0]["studentid"],
                            "data": list(zip(tmp_cd_list, tmptimelist, tmppointinfo))})
        # # [{课时id:xx, 起始时差:datatime, 知识评测完善度, 知识点列表:[{知识点名称:xx, 学习时间:xx, 学习形式:xx, 成绩分值:xx, 成绩类型:xx}]}]
        return cd_dict

    def get_quiz_subject_stage_info(self, subjectid, sectionid, edition, timestart, timeend):
        # 2. 答题相关
        # 2.1 找出测验id
        sectionid = "七年级上学期"
        # 试卷id， 题目id，学生id, 考试日期, 包含知识点的名称，技能等级，思维等级， 答题时间， 建议答题时间
        sqls = """
        SELECT Id FROM bus_course_instance_dtl WHERE CourseInstanceId in (
          select Id from bus_course_instance WHERE Title='{}' AND startTime>='{}' AND startTime<='{}');
        """.format(sectionid, timestart, timeend)
        # subjectid,
        cousids = self.insmysql.exec_sql(sqls)
        print(cousids)
        # 2.2 找出测验
        colname = "studentassignments"
        filter_dic = {"courseId": {"$in": [i1["Id"] for i1 in cousids]}}
        show_dic = {"exercises": 1, "_id": 0}
        sql_tuple = (filter_dic, show_dic)
        tts = self.insmongo.exec_require(sql_tuple, colname)
        # 2.3 找出题目详情
        colname = "studentexercises"
        filter_dic = {"testingTime": {"$gte": parser.parse(timestart), "$lte": parser.parse(timeend)},
                      "_id": {"$in": list(itertools.chain(*[i1["exercises"] for i1 in list(tts)]))}}
        show_dic = None
        sql_tuple = (filter_dic, show_dic)
        return self.insmongo.exec_require(sql_tuple, colname)

    def get_single_quiz_subject_stage_info(self, studentid, subjectid, sectionid, edition, timestart, timeend):
        # 2. 答题相关
        # 试卷id, 题目id, 学生id, 考试日期, 包含知识点的名称, 技能等级, 思维等级, 答题时间, 建议答题时间
        sqls = """
        select * from bus_course_instance WHERE StudentId='{}' and Title='{}' AND startTime>='{}' AND startTime<='{}';
        """.format(subjectid, sectionid, timestart, timeend)
        return self.insmysql.exec_sql(sqls)

    def get_student_info(self, subjectid, sectionid, edition):
        # 3. 背景信息
        # 学生的学校等级, 父母教育程度, 其它业余培训班类型，其它业余培训时间(多个培训班分别算)
        sqls = """
        SELECT * FROM bus_account_info WHERE `Section`="{}" AND Edition IN (
            SELECT `Value` FROM sys_dictdetail WHERE `Name`='{}');
        """.format(sectionid, edition)
        # subjectid,
        return self.insmysql.exec_sql(sqls)

    def get_subject_stage_point_info(self, subjectid, sectionid, edition):
        # 4. 学期的知识点
        # sys_dictdetail 教材版本id, --> 学段相应的章节 --> 章节相应的知识点
        sqls = """
        SELECT PointCode FROM bus_chapter_point WHERE ChapterId IN (
            SELECT Id FROM bus_chapter_info WHERE BookId IN (
                    SELECT Id FROM bus_book_info WHERE `subject`='{}' AND `Section`='{}' AND Edition IN (
                        SELECT `Value` FROM sys_dictdetail WHERE `Name`='{}')));
        """.format(subjectid, sectionid, edition)
        return self.insmysql.exec_sql(sqls)


def main():
    subjectid = "M"
    sectionid = "J"
    edition = "沪教版"
    timestart = "2019-01-01 00:00:00"
    timeend = "2019-09-17 00:00:00"
    insdata = GetData()
    # # 0. 单个学生的课程信息 studentid is not None
    # studentid = "6e3fedf0-33ab-4315-ba1b-31809d160c06"
    # coursedetailkeys = insdata.get_course_subject_stage_info(studentid, subjectid, sectionid, edition, timestart,
    #                                                          timeend)
    # single_course_res_info = insdata.get_course_result_info(coursedetailkeys)
    # 1. 筛选课程相关id  [{课程id:xx, 课程时间:xx}]
    studentid = None
    coursedetailkeys = insdata.get_course_subject_stage_info(studentid, subjectid, sectionid, edition, timestart,
                                                             timeend)
    # 2. 筛选课程id 下的数据 整理成标准形式 [{起始时差:datatime, 课时id:xx, 知识点列表:[{知识点名称:xx, 学习时间:xx, 学习形式:xx, 成绩分值:xx, 成绩类型:xx}]}]
    course_res_info = insdata.get_course_result_info(coursedetailkeys)
    # print(course_res_info)
    # quizres = insdata.get_quiz_subject_stage_info(subjectid, sectionid, edition, timestart, timeend)
    # 3. 学生相关信息    todo: 需要补充：是否根据 学科 学段 过滤，还是从 课程主表里筛选。
    studentinfo = insdata.get_student_info(subjectid, sectionid, edition)
    # print(studentinfo)
    # 4. 学期的知识点 ok
    pointobj = insdata.get_subject_stage_point_info(subjectid, sectionid, edition)
    datadic = {"course_res_info": course_res_info, "studentinfo": studentinfo,
               "pointobj": [i1["PointCode"] for i1 in pointobj]}
    # pprint(datadic)

    # 2. 分析返回
    insres = OutPutResult()
    # 2.1 输入：学生的表现详情。输出：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）
    data_reform = insres.get_learn_info_multistudents(datadic["course_res_info"])
    # pprint(data_reform)
    # 2.2 输入：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）输出：按用户测评时间 分值维度 分值线
    score_dim1_curves, score_dim2_curves = insres.get_score_curve_data(data_reform)
    # pprint(score_dim2_curves)
    # 2.3 输入：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）输出：按用户测评时间 知识点累计值(图谱，曲线) 分布
    accum_map, accum_curve = insres.get_point_accum_data(data_reform, datadic["pointobj"])

    # 2.4 输入：知识点累计图谱，知识点累计曲线，学生素质列表，目标学生当前素质,报课id 输出：统计成长曲线

    def datac2np(accum_curve, pad_lenth=5):
        print(accum_curve)
        alldata = copy.deepcopy(accum_curve)
        paralenth = len(alldata)
        pad_lenth = 5
        lenlist = np.array([len(i1["data"]) for i1 in alldata])
        nplist = []
        nplenlist = []
        for i1 in range(paralenth):
            tmpdata = alldata[i1]["data"]
            nplist.append(np.array([np.vstack([np.array(tmpdata), np.ones(((pad_lenth - lenlist[i1]), 2))])]))
            nplenlist.append(np.hstack([np.ones(lenlist[i1]), np.zeros(pad_lenth - lenlist[i1])]))
        npobj = np.vstack(nplist)
        nplenth = np.vstack(nplenlist)
        return npobj, nplenth

    studentid = "6e3fedf0-33ab-4315-ba1b-31809d160c06"
    trend1_curve, trend2_curve = insres.get_trend_curve_data(accum_map, accum_curve, score_dim2_curves, studentid)
    print(11112)
    pprint(trend1_curve)
    pprint(trend2_curve)
    # todo: 1. 根据 学生的 表现 做 推题，预测
    # 2.3 按用户当前 知识点累计值分值 和 用户的素质 筛选类似的数据
    # 2.4 按用户 的错题 和 知识图谱的顺序，倒推数据
    # 2.5 知识点:难度 == 得分/消耗的时间
    # 由于题目的思维难度根据步骤来，是客观的，不需要人为干预（暂时没考虑不同解决方案的步骤不同，即一道题多个解法的情况）


def trend_back_interface(subjectid="M", sectionid="J", timestart="2019-01-01 00:00:00",
                         timeend="2019-09-17 00:00:00", edition="沪教版", studentid="aaaaa"):
    insdata = GetData()
    # # 0. 单个学生的课程信息 studentid is not None
    # studentid = "6e3fedf0-33ab-4315-ba1b-31809d160c06"
    # coursedetailkeys = insdata.get_course_subject_stage_info(studentid, subjectid, sectionid, edition, timestart,
    #                                                          timeend)
    # single_course_res_info = insdata.get_course_result_info(coursedetailkeys)
    # 1. 筛选课程相关id  [{课程id:xx, 课程时间:xx}]
    studentid = None
    coursedetailkeys = insdata.get_course_subject_stage_info(studentid, subjectid, sectionid, edition, timestart,
                                                             timeend)
    # 2. 筛选课程id 下的数据 整理成标准形式 [{起始时差:datatime, 课时id:xx, 知识点列表:[{知识点名称:xx, 学习时间:xx, 学习形式:xx, 成绩分值:xx, 成绩类型:xx}]}]
    course_res_info = insdata.get_course_result_info(coursedetailkeys)
    # print(course_res_info)
    # quizres = insdata.get_quiz_subject_stage_info(subjectid, sectionid, edition, timestart, timeend)
    # 3. 学生相关信息    todo: 需要补充：是否根据 学科 学段 过滤，还是从 课程主表里筛选。
    studentinfo = insdata.get_student_info(subjectid, sectionid, edition)
    # print(studentinfo)
    # 4. 学期的知识点 ok
    pointobj = insdata.get_subject_stage_point_info(subjectid, sectionid, edition)
    datadic = {"course_res_info": course_res_info, "studentinfo": studentinfo,
               "pointobj": [i1["PointCode"] for i1 in pointobj]}
    # pprint(datadic)

    # 2. 分析返回
    insres = OutPutResult()
    # 2.1 输入：学生的表现详情。输出：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）
    data_reform = insres.get_learn_info_multistudents(datadic["course_res_info"])
    # pprint(data_reform)
    # 2.2 输入：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）输出：按用户测评时间 分值维度 分值线
    score_dim1_curves, score_dim2_curves = insres.get_score_curve_data(data_reform)
    # pprint(score_dim2_curves)
    # 2.3 输入：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）输出：按用户测评时间 知识点累计值(图谱，曲线) 分布
    accum_map, accum_curve = insres.get_point_accum_data(data_reform, datadic["pointobj"])

    # 2.4 输入：知识点累计图谱，知识点累计曲线，学生素质列表，目标学生当前素质,报课id 输出：统计成长曲线
    studentid = "6e3fedf0-33ab-4315-ba1b-31809d160c06"
    trend1_curve, trend2_curve = insres.get_trend_curve_data(accum_map, accum_curve, score_dim2_curves, studentid)
    print(11112)
    pprint(trend1_curve)
    pprint(trend2_curve)
    return trend1_curve, trend2_curve


def recommand_back_interface(subjectid="M", sectionid="J", timestart="2019-01-01 00:00:00",
                             timeend="2019-09-17 00:00:00", points=[], edition="沪教版", studentid="aaaaa"):
    print(points)
    print(type(points))
    datas = None
    return datas


def model_back_interface(subjectid="M", sectionid="J", timestart="2019-01-01 00:00:00", timeend="2019-09-17 00:00:00",
                         edition="沪教版"):
    # 1. 获取素质集合
    insdata = GetData()
    # 1. 筛选课程相关id  [{课程id:xx, 课程时间:xx}]
    studentid = None
    coursedetailkeys = insdata.get_course_subject_stage_info(studentid, subjectid, sectionid, edition, timestart,
                                                             timeend)
    course_res_info = insdata.get_course_result_info(coursedetailkeys)
    # print(len(course_res_info))
    # quizres = insdata.get_quiz_subject_stage_info(subjectid, sectionid, edition, timestart, timeend)
    # 4. 学期的知识点 ok
    pointobj = insdata.get_subject_stage_point_info(subjectid, sectionid, edition)
    # 2. 数据预处理
    insres = OutPutResult()
    # 2.1 输入：学生的表现详情。输出：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）
    data_reform = insres.get_learn_info_multistudents(course_res_info)
    # pprint(data_reform)
    # 2.2 输入：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）输出：按用户测评时间 分值维度 分值线
    score_dim1_curves, score_dim2_curves = insres.get_score_curve_data(data_reform)
    # pprint(score_dim1_curves)
    # pprint(score_dim2_curves)
    # 每个course 抽取成不同 quality
    # # 2.3 输入：不同维度的曲线列表（知识点 方式 文件 多维得分 得分）输出：按用户测评时间 知识点累计值(图谱，曲线) 分布
    # accum_ma, accum_curve = insres.get_point_accum_data(data_reform, [i1["PointCode"] for i1 in pointobj])
    # 2. 聚类
    insres.gene_trend_models(score_dim2_curves)
    # 3. 生成参数
    # 4. 写入中心坐标，参数 (sujectid_sectionid_edition_x_y)
    return True


if __name__ == '__main__':
    logger.info("".center(100, "*"))
    logger.info("welcome to Delphis".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    model_back_interface()
    exit(0)
    main()
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
