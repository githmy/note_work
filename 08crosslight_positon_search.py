# coding:utf-8
import os
import pandas as pd
from sklearn.cluster import KMeans
import time
import copy
import json
import jsonpath
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc_special
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
from collections import OrderedDict
from sklearn.manifold import TSNE
import math
import shutil
from multiprocessing import Process
from multiprocessing import Pool
import multiprocessing
import re


def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))


class ParaSearch_bak(object):
    def __init__(self, fit_func, iter_json, exe_json, project_path, run_file_list, parakeys):
        """
        生成位置，迭代疏远非常近的。
        ---------------------------------------------------
        Input parameters:
            cluster_num: Number of nests
            lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
            upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
            dim_sensitive: 维度的敏感性,越大切分的越细 -- 某维度切分=(upper_boundary-lower_boundary)/dim_sensitive
            exe_json: 磁盘根目录
            project_path: 磁盘全路径
            run_file_list: 运行文件顺序
            parakeys: 参数顺序
        Output:
            generated nests' locations
        """
        self.fit_func = fit_func
        self.iter_json = iter_json
        self.exe_json = exe_json
        self.project_path = project_path
        self.run_file_list = run_file_list
        self.parakeys = parakeys

        # 解聚类个数
        self.cluster_num = self.iter_json["cluster_num"]
        # 参数下边界
        self.lower_boundary = np.array(self.iter_json["lower_boundary"])
        # 参数上边界
        self.upper_boundary = np.array(self.iter_json["upper_boundary"])
        # 参数敏感度
        self.dim_sensitive = np.array(self.iter_json["dim_sensitive"])
        # 维度标量大小
        self.dim_scalar = self.upper_boundary - self.lower_boundary
        # 维度数
        self.m = len(self.lower_boundary)
        # ****** 计算部分 ********
        # 迭代标准化起始步长
        self.init_step = self.iter_json["init_step"]
        # theta是参数，梯度下降通过不断更新theta的值使损失函数达到最小值
        self.now_theta = np.zeros((self.cluster_num, self.m))
        self.new_theta = np.zeros((self.cluster_num, self.m))
        self.old_dx = np.zeros((self.cluster_num, self.m))
        self.now_dx = np.zeros((self.cluster_num, self.m))
        self.grade = np.zeros((self.cluster_num))
        self.new_grade = np.zeros((self.cluster_num))
        # 聚类id合并百分比阈值
        self.merge_percent = self.iter_json["merge_percent"]
        # 标准初始位置
        self.init_normal_positions = None
        # now_normal_posi
        self.now_normal_positions = None
        # normal_posi
        self.old_normal_positions = None
        # 聚类id
        self.cluster_id = None
        # 原始聚类id距离矩阵
        self.ori_distance_matric = np.zeros((self.cluster_num, self.cluster_num))
        self.distance_near = np.zeros((self.cluster_num))
        # oldstepdis
        self.old_step_dist = np.zeros((self.cluster_num))
        self.now_step_dist = np.zeros((self.cluster_num))
        # 初始参数
        self.generate_positions()
        self.reference_olddata()
        # 加载历史记录
        self.load_history()

    def load_history(self):
        loadfile = os.path.join(self.project_path, 'template.csv')
        # 聚类id 当前得分
        self.now_target_Y = np.zeros((self.cluster_num))
        self.old_target_Y = np.zeros((self.cluster_num))
        # 聚类id状态：正常 合并 结束
        self.status_sig = ["正常"] * self.cluster_num
        # accur_posi
        self.target_X = np.zeros((self.cluster_num, self.m))
        self.next_cluster_id = 0
        if os.path.isfile(loadfile):
            # if 0:
            print("loadlog: {}".format(loadfile))
            # 1. 数据加载
            pdobj = pd.read_csv(loadfile, header=0, encoding="gbk")
            print("contents:", pdobj)
            self.result_json = json.loads(pdobj.to_json(orient='records', force_ascii=False), encoding="gbk")
            for item in self.result_json:
                item["old_normal_posi"] = json.loads(item["old_normal_posi"])
                item["now_normal_posi"] = json.loads(item["now_normal_posi"])
                item["now_theta"] = json.loads(item["now_theta"])
                item["new_theta"] = json.loads(item["new_theta"])
                item["accur_posi"] = json.loads(item["accur_posi"])
                item["old_dx"] = json.loads(item["old_dx"])
                item["now_dx"] = json.loads(item["now_dx"])

            # 2. 变量初始化
            iterset = set()
            for idn in range(self.cluster_num):
                tmpjson = [item for item in self.result_json if item["ori_cluster_id"] == idn]
                tmpjson = tmpjson[-1:]
                # new马上会被copy到now
                if len(tmpjson) > 0:
                    self.old_normal_positions[idn] = tmpjson[-1]["old_normal_posi"]
                    self.now_normal_positions[idn] = tmpjson[-1]["now_normal_posi"]
                    self.now_theta[idn] = np.array(tmpjson[-1]["now_theta"])
                    self.new_theta[idn] = np.array(tmpjson[-1]["new_theta"])
                    self.old_target_Y[idn] = tmpjson[-1]["old_score"]
                    self.now_target_Y[idn] = tmpjson[-1]["now_score"]
                    self.old_dx[idn] = tmpjson[-1]["old_dx"]
                    self.now_dx[idn] = tmpjson[-1]["now_dx"]
                    self.status_sig[idn] = tmpjson[-1]["status"]
                    self.grade[idn] = tmpjson[-1]["grade"]
                    self.new_grade[idn] = tmpjson[-1]["new_grade"]
                    self.old_step_dist[idn] = tmpjson[-1]["old_step_dist"]
                    self.now_step_dist[idn] = tmpjson[-1]["now_step_dist"]
                    # self.theta_dis[idn] = tmpjson[-1]["theta_dis"]
                    # self.dy[idn] = tmpjson[-1]["dy"]
                    iterset.add(tmpjson[-1]["iter_id"])
                else:
                    self.iter_id = 0
                    break
            self.result_id = self.result_json[-1]["id"] + 1
            # 筛选出 每个类 最大的id，且为正常的。如果对应的 iter_id 一致 且 result 大于等于1轮cluster num，直接+1
            self.iter_id = self.result_json[-1]["iter_id"]
            if len(iterset) == 1 and len(self.result_json) >= self.cluster_num:
                self.iter_id += 1
            self.next_cluster_id = (self.result_json[-1]["ori_cluster_id"] + 1) % self.cluster_num
        else:
            self.result_json = []
            self.result_id = 0
            self.iter_id = 0

    def generate_positions(self):
        # kmeans 生成位置, 用10倍于聚类数的随机点, 加入敏感度后聚类，然后每个类取中心
        # 1. 生成随机样本
        samples = np.random.rand(self.cluster_num * 10, self.m)
        # 按敏感度 scalar
        samples = samples * self.dim_sensitive
        # 2. 聚类，得出中心点。
        cls = KMeans(n_clusters=self.cluster_num, init='k-means++')
        cls.fit_predict(samples)
        self.init_normal_positions = cls.cluster_centers_
        self.now_normal_positions = cls.cluster_centers_
        self.old_normal_positions = cls.cluster_centers_
        self.cluster_id = cls.predict(cls.cluster_centers_)
        for i1 in range(self.cluster_num):
            for i2 in range(self.cluster_num):
                if i1 == i2:
                    self.ori_distance_matric[i1, i2] = np.float(9999)
                    continue
                tmpv = self.now_normal_positions[i1] - self.now_normal_positions[i2]
                self.ori_distance_matric[i1, i2] = np.sqrt(np.sum(tmpv * tmpv))
        self.distance_near = np.min(self.ori_distance_matric, 1)
        self.now_step_dist = np.sqrt(np.sum(np.square(self.now_normal_positions), 1))
        self.now_dx = self.now_normal_positions - 0

    def reference_olddata(self):
        # 参考老点，如果下一步位置与老点的位置小于，敏感度规的10%，用老点编号，否则用新点
        pass

    def one_iter_adam(self):
        """
        F(X) = F(X0) + F'(X0) * (X - X0) = 0
        J(X0) * H = B
        J(X0) == F'(X0) 雅克比矩阵
        (X - X0) == H  残余量
        B == -F(X0) 方程残余量

        X0 -> J, B 
        J * H = B -> H  
        X = H + X0
        X= H + X0 改
        X= damping_step * H + X0
        :return: 
        """
        # 一次迭代, 尺度含义还原成实际尺寸
        # 步长自适应
        min_sensi = np.min(self.dim_sensitive)
        min_dist = np.min(self.distance_near)
        dim_scalar = min_sensi * min_dist
        self.old_normal_positions = copy.deepcopy(self.now_normal_positions)
        self.now_theta = copy.deepcopy(self.new_theta)
        self.old_step_dist = copy.deepcopy(self.now_step_dist)
        self.old_dx = copy.deepcopy(self.now_dx)
        self.grade = copy.deepcopy(self.new_grade)
        self.target_X = self.lower_boundary + self.old_normal_positions / self.dim_sensitive * self.dim_scalar
        for idn in range(self.cluster_num):
            if idn != self.next_cluster_id:
                continue
            self.next_cluster_id = (self.next_cluster_id + 1) % self.cluster_num
            if self.status_sig[idn] != "正常":
                continue
            # 1. 遍历每一个类, 生成对应的结果
            sstart = time.time()
            print("class:", idn)
            # print("456")
            # print(self.now_target_Y[idn])
            self.old_target_Y[idn] = self.now_target_Y[idn]
            self.now_target_Y[idn] = self.fit_func(self.target_X[idn], self.exe_json, self.project_path,
                                                   self.run_file_list,
                                                   self.parakeys)
            print(self.now_target_Y[idn])
            # 时间以分钟为单位
            usetime = (time.time() - sstart) / 60
            print("usetime: {}mins".format(usetime))
            # 2. 生成新位置 梯度下降过程 用当下theta，保存下一步theta, 可以直接调用
            dy = self.now_target_Y[idn] - self.old_target_Y[idn]
            dx_list = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).old_dx]".format(idn))
            theta_list = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).now_theta]".format(idn))
            dis_list = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).old_step_dist]".format(idn))
            # sig_list = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).status]".format(idn))
            if dx_list:
                old_dx = np.array(dx_list[-1])
                old_theta = np.array(theta_list[-1])
                old_dis = dis_list[-1]
            else:
                old_dx = np.zeros((self.m))
                old_theta = np.zeros((self.m))
                old_dis = np.sqrt(np.sum(np.square(self.init_normal_positions[idn])))
            # 2.1 曲面导数变化经验化
            add_dx_scalar = np.sqrt(np.sum(np.square(old_dx + self.old_dx[idn])))
            self.new_grade[idn] = dy / self.old_step_dist[idn]
            self.new_theta[idn] = self.new_grade[idn] * self.old_dx[idn]
            max_theta_dis = np.sqrt(np.sum(np.square(self.new_theta[idn])))
            self.new_theta[idn] = self.new_theta[idn] / max_theta_dis if max_theta_dis > 1 else self.new_theta[idn]
            # 综合 最近两次的 theta
            max_theta = (self.new_theta[idn] / self.old_step_dist[idn] + old_theta / old_dis) * add_dx_scalar
            max_theta_dis = np.sqrt(np.sum(np.square(max_theta)))
            # 随机改变方向和步长, 20%
            rdxy = np.random.rand(self.m)
            # 指方向占90%， 10%近似垂直方向随机，均值后只占50%权重
            new_theta1 = max_theta * rdxy
            new_theta2 = np.sign(max_theta) * (max_theta_dis - np.abs(max_theta)) * rdxy
            # 随机防止永远下一步在直线上
            # max_theta = 1.8 * new_theta1 + 0.2 * new_theta2
            # max_theta = 1.6 * new_theta1 + 0.4 * new_theta2
            max_theta = 1.2 * new_theta1 + 0.8 * new_theta2
            # max_theta_dis = np.sqrt(np.sum(np.square(max_theta)))
            # 1. 限制导数
            # max_theta = max_theta / max_theta_dis if max_theta_dis > 1 else max_theta
            real_thetadis = np.sqrt(np.sum(np.square(max_theta)))
            # 2. 最大步长
            max_step = 1.0
            # real_thetadis = real_thetadis / abs(real_thetadis) * ((abs(real_thetadis) - max_step) / 10 + max_step)
            step_scalar = max_step / real_thetadis
            # 3. 限制 step_scalar * real_thetadis 的最小值
            min_step = 0.1
            # print(step_scalar, real_thetadis, step_scalar * real_thetadis)
            step_scalar = min_step / real_thetadis if step_scalar * real_thetadis < min_step else step_scalar
            if step_scalar * real_thetadis < min_step:
                step_scalar = (step_scalar - min_step / real_thetadis) * 10 + min_step / real_thetadis
            # 4. 新位置
            self.now_normal_positions[idn] = self.old_normal_positions[
                                                 idn] - self.init_step * dim_scalar * step_scalar * max_theta
            # 触壁操作
            outbandscalar = []
            direction_posit = self.now_normal_positions[idn] - self.old_normal_positions[idn]
            for idt in range(self.m):
                if self.now_normal_positions[idn][idt] < 0:
                    outbandscalar.append(-0.5 * self.old_normal_positions[idn][idt] / direction_posit[idt])
                elif self.now_normal_positions[idn][idt] > self.dim_sensitive[idt]:
                    outbandscalar.append(
                        0.5 * (self.dim_sensitive[idt] - self.old_normal_positions[idn][idt]) / direction_posit[idt])
                else:
                    outbandscalar.append(1)
            outbandscalar = min(outbandscalar)
            self.now_normal_positions[idn] = self.old_normal_positions[idn] + direction_posit * outbandscalar
            # 5. 位置差重赋值
            self.now_dx[idn] = self.now_normal_positions[idn] - self.old_normal_positions[idn]
            self.now_step_dist[idn] = np.sqrt(np.sum(np.square(self.now_dx[idn])))
            print("new_step_dist: {}, theta_dis: {}, step_scalar:{}, max_theta: {}, new_theta: {}".format(
                self.now_step_dist[idn], max_theta_dis, step_scalar, max_theta, self.new_theta[idn]))
            # 2.1 判断优秀结束 dy相对y 在 敏感尺寸的1/100 变化很小
            # target_Y 默认传入是100+
            print("dim_scalar: {}, dy: {}, dx_dis: {}, y: {}".format(dim_scalar, dy, self.old_step_dist[idn],
                                                                     self.now_target_Y[idn]))
            siglist = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).dy]".format(idn))
            # 如果list 采样数据至少四个，有负值，他们之间相差不大，说明到底部了。
            status_sig = None
            if self.now_step_dist[idn] < 0.0001 * min_sensi:
                self.status_sig[idn] = "步长结束"
                status_sig = "步长结束"
                print("类{}步长结束".format(idn))
            # elif siglist and len(siglist) > 3:
            #     if min(siglist[-4:]) < 0 and sum(siglist[-4:]) / 4 < siglist[-1] * 0.1:
            #         self.status_sig[idn] = "误差结束"
            # 2.2 合并判断
            for i2 in range(self.cluster_num):
                if idn == i2 or self.status_sig[i2] != "正常":
                    continue
                # 2.2.1 新点 与活动点 最近距离  判断 merge_percent 2/10 合并。
                everposi = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).now_normal_posi]".format(i2))
                if everposi:
                    mindis = []
                    for oneposi in everposi:
                        new_dis = self.now_normal_positions[idn] - oneposi
                        mindis.append(np.sqrt(np.sum(new_dis * new_dis)))
                    mindis = min(mindis)
                    if mindis / self.distance_near[idn] < self.merge_percent:
                        # 比较不同的值合并较低分值的。 正常 或 步长过小才能 合并。
                        if self.now_target_Y[idn] <= self.now_target_Y[i2] and not self.status_sig[idn].startswith(
                                "合并"):
                            self.status_sig[i2] = "合并{}到{}".format(i2, idn)
                            # status_sig = "合并{}到{}".format(i2, idn)
                            idreplace = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).id]".format(i2))
                            idreplace = idreplace[-1]
                            for id3, i3 in enumerate(self.result_json):
                                if i3["id"] == idreplace:
                                    self.result_json[id3]["status"] = self.status_sig[i2]
                            print(self.status_sig[i2])
                            break
                        elif self.now_target_Y[idn] > self.now_target_Y[i2] and not self.status_sig[i2].startswith(
                                "合并"):
                            self.status_sig[idn] = "合并{}到{}".format(idn, i2)
                            status_sig = "合并{}到{}".format(idn, i2)
                            print(self.status_sig[idn])
                            break
                        else:
                            pass
            # if idn == 1 and self.iter_id == 4:
            #     exit()
            # 6. 重新赋值
            self.save_result(idn, usetime, status_sig, max_theta_dis, dy)
            self.result_id += 1
        self.iter_id += 1

    def one_iter_random(self):
        # 一次迭代, 尺度含义还原成实际尺寸

        def learn_rate(t):
            return 0.9

        # 步长自适应
        min_sensi = np.min(self.dim_sensitive)
        min_dist = np.min(self.distance_near)
        dim_scalar = min_sensi * min_dist
        self.old_normal_positions = copy.deepcopy(self.now_normal_positions)
        self.now_theta = copy.deepcopy(self.new_theta)
        self.old_step_dist = copy.deepcopy(self.now_step_dist)
        self.target_X = self.lower_boundary + self.old_normal_positions / self.dim_sensitive * self.dim_scalar
        for idn in range(self.cluster_num):
            if idn != self.next_cluster_id:
                continue
            self.next_cluster_id = (self.next_cluster_id + 1) % self.cluster_num
            if self.status_sig[idn] != "正常":
                continue
            # 1. 遍历每一个类, 生成对应的结果
            sstart = time.time()
            print("class:", idn)
            self.old_target_Y[idn] = self.now_target_Y[idn]
            self.now_target_Y[idn] = self.fit_func(self.target_X[idn])
            # 时间以分钟为单位
            usetime = (time.time() - sstart) / 60
            print("usetime: {}mins".format(usetime))
            # 2. 生成新位置 梯度下降过程 用当下theta，保存下一步theta, 可以直接调用
            dy = self.now_target_Y[idn] - self.old_target_Y[idn]
            posilist = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).normal_posi]".format(idn))
            if posilist:
                dx = self.old_normal_positions[idn] - posilist[-1]
            else:
                dx = self.old_normal_positions[idn] - np.zeros((self.m))
            # 2.1 曲面导数变化经验化
            self.new_theta[idn] = learn_rate(1) * dy / np.sqrt(np.sum(np.square(dx))) * dx \
                                  + (1 - learn_rate(1)) * self.now_theta[idn]
            uniform_dis = np.sqrt(np.sum(np.square(self.new_theta[idn])))
            max_theta = 2
            thread_tmp = max_theta / uniform_dis
            self.new_theta[idn] = self.new_theta[idn] * thread_tmp if uniform_dis > max_theta else self.new_theta[idn]
            uniform_dis = np.sqrt(np.sum(np.square(self.new_theta[idn])))
            # 随机改变方向和步长, 20%
            rdxy = np.random.rand(self.m)
            # 垂直方向随机，均值后只占50%权重
            new_theta1 = self.new_theta[idn] * rdxy
            new_theta2 = np.sign(self.new_theta[idn]) * (uniform_dis - np.abs(self.new_theta[idn])) * rdxy
            new_theta = new_theta1 + new_theta2
            thetadis = np.sqrt(np.sum(np.square(new_theta)))
            # 1. 限制导数
            new_theta = new_theta / thetadis if thetadis > 2 else new_theta
            realdis = np.sqrt(np.sum(np.square(new_theta)))
            # 2. 最大步长
            max_step = 1
            step_scalar = max_step / realdis
            # 3. 限制step_scalar*realdis 的最小值
            min_step = 0.1
            step_scalar = min_step / realdis if step_scalar * realdis < min_step else step_scalar
            if step_scalar * realdis < min_step:
                step_scalar = (step_scalar - min_step / realdis) * 10 + min_step / realdis
            # 4. 新位置
            self.now_normal_positions[idn] = self.old_normal_positions[
                                                 idn] - self.init_step * dim_scalar * step_scalar * new_theta
            # 触壁操作
            outbandscalar = []
            direction_posit = self.now_normal_positions[idn] - self.old_normal_positions[idn]
            for idt in range(self.m):
                if self.now_normal_positions[idn][idt] < 0:
                    outbandscalar.append(-0.5 * self.old_normal_positions[idn][idt] / direction_posit[idt])
                elif self.now_normal_positions[idn][idt] > self.dim_sensitive[idt]:
                    outbandscalar.append(
                        0.5 * (self.dim_sensitive[idt] - self.old_normal_positions[idn][idt]) / direction_posit[idt])
                else:
                    outbandscalar.append(1)
            outbandscalar = min(outbandscalar)
            self.now_normal_positions[idn] = self.old_normal_positions[idn] + direction_posit * outbandscalar
            # 5. 位置差重赋值
            direction_posit = self.now_normal_positions[idn] - self.old_normal_positions[idn]
            self.now_step_dist[idn] = np.sqrt(np.sum(np.square(direction_posit)))
            print("step_dis: {}, theta_dis: {}, dy: {}, new_theta: {}".format(self.now_step_dist[idn], thetadis, dy,
                                                                              self.new_theta[idn]))
            # 2.1 判断优秀结束 dy相对y 在 敏感尺寸的1/100 变化很小
            # target_Y 默认传入是100+
            print("new_step_dist: {}, dim_scalar: {}, y: {}".format(self.now_step_dist[idn], dim_scalar,
                                                                    self.now_target_Y[idn]))
            siglist = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).dy]".format(idn))
            # 如果list 采样数据至少四个，有负值，他们之间相差不大，说明到底部了。
            if self.now_step_dist[idn] < 0.0001 * min_sensi:
                self.status_sig[idn] = "步长结束"
            # elif siglist and len(siglist) > 3:
            #     if min(siglist[-4:]) < 0 and sum(siglist[-4:]) / 4 < siglist[-1] * 0.1:
            #         self.status_sig[idn] = "误差结束"
            # 2.2 合并判断
            for i2 in range(self.cluster_num):
                if idn == i2:
                    continue
                # 2.2.1 新点 与活动点 最近距离  判断 2/10 合并。
                everposi = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).now_normal_posi]".format(i2))
                if everposi:
                    mindis = []
                    for oneposi in everposi:
                        new_dis = self.now_normal_positions[idn] - oneposi
                        mindis.append(np.sqrt(np.sum(new_dis * new_dis)))
                    mindis = min(mindis)
                    if mindis / self.distance_near[idn] < self.merge_percent:
                        # 比较不同的值合并较低分值的
                        if self.now_target_Y[idn] <= self.now_target_Y[i2] and not self.status_sig[idn].startswith(
                                "合并"):
                            self.status_sig[i2] = "合并{}到{}".format(i2, idn)
                            break
                        elif self.now_target_Y[idn] > self.now_target_Y[i2] and not self.status_sig[i2].startswith(
                                "合并"):
                            self.status_sig[idn] = "合并{}到{}".format(idn, i2)
                            break
                        else:
                            pass
            self.save_result(idn, usetime, thetadis, dy)
            self.result_id += 1
        self.iter_id += 1

    def call(self, n):
        # 完整的流程，不含展示
        for iter in range(n):
            if self.iter_id > iter:
                continue
            if "正常" not in self.status_sig:
                print("任务完成！")
                return None
            print("iter:", iter)
            # self.one_iter_random()
            self.one_iter_adam()

    def save_result(self, idn, usetime, status_sig, theta_dis, dy):
        # 3. 保存结果 json：ori_cluster_id, 新位置序号, 旧位置, 旧分值, theta, 新位置, 新分值, accur_posi, status
        loadfile = os.path.join(self.project_path, 'template.csv')
        tmpjson = {
            "id": self.result_id,
            "ori_cluster_id": self.cluster_id[idn],
            "iter_id": self.iter_id,
            "old_normal_posi": list(self.old_normal_positions[idn]),
            "now_normal_posi": list(self.now_normal_positions[idn]),
            "now_score": self.now_target_Y[idn],
            "old_score": self.old_target_Y[idn],
            "now_theta": list(self.now_theta[idn]),
            "new_theta": list(self.new_theta[idn]),
            "old_dx": list(self.old_dx[idn]),
            "now_dx": list(self.now_dx[idn]),
            "grade": self.grade[idn],
            "new_grade": self.new_grade[idn],
            "accur_posi": list(self.target_X[idn]),
            "status": status_sig if status_sig else self.status_sig[idn],
            "usetime": usetime,
            "old_step_dist": self.old_step_dist[idn],
            "now_step_dist": self.now_step_dist[idn],
            "theta_dis": theta_dis,
            "dy": dy,
        }
        print(status_sig, self.status_sig[idn])
        self.result_json.append(tmpjson)
        pdobj = pd.DataFrame(self.result_json)
        pdobj.to_csv(loadfile, index=False, header=True, encoding='gbk')
        # pdobj.to_csv(loadfile, index=False, header=True, encoding='uft8')

    def show_result(self):
        # 降维 分值分布图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 红绿蓝
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (
            self.cluster_num // 7 + 1)
        # xy = jsonpath.jsonpath(self.result_json, "$.['now_normal_posi']")
        xy = jsonpath.jsonpath(self.result_json, "$.['old_normal_posi']")
        xy = list(zip(*xy))
        zs = jsonpath.jsonpath(self.result_json, "$.['now_score']")
        clss = jsonpath.jsonpath(self.result_json, "$.['ori_cluster_id']")
        m = jsonpath.jsonpath(self.result_json, "$.['iter_id']")

        if self.m > 2:
            tsne = TSNE(n_components=2)
            tmX = np.array(xy)
            tmX = np.transpose(tmX)
            X_embedded = tsne.fit_transform(tmX, zs)
            X_embedded = np.transpose(X_embedded)
            xs = X_embedded[0]
            ys = X_embedded[1]
        else:
            xs = xy[0]
            ys = xy[1]
        c = [colors[i1] for i1 in clss]
        m = ['${}$'.format(i1) for i1 in m]
        for i1 in zip(xs, ys, zs, c, m):
            ax.scatter(i1[0], i1[1], i1[2], c=i1[3], marker=i1[4], s=50)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


class run_lines(object):
    def __init__(self):
        pass

    def run1line(self, line_json):
        pass

    def run1file(self, file_json):
        pass


class ParaSearch(object):
    def __init__(self, fit_func, iter_json, exe_json, project_path, run_info_list):
        """
        生成位置，迭代疏远非常近的。
        ---------------------------------------------------
        Input parameters:
          iter_json:
            cluster_num: Number of nests
            lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
            upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
            dim_sensitive: 维度的敏感性,越大切分的越细 -- 某维度切分=(upper_boundary-lower_boundary)/dim_sensitive
            exe_json: 运行位置 进程 等参数
            project_path: 磁盘全路径
            run_file_list: 运行文件顺序
            parakeys: 参数顺序
        Output:
            generated nests' locations
        参数数组：
          运行 参数数组 最后一组参数
            根据一个参数的数组，替换文件内容，运行这个文件。
        
          多个文件的参数都运行完，得到评估值。
          评估值更新到刚运行一组参数中。
          
          fit_func
          得出下一个新位置
            如果新位置和老位置小于某个阈值，直接用老位置得出下一个新位置
            如果新位置和老位置不小于某个阈值，直接用新位置
          把新位置加入参数数组，循环
        """
        self.fit_func = fit_func
        self.iter_json = iter_json
        self.exe_json = exe_json
        self.project_path = project_path
        self.run_file_list = run_info_list
        self.parakeys = iter_json

        # 解聚类个数
        self.cluster_num = self.iter_json["cluster_num"]
        # 参数下边界
        self.lower_boundary = np.array(self.iter_json["lower_boundary"])
        # 参数上边界
        self.upper_boundary = np.array(self.iter_json["upper_boundary"])
        # 参数敏感度
        self.dim_sensitive = np.array(self.iter_json["dim_sensitive"])
        # 维度标量大小
        self.dim_scalar = self.upper_boundary - self.lower_boundary
        # 维度数
        self.m = len(self.lower_boundary)
        # ****** 计算部分 ********
        # 迭代标准化起始步长
        self.init_step = self.iter_json["init_step"]
        self.max_iter_num = self.iter_json["max_iter_num"]
        self.max_step = self.iter_json["max_step"]
        self.min_step = self.iter_json["min_step"]
        self.max_process = self.exe_json["max_process"]
        # theta是参数，梯度下降通过不断更新theta的值使损失函数达到最小值
        self.now_theta = np.zeros((self.cluster_num, self.m))
        self.new_theta = np.zeros((self.cluster_num, self.m))
        self.old_dx = np.zeros((self.cluster_num, self.m))
        self.now_dx = np.zeros((self.cluster_num, self.m))
        self.grade = np.zeros((self.cluster_num))
        self.new_grade = np.zeros((self.cluster_num))
        # 聚类id合并百分比阈值
        self.merge_percent = self.iter_json["merge_percent"]
        self.reuse_percent = self.iter_json["reuse_percent"]
        # 标准初始位置
        self.init_normal_positions = None
        self.now_normal_positions = None
        self.old_normal_positions = None
        # 聚类id
        self.cluster_id = None
        # 原始聚类id距离矩阵
        self.ori_distance_matric = np.zeros((self.cluster_num, self.cluster_num))
        self.distance_near = np.zeros((self.cluster_num))
        # oldstepdis
        self.old_step_dist = np.zeros((self.cluster_num))
        self.now_step_dist = np.zeros((self.cluster_num))
        # 聚类id 当前得分
        self.now_target_Y = np.zeros((self.cluster_num))
        self.old_target_Y = np.zeros((self.cluster_num))
        # 聚类id状态：正常 合并 结束
        self.status_sig = ["正常"] * self.cluster_num
        # accur_posi
        self.target_X = np.zeros((self.cluster_num, self.m))

    def load_history(self):
        # todo: 默认数据记录文件
        loadfile = os.path.join(self.project_path, 'template.csv')
        self.next_cluster_id = 0
        if os.path.isfile(loadfile):
            # if 0:
            print("loadlog: {}".format(loadfile))
            # 1. 数据加载
            pdobj = pd.read_csv(loadfile, header=0, encoding="gbk")
            print("contents:", pdobj)
            self.result_json = json.loads(pdobj.to_json(orient='records', force_ascii=False), encoding="gbk")
            if len(json.loads(self.result_json["old_normal_posi"])) == 0:
                # 如果数据为空，初始化位置参数
                self.generate_positions()
            for item in self.result_json:
                item["old_normal_posi"] = json.loads(item["old_normal_posi"])
                item["now_normal_posi"] = json.loads(item["now_normal_posi"])
                item["now_theta"] = json.loads(item["now_theta"])
                item["new_theta"] = json.loads(item["new_theta"])
                item["accur_posi"] = json.loads(item["accur_posi"])
                item["old_dx"] = json.loads(item["old_dx"])
                item["now_dx"] = json.loads(item["now_dx"])
            # 2. 变量初始化
            iterset = set()
            for idn in range(self.cluster_num):
                tmpjson = [item for item in self.result_json if item["ori_cluster_id"] == idn]
                tmpjson = tmpjson[-1:]
                # new马上会被copy到now
                if len(tmpjson) > 0:
                    self.old_normal_positions[idn] = tmpjson[-1]["old_normal_posi"]
                    self.now_normal_positions[idn] = tmpjson[-1]["now_normal_posi"]
                    self.now_theta[idn] = np.array(tmpjson[-1]["now_theta"])
                    self.new_theta[idn] = np.array(tmpjson[-1]["new_theta"])
                    self.old_target_Y[idn] = tmpjson[-1]["old_score"]
                    self.now_target_Y[idn] = tmpjson[-1]["now_score"]
                    self.old_dx[idn] = tmpjson[-1]["old_dx"]
                    self.now_dx[idn] = tmpjson[-1]["now_dx"]
                    self.status_sig[idn] = tmpjson[-1]["status"]
                    self.grade[idn] = tmpjson[-1]["grade"]
                    self.new_grade[idn] = tmpjson[-1]["new_grade"]
                    self.old_step_dist[idn] = tmpjson[-1]["old_step_dist"]
                    self.now_step_dist[idn] = tmpjson[-1]["now_step_dist"]
                    # self.theta_dis[idn] = tmpjson[-1]["theta_dis"]
                    # self.dy[idn] = tmpjson[-1]["dy"]
                    iterset.add(tmpjson[-1]["iter_id"])
                else:
                    self.iter_id = 0
                    break
            self.result_id = self.result_json[-1]["id"] + 1
            # 筛选出 每个类 最大的id，且为正常的。如果对应的 iter_id 一致 且 result 大于等于1轮cluster num，直接+1
            self.iter_id = self.result_json[-1]["iter_id"]
            if len(iterset) == 1 and len(self.result_json) >= self.cluster_num:
                self.iter_id += 1
            self.next_cluster_id = (self.result_json[-1]["ori_cluster_id"] + 1) % self.cluster_num
        else:
            self.result_json = []
            self.result_id = 0
            self.iter_id = 0

    def generate_positions(self):
        # kmeans 生成位置, 用10倍于聚类数的随机点, 加入敏感度后聚类，然后每个类取中心
        # 1. 生成随机样本
        samples = np.random.rand(self.cluster_num * 10, self.m)
        # 按敏感度 scalar
        samples = samples * self.dim_sensitive
        # 2. 聚类，得出中心点。
        cls = KMeans(n_clusters=self.cluster_num, init='k-means++')
        cls.fit_predict(samples)
        self.init_normal_positions = cls.cluster_centers_
        self.now_normal_positions = cls.cluster_centers_
        self.old_normal_positions = cls.cluster_centers_
        self.cluster_id = cls.predict(cls.cluster_centers_)
        for i1 in range(self.cluster_num):
            for i2 in range(self.cluster_num):
                if i1 == i2:
                    self.ori_distance_matric[i1, i2] = np.float(9999)
                    continue
                tmpv = self.now_normal_positions[i1] - self.now_normal_positions[i2]
                self.ori_distance_matric[i1, i2] = np.sqrt(np.sum(tmpv * tmpv))
        self.distance_near = np.min(self.ori_distance_matric, 1)
        self.now_step_dist = np.sqrt(np.sum(np.square(self.now_normal_positions), 1))
        self.now_dx = self.now_normal_positions - 0

    def reference_olddata(self):
        # 参考老点，如果下一步位置与老点的位置小于，敏感度规的10%，用老点编号，否则用新点
        pass

    def update_position_list(self):
        # 单线程只需把 列表 更新到数据库，多线程需要再从 数据库 读回到 position_list
        self.position_list = self.position_list

    def one_iter_adam(self, position_list):
        """
        F(X) = F(X0) + F'(X0) * (X - X0) = 0
        J(X0) * H = B
        J(X0) == F'(X0) 雅克比矩阵
        (X - X0) == H  残余量
        B == -F(X0) 方程残余量

        X0 -> J, B 
        J * H = B -> H  
        X = H + X0
        X= H + X0 改
        X= damping_step * H + X0
        :return: 
        """
        # 一次迭代, 尺度含义还原成实际尺寸
        # 步长自适应
        min_sensi = np.min(self.dim_sensitive)
        min_dist = np.min(self.distance_near)
        dim_scalar = min_sensi * min_dist
        self.old_normal_positions = copy.deepcopy(self.now_normal_positions)
        self.now_theta = copy.deepcopy(self.new_theta)
        self.old_step_dist = copy.deepcopy(self.now_step_dist)
        self.old_dx = copy.deepcopy(self.now_dx)
        self.grade = copy.deepcopy(self.new_grade)
        self.target_X = self.lower_boundary + self.old_normal_positions / self.dim_sensitive * self.dim_scalar
        for idn in range(self.cluster_num):
            if idn != self.next_cluster_id:
                continue
            self.next_cluster_id = (self.next_cluster_id + 1) % self.cluster_num
            if self.status_sig[idn] != "正常":
                continue
            # 1. 遍历每一个类, 生成对应的结果
            sstart = time.time()
            print("class:", idn)
            # print("456")
            # print(self.now_target_Y[idn])
            self.old_target_Y[idn] = self.now_target_Y[idn]
            # todo: 调用外部程序的 自定义接口。
            self.now_target_Y[idn] = self.fit_func(self.target_X[idn], self.exe_json, self.project_path,
                                                   self.run_file_list,
                                                   self.parakeys)
            print(self.now_target_Y[idn])
            # 时间以分钟为单位
            usetime = (time.time() - sstart) / 60
            print("usetime: {}mins".format(usetime))
            # 2. 生成新位置 梯度下降过程 用当下theta，保存下一步theta, 可以直接调用
            dy = self.now_target_Y[idn] - self.old_target_Y[idn]
            dx_list = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).old_dx]".format(idn))
            theta_list = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).now_theta]".format(idn))
            dis_list = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).old_step_dist]".format(idn))
            # sig_list = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).status]".format(idn))
            if dx_list:
                old_dx = np.array(dx_list[-1])
                old_theta = np.array(theta_list[-1])
                old_dis = dis_list[-1]
            else:
                old_dx = np.zeros((self.m))
                old_theta = np.zeros((self.m))
                old_dis = np.sqrt(np.sum(np.square(self.init_normal_positions[idn])))
            # 2.1 曲面导数变化经验化
            add_dx_scalar = np.sqrt(np.sum(np.square(old_dx + self.old_dx[idn])))
            self.new_grade[idn] = dy / self.old_step_dist[idn]
            self.new_theta[idn] = self.new_grade[idn] * self.old_dx[idn]
            max_theta_dis = np.sqrt(np.sum(np.square(self.new_theta[idn])))
            self.new_theta[idn] = self.new_theta[idn] / max_theta_dis if max_theta_dis > 1 else self.new_theta[idn]
            # 综合 最近两次的 theta
            max_theta = (self.new_theta[idn] / self.old_step_dist[idn] + old_theta / old_dis) * add_dx_scalar
            max_theta_dis = np.sqrt(np.sum(np.square(max_theta)))
            # 随机改变方向和步长, 20%
            rdxy = np.random.rand(self.m)
            # 指方向占90%， 10%近似垂直方向随机，均值后只占50%权重
            new_theta1 = max_theta * rdxy
            new_theta2 = np.sign(max_theta) * (max_theta_dis - np.abs(max_theta)) * rdxy
            # 随机防止永远下一步在直线上
            # max_theta = 1.8 * new_theta1 + 0.2 * new_theta2
            # max_theta = 1.6 * new_theta1 + 0.4 * new_theta2
            max_theta = 1.2 * new_theta1 + 0.8 * new_theta2
            # max_theta_dis = np.sqrt(np.sum(np.square(max_theta)))
            # 1. 限制导数
            # max_theta = max_theta / max_theta_dis if max_theta_dis > 1 else max_theta
            real_thetadis = np.sqrt(np.sum(np.square(max_theta)))
            # 2. 最大步长
            max_step = 1.0
            # real_thetadis = real_thetadis / abs(real_thetadis) * ((abs(real_thetadis) - max_step) / 10 + max_step)
            step_scalar = max_step / real_thetadis
            # 3. 限制 step_scalar * real_thetadis 的最小值
            min_step = 0.1
            # print(step_scalar, real_thetadis, step_scalar * real_thetadis)
            step_scalar = min_step / real_thetadis if step_scalar * real_thetadis < min_step else step_scalar
            if step_scalar * real_thetadis < min_step:
                step_scalar = (step_scalar - min_step / real_thetadis) * 10 + min_step / real_thetadis
            # 4. 新位置
            self.now_normal_positions[idn] = self.old_normal_positions[
                                                 idn] - self.init_step * dim_scalar * step_scalar * max_theta
            # 触壁操作
            outbandscalar = []
            direction_posit = self.now_normal_positions[idn] - self.old_normal_positions[idn]
            for idt in range(self.m):
                if self.now_normal_positions[idn][idt] < 0:
                    outbandscalar.append(-0.5 * self.old_normal_positions[idn][idt] / direction_posit[idt])
                elif self.now_normal_positions[idn][idt] > self.dim_sensitive[idt]:
                    outbandscalar.append(
                        0.5 * (self.dim_sensitive[idt] - self.old_normal_positions[idn][idt]) / direction_posit[idt])
                else:
                    outbandscalar.append(1)
            outbandscalar = min(outbandscalar)
            self.now_normal_positions[idn] = self.old_normal_positions[idn] + direction_posit * outbandscalar
            # 5. 位置差重赋值
            self.now_dx[idn] = self.now_normal_positions[idn] - self.old_normal_positions[idn]
            self.now_step_dist[idn] = np.sqrt(np.sum(np.square(self.now_dx[idn])))
            print("new_step_dist: {}, theta_dis: {}, step_scalar:{}, max_theta: {}, new_theta: {}".format(
                self.now_step_dist[idn], max_theta_dis, step_scalar, max_theta, self.new_theta[idn]))
            # 2.1 判断优秀结束 dy相对y 在 敏感尺寸的1/100 变化很小
            # target_Y 默认传入是100+
            print("dim_scalar: {}, dy: {}, dx_dis: {}, y: {}".format(dim_scalar, dy, self.old_step_dist[idn],
                                                                     self.now_target_Y[idn]))
            siglist = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).dy]".format(idn))
            # 如果list 采样数据至少四个，有负值，他们之间相差不大，说明到底部了。
            status_sig = None
            if self.now_step_dist[idn] < 0.0001 * min_sensi:
                self.status_sig[idn] = "步长结束"
                status_sig = "步长结束"
                print("类{}步长结束".format(idn))
            # elif siglist and len(siglist) > 3:
            #     if min(siglist[-4:]) < 0 and sum(siglist[-4:]) / 4 < siglist[-1] * 0.1:
            #         self.status_sig[idn] = "误差结束"
            # 2.2 合并判断
            for i2 in range(self.cluster_num):
                if idn == i2 or self.status_sig[i2] != "正常":
                    continue
                # 2.2.1 新点 与活动点 最近距离  判断 merge_percent 2/10 合并。
                everposi = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).now_normal_posi]".format(i2))
                if everposi:
                    mindis = []
                    for oneposi in everposi:
                        new_dis = self.now_normal_positions[idn] - oneposi
                        mindis.append(np.sqrt(np.sum(new_dis * new_dis)))
                    mindis = min(mindis)
                    if mindis / self.distance_near[idn] < self.merge_percent:
                        # 比较不同的值合并较低分值的。 正常 或 步长过小才能 合并。
                        if self.now_target_Y[idn] <= self.now_target_Y[i2] and not self.status_sig[idn].startswith(
                                "合并"):
                            self.status_sig[i2] = "合并{}到{}".format(i2, idn)
                            # status_sig = "合并{}到{}".format(i2, idn)
                            idreplace = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).id]".format(i2))
                            idreplace = idreplace[-1]
                            for id3, i3 in enumerate(self.result_json):
                                if i3["id"] == idreplace:
                                    self.result_json[id3]["status"] = self.status_sig[i2]
                            print(self.status_sig[i2])
                            break
                        elif self.now_target_Y[idn] > self.now_target_Y[i2] and not self.status_sig[i2].startswith(
                                "合并"):
                            self.status_sig[idn] = "合并{}到{}".format(idn, i2)
                            status_sig = "合并{}到{}".format(idn, i2)
                            print(self.status_sig[idn])
                            break
                        else:
                            pass
            # if idn == 1 and self.iter_id == 4:
            #     exit()
            # 6. 重新赋值
            self.save_result(idn, usetime, status_sig, max_theta_dis, dy)
            self.result_id += 1
        self.iter_id += 1

    def call(self):
        # 完整的流程，不含展示
        self.position_list = []
        for iter in range(self.max_iter_num):
            # 加载历史记录
            self.load_history()
            linekeys = self.process_lines()
            for linprocess in linekeys:
                p = Process(target=run_proc, args=(linprocess,))
                print('Child process will start.')
                p.start()
                p.join()
            # 判断执行哪个cluster, 返回clusterid
            self.reference_olddata()
            if self.iter_id > iter:
                continue
            if "正常" not in self.status_sig:
                print("任务完成！")
                return None
            print("iter:", iter)
            # self.position_list = []
            self.update_position_list()
            self.one_iter_adam(self.position_list)

    def process_lines(self):
        # 判断可用进程数，生成对应的行号
        running_num = 1
        self.max_process
        linekeys = []

        return linekeys

    def save_result(self, idn, usetime, status_sig, theta_dis, dy):
        # 3. 保存结果 json: ori_cluster_id, 新位置序号, 旧位置, 旧分值, theta, 新位置, 新分值, accur_posi, status
        loadfile = os.path.join(self.project_path, 'template.csv')
        tmpjson = {
            "id": self.result_id,
            "ori_cluster_id": self.cluster_id[idn],
            "iter_id": self.iter_id,
            "old_normal_posi": list(self.old_normal_positions[idn]),
            "now_normal_posi": list(self.now_normal_positions[idn]),
            "now_score": self.now_target_Y[idn],
            "old_score": self.old_target_Y[idn],
            "now_theta": list(self.now_theta[idn]),
            "new_theta": list(self.new_theta[idn]),
            "old_dx": list(self.old_dx[idn]),
            "now_dx": list(self.now_dx[idn]),
            "grade": self.grade[idn],
            "new_grade": self.new_grade[idn],
            "accur_posi": list(self.target_X[idn]),
            "status": status_sig if status_sig else self.status_sig[idn],
            "usetime": usetime,
            "old_step_dist": self.old_step_dist[idn],
            "now_step_dist": self.now_step_dist[idn],
            "theta_dis": theta_dis,
            "dy": dy,
        }
        print(status_sig, self.status_sig[idn])
        self.result_json.append(tmpjson)
        pdobj = pd.DataFrame(self.result_json)
        pdobj.to_csv(loadfile, index=False, header=True, encoding='gbk')
        # pdobj.to_csv(loadfile, index=False, header=True, encoding='uft8')

    def show_result(self):
        # 降维 分值分布图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 红绿蓝
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (
            self.cluster_num // 7 + 1)
        # xy = jsonpath.jsonpath(self.result_json, "$.['now_normal_posi']")
        xy = jsonpath.jsonpath(self.result_json, "$.['old_normal_posi']")
        xy = list(zip(*xy))
        zs = jsonpath.jsonpath(self.result_json, "$.['now_score']")
        clss = jsonpath.jsonpath(self.result_json, "$.['ori_cluster_id']")
        m = jsonpath.jsonpath(self.result_json, "$.['iter_id']")

        if self.m > 2:
            tsne = TSNE(n_components=2)
            tmX = np.array(xy)
            tmX = np.transpose(tmX)
            X_embedded = tsne.fit_transform(tmX, zs)
            X_embedded = np.transpose(X_embedded)
            xs = X_embedded[0]
            ys = X_embedded[1]
        else:
            xs = xy[0]
            ys = xy[1]
        c = [colors[i1] for i1 in clss]
        m = ['${}$'.format(i1) for i1 in m]
        for i1 in zip(xs, ys, zs, c, m):
            ax.scatter(i1[0], i1[1], i1[2], c=i1[3], marker=i1[4], s=50)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


def fit_func_bak(nest):
    """
    1. 各维坐标，代入函数。
    2. 输出多指标，需要一个综合的公式，分值越高越好, 要求非负，默认1e2+ 一般不会碰到这么大的数值。
    """
    x1, x2 = nest
    # time.sleep(0.2)
    return -1 * np.cos(x1) * np.cos(x2) + 1e1
    return 3 * (1 - x1) ** 2 * np.e ** (-x1 ** 2 - (x2 + 1) ** 2) - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.e ** (
        -x1 ** 2 - x2 ** 2) - (np.e ** (-(x1 + 1) ** 2 - x2 ** 2)) / 3


def fit_func(target_X, exe_json, project_path, run_file_list, parakeys):
    """
    1. 执行根目录
    2. 执行全路径
    3. 参数键值对
    4. 各维坐标，代入函数。
    5. 输出多指标，需要一个综合的公式，分值越高越好, 要求非负，默认1e2+ 一般不会碰到这么大的数值。
    """
    jsonkvs = {k: str(v) for k, v in zip(parakeys, target_X)}
    try:
        # 1. 生成文件目录
        print("copy", project_path, jsonkvs)
        # template_copy(project_path, jsonkvs)
        # 2. 程序调用生成 结果文件
        for onfile in run_file_list:
            if onfile.endswith(".in"):
                print(f"runing {onfile}...")
                # todo: 修改命令行 去掉continue
                continue
                os.system(f'{exe_json} && {project_path} && python v_print.py')
            elif onfile.endswith(".sol"):
                print(f"runing {onfile}...")
                continue
                os.system(f'{exe_json} && {project_path} && python nt.py')
        # 3. 提取结果文件相关信息，生成targetY
        result = extract_result(project_path, run_file_list, jsonkvs)
    except Exception as e:
        print(e)
        result = None
    return result


def template_copy(path, jsonkvs):
    """
    文件夹路径 
    单组变量替换。
    """
    if not os.path.isdir(path):
        raise Exception("需要的文件夹不存在")
    patht = os.path.join(path, "template")
    files = os.listdir(patht)
    files = [file for file in files if file.endswith(".in") or file.endswith(".sol")]
    dirname = "-".join([k + "_" + v for k, v in jsonkvs.items()])
    dirname = dirname.replace("@", "")
    newdir = os.path.join(path, dirname)
    shutil.copytree(patht, newdir)
    for onefile in files:
        tfile = os.path.join(newdir, onefile)
        with open(tfile, 'rt', encoding='utf-8') as f:
            # rstrs = f.readlines()
            rstrs = f.read()
            for k, v in jsonkvs.items():
                rstrs = rstrs.replace(k, v)
        with open(tfile, 'wt', encoding='utf-8') as f:
            f.write(rstrs)


def extract_result(project_path, run_file_list, jsonkvs=None):
    """
    :param project_path: 根据目录名从文件提取结果数据 
    :param jsonkvs: 这项可能用不到
    :return: 
    """
    # # 1. 读取文件
    # # todo: 添加文件处理
    # ana_file_list = [".".join(file.split(".")[:-1]) + ".log" for file in run_file_list if file.endswith(".sol")]
    # varlist = []
    # # 每个文件不同的处理
    # with open(os.path.join(project_path, ana_file_list[0]), 'rt', encoding='utf-8') as f:
    #     fstrs = f.read()
    #     ustr = fstrs + ""
    #     varlist.append(float(ustr))
    #     print()
    # # 2. 返回运算后的数据
    # result = varlist[0] + varlist[1]
    x1, x2 = jsonkvs.values()
    x1, x2 = np.float(x1), np.float(x2)
    print(x1, x2)
    return -1 * np.cos(x1) * np.cos(x2) - 1e2


def main():
    # 必须有种子值，便于中断后根据记录文件重新加载没运行的。
    np.random.seed(546998)
    iter_json = {
        "cluster_num": 3,
        "lower_boundary": [-1, -1],
        "upper_boundary": [1, 1],
        "dim_sensitive": [1, 1],
        "init_step": 0.1,
        "max_step": 0.15,
        "min_step": 0.05,
        "merge_percent": 0.2,
        "reuse_percent": 0.05,
        "max_iter_num": 9,
    }
    exe_json = {"apsys": "", "csuprem": "", "pics3d": "", "max_process": 2}
    project_path = os.path.join("c:\\", "project", "simuproject", "simon", "test_qyb")
    run_info_list = [
        ["apsys", "t.sol", ["@doe_fa", 0], ["@doe_fb", 0]],
        ["apsys", "t.plt", ["@doe_fa", 0], ["@doe_fc", 0]]
    ]
    psins = ParaSearch(fit_func, iter_json, exe_json, project_path, run_info_list)
    n = 20
    psins.call()
    psins.show_result()


def dim3_surface():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    dimx = 1
    dimy = 1
    x = np.arange(-1, dimx, 0.1)
    y = np.arange(-1, dimy, 0.1)

    x, y = np.meshgrid(y, x)
    m = np.arange(20 * 20).reshape(20, 20)
    print(m)
    for i in range(20):
        m[i, :] = i

    tt = 1 / np.power(1000, 2 * m / 100)
    for i in range(dimy):
        tt[:, i] = tt[:, i] * i

    z = np.sin(tt)
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)  # 用取样点(x,y,z)去构建曲面
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))  # 用取样点(x,y,z)去构建曲面
    plt.show()


if __name__ == '__main__':
    # dim3_surface()
    main()
