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
from sklearn.manifold import TSNE
import math

bathpath = None


class ParaSearch(object):
    def __init__(self, fit_func, parajson):
        """
        生成位置，迭代疏远非常近的。
        ---------------------------------------------------
        Input parameters:
            cluster_num: Number of nests
            lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
            upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
            dim_sensitive: 维度的敏感性,越大切分的越细 -- 某维度切分=(upper_boundary-lower_boundary)/dim_sensitive
        Output:
            generated nests' locations
        """
        self.fit_func = fit_func
        self.parajson = parajson

        # 项目名
        self.project_name = self.parajson["project"]
        # 解聚类个数
        self.cluster_num = self.parajson["cluster_num"]
        # 参数下边界
        self.lower_boundary = np.array(self.parajson["lower_boundary"])
        # 参数上边界
        self.upper_boundary = np.array(self.parajson["upper_boundary"])
        # 参数敏感度
        self.dim_sensitive = np.array(self.parajson["dim_sensitive"])
        # 维度标量大小
        self.dim_scalar = self.upper_boundary - self.lower_boundary
        # 维度数
        self.m = len(self.lower_boundary)
        # ****** 计算部分 ********
        # 迭代标准化起始步长
        self.init_step = self.parajson["init_step"]
        # theta是参数，梯度下降通过不断更新theta的值使损失函数达到最小值
        self.theta = np.zeros((self.cluster_num, self.m))
        self.new_theta = np.zeros((self.cluster_num, self.m))
        # 聚类id合并百分比阈值
        self.merge_percent = self.parajson["merge_percent"]
        # 标准初始位置
        self.init_normal_positions = None
        # new_normal_posi
        self.new_normal_positions = None
        # normal_posi
        self.now_normal_positions = None
        # 聚类id
        self.cluster_id = None
        # 原始聚类id距离矩阵
        self.ori_distance_matric = np.zeros((self.cluster_num, self.cluster_num))
        self.distance_near = np.zeros((self.cluster_num))
        # oldstepdis
        self.old_step_dist = np.zeros((self.cluster_num))
        self.new_step_dist = np.zeros((self.cluster_num))
        # 初始参数
        self.generate_positions()
        # 加载历史记录
        self.load_history()

        # self.theta = np.zeros(self.m)  # 参数
        self.alpha = 0.01 * np.ones(self.cluster_num)  # 学习率
        self.momentum = 0.1 * np.ones(self.cluster_num)  # 冲量
        self.threshold = 0.0001  # 停止迭代的错误阈值
        self.error = 0  # 初始错误为0

        # self.b1 = 0.9  # 算法作者建议的默认值
        # self.b2 = 0.999  # 算法作者建议的默认值
        self.b1 = 0.9  # 算法作者建议的默认值
        self.b2 = 0.999  # 算法作者建议的默认值
        self.e = 0.00000001  # 算法作者建议的默认值
        self.mt = np.zeros((self.cluster_num, self.m))
        self.vt = np.zeros((self.cluster_num, self.m))

    def load_history(self):
        loadfile = os.path.join(bathpath, self.project_name + '.csv')
        # 聚类id 当前得分
        self.target_Y = np.zeros((self.cluster_num))
        self.ori_target_Y = np.zeros((self.cluster_num))
        # 聚类id状态：正常 合并 结束
        self.status_sig = ["正常"] * self.cluster_num
        # accur_posi
        self.target_X = np.zeros((self.cluster_num, self.m))
        self.next_cluster_id = 0
        if not os.path.isfile(loadfile):
            print("loadlog: {}".format(loadfile))
            # 1. 数据加载
            pdobj = pd.read_csv(loadfile, header=0, encoding="gbk")
            self.result_json = json.loads(pdobj.to_json(orient='records', force_ascii=False), encoding="utf-8")
            for item in self.result_json:
                item["normal_posi"] = json.loads(item["normal_posi"])
                item["accur_posi"] = json.loads(item["accur_posi"])
                item["new_normal_posi"] = json.loads(item["new_normal_posi"])
                item["now_theta"] = json.loads(item["now_theta"])
                item["new_theta"] = json.loads(item["new_theta"])
            # 2. 变量初始化
            iterset = set()
            for idn in range(self.cluster_num):
                tmpjson = [item for item in self.result_json if item["ori_cluster_id"] == idn]
                tmpjson = tmpjson[-1:]
                # new马上会被copy到now
                if len(tmpjson) > 0:
                    self.new_normal_positions[idn] = tmpjson[-1]["new_normal_posi"]
                    self.theta[idn] = tmpjson[-1]["now_theta"]
                    self.new_theta[idn] = tmpjson[-1]["new_theta"]
                    self.status_sig[idn] = tmpjson[-1]["status"]
                    self.ori_target_Y[idn] = tmpjson[-1]["old_score"]
                    self.target_Y[idn] = tmpjson[-1]["now_score"]
                    # if tmpjson[-1]["status"] == "正常":
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
        self.new_normal_positions = cls.cluster_centers_
        self.cluster_id = cls.predict(cls.cluster_centers_)
        for i1 in range(self.cluster_num):
            for i2 in range(self.cluster_num):
                if i1 == i2:
                    self.ori_distance_matric[i1, i2] = np.float(9999)
                    continue
                tmpv = self.new_normal_positions[i1] - self.new_normal_positions[i2]
                self.ori_distance_matric[i1, i2] = np.sqrt(np.sum(tmpv * tmpv))
        self.distance_near = np.min(self.ori_distance_matric, 1)
        self.new_step_dist = np.sqrt(np.sum(np.square(self.new_normal_positions), 1))

    def one_iter_waite(self):
        # 一次迭代, 尺度含义还原成实际尺寸
        # 步长自适应
        min_sensi = np.min(self.dim_sensitive)
        min_dist = np.min(self.distance_near)
        dim_scalar = min_sensi * min_dist
        self.now_normal_positions = copy.deepcopy(self.new_normal_positions)
        self.theta = copy.deepcopy(self.new_theta)
        self.old_step_dist = copy.deepcopy(self.new_step_dist)
        self.target_X = self.lower_boundary + self.now_normal_positions / self.dim_sensitive * self.dim_scalar
        for idn in range(self.cluster_num):
            if idn != self.next_cluster_id:
                continue
            self.next_cluster_id = (self.next_cluster_id + 1) % self.cluster_num
            if self.status_sig[idn] != "正常":
                continue
            # 1. 遍历每一个类, 生成对应的结果
            sstart = time.time()
            print("class:", idn)
            self.ori_target_Y[idn] = self.target_Y[idn]
            self.target_Y[idn] = self.fit_func(self.target_X[idn])
            # 时间以分钟为单位
            usetime = (time.time() - sstart) / 60
            print("usetime: {}mins".format(usetime))
            # 2. 生成新位置 梯度下降过程 用当下theta，保存下一步theta, 可以直接调用
            dy = self.target_Y[idn] - self.ori_target_Y[idn]

            posilist = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).normal_posi]".format(idn))
            if posilist:
                dx = self.now_normal_positions[idn] - posilist[-1]
            else:
                dx = self.now_normal_positions[idn] - np.zeros((self.m))
            gradient = self.now_normal_positions[idn] * (np.dot(self.now_normal_positions[idn], self.theta[idn]) - self.target_Y[idn])
            print(gradient)
            self.mt[idn] = self.b1 * self.mt[idn] + (1 - self.b1) * gradient
            self.vt[idn] = self.b2 * self.vt[idn] + (1 - self.b2) * (gradient ** 2)
            mtt = self.mt[idn] / (1 - (self.b1 ** (self.result_id + 1)))
            vtt = self.vt[idn] / (1 - (self.b2 ** (self.result_id + 1)))
            vtt_sqrt = np.array([math.sqrt(vtt[0]), math.sqrt(vtt[1])])  # 因为只能对标量进行开方
            self.new_theta[idn] = self.theta[idn] - self.alpha[idn] * mtt / (vtt_sqrt + self.e)

            new_theta = self.new_theta[idn]
            # 1. 限制导数
            uniform_dis = np.sqrt(np.sum(np.square(new_theta)))
            max_theta = 2
            thread_tmp = max_theta / uniform_dis
            new_theta = new_theta * thread_tmp if uniform_dis > max_theta else new_theta
            thetadis = np.sqrt(np.sum(np.square(new_theta)))
            # 2. 最大步长
            max_step = 1
            step_scalar = max_step / uniform_dis
            # 3. 限制step_scalar*realdis 的最小值
            min_step = 0.1
            step_scalar = min_step / uniform_dis if step_scalar * uniform_dis < min_step else step_scalar
            if step_scalar * uniform_dis < min_step:
                step_scalar = (step_scalar - min_step / uniform_dis) * 10 + min_step / uniform_dis
            # 4. 新位置
            self.new_normal_positions[idn] = self.now_normal_positions[
                                                 idn] - self.init_step * dim_scalar * step_scalar * new_theta
            # 触壁操作
            outbandscalar = []
            direction_posit = self.new_normal_positions[idn] - self.now_normal_positions[idn]
            for idt in range(self.m):
                if self.new_normal_positions[idn][idt] < 0:
                    outbandscalar.append(-0.5 * self.now_normal_positions[idn][idt] / direction_posit[idt])
                elif self.new_normal_positions[idn][idt] > self.dim_sensitive[idt]:
                    outbandscalar.append(
                        0.5 * (self.dim_sensitive[idt] - self.now_normal_positions[idn][idt]) / direction_posit[idt])
                else:
                    outbandscalar.append(1)
            outbandscalar = min(outbandscalar)
            self.new_normal_positions[idn] = self.now_normal_positions[idn] + direction_posit * outbandscalar
            # 5. 位置差重赋值
            direction_posit = self.new_normal_positions[idn] - self.now_normal_positions[idn]
            self.new_step_dist[idn] = np.sqrt(np.sum(np.square(direction_posit)))
            print("step_dis: {}, theta_dis: {}, dy: {}, new_theta: {}".format(self.new_step_dist[idn], thetadis, dy,
                                                                              self.new_theta[idn]))
            # 2.1 判断优秀结束 dy相对y 在 敏感尺寸的1/100 变化很小
            # target_Y 默认传入是100+
            print("new_step_dist: {}, dim_scalar: {}, y: {}".format(self.new_step_dist[idn], dim_scalar,
                                                                    self.target_Y[idn]))
            siglist = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).dy]".format(idn))
            # 如果list 采样数据至少四个，有负值，他们之间相差不大，说明到底部了。
            if self.new_step_dist[idn] < 0.0001 * min_sensi:
                self.status_sig[idn] = "步长结束"
            # elif siglist and len(siglist) > 3:
            #     if min(siglist[-4:]) < 0 and sum(siglist[-4:]) / 4 < siglist[-1] * 0.1:
            #         self.status_sig[idn] = "误差结束"
            # 2.2 合并判断
            for i2 in range(self.cluster_num):
                if idn == i2:
                    continue
                # 2.2.1 新点 与活动点 最近距离  判断 2/10 合并。
                everposi = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).new_normal_posi]".format(i2))
                if everposi:
                    mindis = []
                    for oneposi in everposi:
                        new_dis = self.new_normal_positions[idn] - oneposi
                        mindis.append(np.sqrt(np.sum(new_dis * new_dis)))
                    mindis = min(mindis)
                    if mindis / self.distance_near[idn] < self.merge_percent:
                        # 比较不同的值合并较低分值的
                        if self.target_Y[idn] <= self.target_Y[i2] and not self.status_sig[idn].startswith("合并"):
                            self.status_sig[i2] = "合并{}到{}".format(i2, idn)
                            break
                        elif self.target_Y[idn] > self.target_Y[i2] and not self.status_sig[i2].startswith("合并"):
                            self.status_sig[idn] = "合并{}到{}".format(idn, i2)
                            break
                        else:
                            pass
            self.save_result(idn, usetime, thetadis, dy)
            self.result_id += 1
        self.iter_id += 1

    def one_iter(self):
        # 一次迭代, 尺度含义还原成实际尺寸

        def learn_rate(t):
            return 0.9

        # 步长自适应
        min_sensi = np.min(self.dim_sensitive)
        min_dist = np.min(self.distance_near)
        dim_scalar = min_sensi * min_dist
        self.now_normal_positions = copy.deepcopy(self.new_normal_positions)
        self.theta = copy.deepcopy(self.new_theta)
        self.old_step_dist = copy.deepcopy(self.new_step_dist)
        self.target_X = self.lower_boundary + self.now_normal_positions / self.dim_sensitive * self.dim_scalar
        for idn in range(self.cluster_num):
            if idn != self.next_cluster_id:
                continue
            self.next_cluster_id = (self.next_cluster_id + 1) % self.cluster_num
            if self.status_sig[idn] != "正常":
                continue
            # 1. 遍历每一个类, 生成对应的结果
            sstart = time.time()
            print("class:", idn)
            self.ori_target_Y[idn] = self.target_Y[idn]
            self.target_Y[idn] = self.fit_func(self.target_X[idn])
            # 时间以分钟为单位
            usetime = (time.time() - sstart) / 60
            print("usetime: {}mins".format(usetime))
            # 2. 生成新位置 梯度下降过程 用当下theta，保存下一步theta, 可以直接调用
            dy = self.target_Y[idn] - self.ori_target_Y[idn]
            posilist = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).normal_posi]".format(idn))
            if posilist:
                dx = self.now_normal_positions[idn] - posilist[-1]
            else:
                dx = self.now_normal_positions[idn] - np.zeros((self.m))
            # 2.1 曲面导数变化经验化
            self.new_theta[idn] = learn_rate(1) * dy / np.sqrt(np.sum(np.square(dx))) * dx \
                                  + (1 - learn_rate(1)) * self.theta[idn]
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
            self.new_normal_positions[idn] = self.now_normal_positions[
                                                 idn] - self.init_step * dim_scalar * step_scalar * new_theta
            # 触壁操作
            outbandscalar = []
            direction_posit = self.new_normal_positions[idn] - self.now_normal_positions[idn]
            for idt in range(self.m):
                if self.new_normal_positions[idn][idt] < 0:
                    outbandscalar.append(-0.5 * self.now_normal_positions[idn][idt] / direction_posit[idt])
                elif self.new_normal_positions[idn][idt] > self.dim_sensitive[idt]:
                    outbandscalar.append(
                        0.5 * (self.dim_sensitive[idt] - self.now_normal_positions[idn][idt]) / direction_posit[idt])
                else:
                    outbandscalar.append(1)
            outbandscalar = min(outbandscalar)
            self.new_normal_positions[idn] = self.now_normal_positions[idn] + direction_posit * outbandscalar
            # 5. 位置差重赋值
            direction_posit = self.new_normal_positions[idn] - self.now_normal_positions[idn]
            self.new_step_dist[idn] = np.sqrt(np.sum(np.square(direction_posit)))
            print("step_dis: {}, theta_dis: {}, dy: {}, new_theta: {}".format(self.new_step_dist[idn], thetadis, dy,
                                                                              self.new_theta[idn]))
            # 2.1 判断优秀结束 dy相对y 在 敏感尺寸的1/100 变化很小
            # target_Y 默认传入是100+
            print("new_step_dist: {}, dim_scalar: {}, y: {}".format(self.new_step_dist[idn], dim_scalar,
                                                                    self.target_Y[idn]))
            siglist = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).dy]".format(idn))
            # 如果list 采样数据至少四个，有负值，他们之间相差不大，说明到底部了。
            if self.new_step_dist[idn] < 0.0001 * min_sensi:
                self.status_sig[idn] = "步长结束"
            # elif siglist and len(siglist) > 3:
            #     if min(siglist[-4:]) < 0 and sum(siglist[-4:]) / 4 < siglist[-1] * 0.1:
            #         self.status_sig[idn] = "误差结束"
            # 2.2 合并判断
            for i2 in range(self.cluster_num):
                if idn == i2:
                    continue
                # 2.2.1 新点 与活动点 最近距离  判断 2/10 合并。
                everposi = jsonpath.jsonpath(self.result_json, "$.[?(@.ori_cluster_id=={}).new_normal_posi]".format(i2))
                if everposi:
                    mindis = []
                    for oneposi in everposi:
                        new_dis = self.new_normal_positions[idn] - oneposi
                        mindis.append(np.sqrt(np.sum(new_dis * new_dis)))
                    mindis = min(mindis)
                    if mindis / self.distance_near[idn] < self.merge_percent:
                        # 比较不同的值合并较低分值的
                        if self.target_Y[idn] <= self.target_Y[i2] and not self.status_sig[idn].startswith("合并"):
                            self.status_sig[i2] = "合并{}到{}".format(i2, idn)
                            break
                        elif self.target_Y[idn] > self.target_Y[i2] and not self.status_sig[i2].startswith("合并"):
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
            self.one_iter()

    def save_result(self, idn, usetime, theta_dis, dy):
        # 3. 保存结果 json：ori_cluster_id, 新位置序号, 旧位置, 旧分值, theta, 新位置, 新分值, accur_posi, status
        loadfile = os.path.join(bathpath, self.project_name + '.csv')
        tmpjson = {
            "id": self.result_id,
            "ori_cluster_id": self.cluster_id[idn],
            "iter_id": self.iter_id,
            "normal_posi": list(self.now_normal_positions[idn]),
            "new_normal_posi": list(self.new_normal_positions[idn]),
            "now_score": self.target_Y[idn],
            "old_score": self.ori_target_Y[idn],
            "now_theta": list(self.theta[idn]),
            "new_theta": list(self.new_theta[idn]),
            "accur_posi": list(self.target_X[idn]),
            "status": self.status_sig[idn],
            "usetime": usetime,
            "old_step_dis": self.old_step_dist[idn],
            "now_step_dis": self.new_step_dist[idn],
            "theta_dis": theta_dis,
            "dy": dy,
        }
        self.result_json.append(tmpjson)
        pdobj = pd.DataFrame(self.result_json)
        pdobj.to_csv(loadfile, index=False, header=True, encoding='gbk')

    def show_result(self):
        # 降维 分值分布图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 红绿蓝
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (
            self.cluster_num // 7 + 1)
        xy = jsonpath.jsonpath(self.result_json, "$.['normal_posi']")
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


def fit_func_demo(nest):
    """
    1. 各维坐标，代入函数。
    2. 输出多指标，需要一个综合的公式，分值越高越好。
    """
    x1, x2 = nest
    return 3 * (1 - x1) ** 2 * np.e ** (-x1 ** 2 - (x2 + 1) ** 2) - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.e ** (
        -x1 ** 2 - x2 ** 2) - (np.e ** (-(x1 + 1) ** 2 - x2 ** 2)) / 3


def fit_func(nest):
    """
    1. 各维坐标，代入函数。
    2. 输出多指标，需要一个综合的公式，分值越高越好, 要求非负，默认1e2+ 一般不会碰到这么大的数值。
    """
    x1, x2 = nest
    # time.sleep(0.2)
    return -10000 * np.cos(x1) * np.cos(x2) + 1e2
    return 3 * (1 - x1) ** 2 * np.e ** (-x1 ** 2 - (x2 + 1) ** 2) - 10 * (x1 / 5 - x1 ** 3 - x2 ** 5) * np.e ** (
        -x1 ** 2 - x2 ** 2) - (np.e ** (-(x1 + 1) ** 2 - x2 ** 2)) / 3


def main():
    # 必须有种子值，便于中断后根据记录文件重新加载没运行的。
    np.random.seed(546)
    parajson = {
        "project": "测试",
        "bathpath": "..",
        "cluster_num": 3,
        "lower_boundary": [-1, -1],
        "upper_boundary": [1, 1],
        "dim_sensitive": [1, 1],
        "init_step": 0.2,
        "merge_percent": 0.2,
    }
    global bathpath
    bathpath = parajson["bathpath"]
    psins = ParaSearch(fit_func, parajson)
    n = 20
    psins.call(n)
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
    # data_gene()
    # dim3_surface()
    main()
