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
        # 新标准位置
        self.new_normal_positions = None
        # 标准位置
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

    def load_history(self):
        # loadfile = os.path.join(bathpath, self.project_name + '.xls')
        loadfile = os.path.join(bathpath, self.project_name + '.csv')
        # 聚类id 当前得分
        self.target_Y = np.zeros((self.cluster_num))
        self.ori_target_Y = np.zeros((self.cluster_num))
        # 聚类id状态：正常 合并 结束
        self.status_sig = ["正常"] * self.cluster_num
        # 实际位置
        self.target_X = np.zeros((self.cluster_num, self.m))
        self.next_cluster_id = 0
        if not os.path.isfile(loadfile):
            print("loadlog: {}".format(loadfile))
            # 1. 数据加载
            # pdobj = pd.read_excel(loadfile, sheet_name='Sheet1', header=0, encoding="utf-8")
            pdobj = pd.read_csv(loadfile, header=0, encoding="gbk")
            self.result_json = json.loads(pdobj.to_json(orient='records', force_ascii=False), encoding="utf-8")
            for item in self.result_json:
                item["标准位置"] = json.loads(item["标准位置"])
                item["实际位置"] = json.loads(item["实际位置"])
                item["新标准位置"] = json.loads(item["新标准位置"])
                item["当前theta"] = json.loads(item["当前theta"])
                item["未来theta"] = json.loads(item["未来theta"])
            # 2. 变量初始化
            iterset = set()
            for idn in range(self.cluster_num):
                tmpjson = [item for item in self.result_json if item["原始类id"] == idn]
                tmpjson = tmpjson[-1:]
                # new马上会被copy到now
                if len(tmpjson) > 0:
                    self.new_normal_positions[idn] = tmpjson[-1]["新标准位置"]
                    self.theta[idn] = tmpjson[-1]["当前theta"]
                    self.new_theta[idn] = tmpjson[-1]["未来theta"]
                    self.status_sig[idn] = tmpjson[-1]["聚类运行状态"]
                    self.ori_target_Y[idn] = tmpjson[-1]["过去分值"]
                    self.target_Y[idn] = tmpjson[-1]["当前分值"]
                    # if tmpjson[-1]["聚类运行状态"] == "正常":
                    iterset.add(tmpjson[-1]["类内iterid"])
                else:
                    self.iter_id = 0
                    break
            self.result_id = self.result_json[-1]["id"] + 1
            # 筛选出 每个类 最大的id，且为正常的。如果对应的 类内iterid 一致 且 result 大于等于1轮cluster num，直接+1
            self.iter_id = self.result_json[-1]["类内iterid"]
            if len(iterset) == 1 and len(self.result_json) >= self.cluster_num:
                self.iter_id += 1
            self.next_cluster_id = (self.result_json[-1]["原始类id"] + 1) % self.cluster_num
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

    def one_iter(self):
        # 一次迭代, 尺度含义还原成实际尺寸
        t0 = 5
        t1 = 50

        def learn_rate(t):
            return t0 / (t + t1)

        # 步长自适应
        min_sensi = np.min(self.dim_sensitive)
        min_dist = np.min(self.distance_near)
        dim_scalar = min_sensi * min_dist
        epitheta = min_sensi * 1e-8
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
            gradient = dLoss_sgd(self.theta[idn], self.now_normal_positions[idn], self.target_Y[idn])
            self.new_theta[idn] = self.theta[idn] - learn_rate(self.iter_id) * gradient
            thetadis = np.sqrt(np.sum(np.square(self.new_theta[idn])))
            # 使hisratio 接近1，步长效果较好
            hisratio = dy / self.new_step_dist[idn]
            hisratio = hisratio /abs(hisratio) * ((abs(hisratio)-2)/10+2)
            # 阶段梯度
            # thetascalar = abs(dy / (thetadis + epitheta) / self.new_step_dist[idn])
            thetascalar = abs(hisratio / (thetadis + epitheta))
            # 随机改变方向和步长, 20%
            rdxy = np.random.rand(self.m)
            self.new_theta[idn] = self.new_theta[idn] * (0.8 + 0.2 * (rdxy - 0.5))
            if thetascalar > 1:
                # self.new_normal_positions[idn] = self.now_normal_positions[idn] - \
                #                                  self.init_step * dim_scalar * self.new_theta[
                #                                      idn] * dy / self.new_step_dist[idn] / thetadis / thetascalar
                self.new_normal_positions[idn] = self.now_normal_positions[idn] - \
                                                 self.init_step * dim_scalar * self.new_theta[
                                                     idn] * hisratio / thetadis / thetascalar
            else:
                # self.new_normal_positions[idn] = self.now_normal_positions[idn] - \
                #                                  self.init_step * dim_scalar * self.new_theta[
                #                                      idn] * dy / self.new_step_dist[idn] / thetadis
                self.new_normal_positions[idn] = self.now_normal_positions[idn] - \
                                                 self.init_step * dim_scalar * self.new_theta[
                                                     idn] * hisratio / thetadis
            # step_dis = np.sqrt(np.sum(np.square(self.new_normal_positions[idn] - self.now_normal_positions[idn])))
            self.new_step_dist[idn] = np.sqrt(
                np.sum(np.square(self.new_normal_positions[idn] - self.now_normal_positions[idn])))
            print("step_dis: {}, theta_dis: {}, thetascalar: {}, dy: {}".format(self.new_step_dist[idn], thetadis,
                                                                                thetascalar, dy))
            print("init_step: {}, dim_scalar: {}, y: {}".format(self.init_step, dim_scalar, self.target_Y[idn]))
            # 2.1 判断优秀结束 dy相对y 在 敏感尺寸的1/100 变化很小
            episilon = 1e-5  # episilon用来判断损失函数是否收敛
            oversig = abs(dy / (self.target_Y[idn] + episilon) / min_sensi / self.new_step_dist[idn] / 100)
            print(oversig)
            if oversig < episilon:
                self.status_sig[idn] = "结束"
            # 2.2 合并判断
            for i2 in range(self.cluster_num):
                if idn == i2:
                    continue
                # 2.2.1 活动点 固定点 距离//原始最近的点  判断 1/10 合并。
                new_dis = self.new_normal_positions[idn] - self.init_normal_positions[i2]
                new_dis = np.sqrt(np.sum(new_dis * new_dis))
                if new_dis / self.distance_near[idn] < 0.1:
                    # 比较不同的值合并较低分值的
                    if self.target_Y[idn] <= self.target_Y[i2] and self.target_Y[i2] != "合并":
                        self.status_sig[idn] = "合并"
                        break
                    elif self.target_Y[idn] > self.target_Y[i2] and self.target_Y[i2] != "合并":
                        self.status_sig[i2] = "合并"
                        break
                    else:
                        pass
                # 2.2.2 活动点 活动点 距离/原始最近的点  判断 1/10 合并。
                new_dis = self.new_normal_positions[idn] - self.new_normal_positions[i2]
                new_dis = np.sqrt(np.sum(new_dis * new_dis))
                if new_dis / self.distance_near[idn] < self.merge_percent:
                    # 比较不同的值合并较低分值的
                    if self.target_Y[idn] <= self.target_Y[i2] and self.target_Y[i2] != "合并":
                        self.status_sig[idn] = "合并"
                        break
                    elif self.target_Y[idn] > self.target_Y[i2] and self.target_Y[i2] != "合并":
                        self.status_sig[i2] = "合并"
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
        # 3. 保存结果 json：原始类id, 新位置序号, 旧位置, 旧分值, theta, 新位置, 新分值, 实际位置, 聚类运行状态
        # loadfile = os.path.join(bathpath, self.project_name + '.xls')
        loadfile = os.path.join(bathpath, self.project_name + '.csv')
        tmpjson = {
            "id": self.result_id,
            "原始类id": self.cluster_id[idn],
            "类内iterid": self.iter_id,
            "标准位置": list(self.now_normal_positions[idn]),
            "新标准位置": list(self.new_normal_positions[idn]),
            "当前分值": self.target_Y[idn],
            "过去分值": self.ori_target_Y[idn],
            "当前theta": list(self.theta[idn]),
            "未来theta": list(self.new_theta[idn]),
            "实际位置": list(self.target_X[idn]),
            "聚类运行状态": self.status_sig[idn],
            "耗时": usetime,
            "过去step_dis": self.old_step_dist[idn],
            "当前step_dis": self.new_step_dist[idn],
            "theta_dis": theta_dis,
            "dy": dy,
        }
        self.result_json.append(tmpjson)
        # pprint(self.result_json[-2:])
        pdobj = pd.DataFrame(self.result_json)
        # pdobj.to_excel(loadfile, sheet_name='Sheet1', index=False, header=True, encoding="utf-8")
        pdobj.to_csv(loadfile, index=False, header=True, encoding='gbk')

    def show_result(self):
        # 降维 分值分布图
        # todo: 1. 检验调整常规参数，2. 步长 合并任意一点， 3. 触壁反弹
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 红绿蓝
        colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (
            self.cluster_num // 7 + 1)
        # res = jsonpath.jsonpath(self.result_json, "$.sensor[?(@.sensor_type=='Temperature')]")
        # xy = jsonpath.jsonpath(self.result_json, "$.['实际位置']")
        xy = jsonpath.jsonpath(self.result_json, "$.['标准位置']")
        xy = list(zip(*xy))
        zs = jsonpath.jsonpath(self.result_json, "$.['当前分值']")
        clss = jsonpath.jsonpath(self.result_json, "$.['原始类id']")
        m = jsonpath.jsonpath(self.result_json, "$.['类内iterid']")
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


def dLoss_sgd(theta, X_b_i, y_i):
    '''
    单样本随机梯度下降，损失函数对theta的偏导数
    '''
    return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2


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
    2. 输出多指标，需要一个综合的公式，分值越高越好, 要求非负。
    """
    x1, x2 = nest
    # time.sleep(0.2)
    return np.sin(x1) * np.cos(x2)
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
        "init_step": 1,
        "merge_percent": 0.1,
    }
    global bathpath
    bathpath = parajson["bathpath"]
    psins = ParaSearch(fit_func, parajson)
    n = 200
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
