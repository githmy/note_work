# -*- coding: UTF-8 -*-

# from pyspark import SparkContext


def get_data():
    print("get_data")


def gene_model():
    print("gene")
    real_info = {
        "rack_num": 100,
        "cell_num": 8,
        "cell_volume": 1000,
    }
    config = {
        "gene": {
            "dna_lenth": real_info["rack_num"] * real_info["cell_num"]
        },
        "env": {},
        "loss_func": {},

    }


def deal_data():
    print("deal_data")
    # composite_1 = [a1*sku_1..an*sku_n]
    # 单客户
    # customer_1 = [composite_1..composite_m] = [b1*sku_1..bn*sku_l]
    # all_data = [customer_1..customer_o] = [composite_1..composite_p] = [b1*sku_1..bn*sku_q]
    # 1. 基本kmeans knn 决策树。每个sku(大类)为一个维度，对订单做聚类。
    # 2. 遗传算法 趋势。神经网络 cell rack 权重。
    # 2.1 起始时间，迭代次数
    # 2.2 续接时间，迭代次数
    # 2.3 是否续接
    gene_model()
    # 3. 规则


def plot_data():
    print("plot_data")
    # 1. 总时间段不同类的统计
    # 2. 每年的递变,同期递变
    # 3. 总时间递变
    # 4. 属性的数量，一级二级分类


def main():
    get_data()
    deal_data()
    plot_data()


if __name__ == '__main__':
    main()
    print("end")
