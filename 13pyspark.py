# -*- coding: UTF-8 -*-
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import pandas as pd
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, LongType,FloatType
import os
from pyspark.sql import Row

def main():
    # 初始化SparkSession
    spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
    sc = spark.sparkContext

    # 读文件
    # 直接读rdd转DataFrame
    fpath = os.path.join("..", "data", "tt_orderdetail7tt.csv", )
    lines = sc.textFile(fpath, minPartitions=2)
    header = lines.first()  # 第一行
    print(header)
    lines = lines.filter(lambda row: row != header)  # 删除第一行
    parts = lines.map(lambda l: l.split(","))
    linesdf = parts.map(lambda x: Row(
        OrderDetailID=x[0],
        OrderID=x[1],
        CommodityID=int(x[2]),
        Amount=int(x[3]),
        TotalPrice=float(x[4]),
        total=float(x[5]),
    ))
    lines_df = spark.createDataFrame(linesdf)
    lines_df.show()

    # demo 2
    conf = SparkConf().setMaster("local[*]").setAppName("My App")
    sc = SparkContext(conf=conf)

    # 借助pandas 转rdd
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    df = pd.read_csv(r'game-clicks.csv')
    sdf = sqlContext.createDataFrame(df)

    # 借助pandas
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    sqlContext.read.format('spark.csv').options(header='true', inferschema='true').load('game-clicks.csv')

    # 直接转RDD
    sc = SparkContext()
    lines = sc.textFile('file', minPartitions=2)
    # 第一行
    header = lines.first()
    # 删除第一行
    csvRDD = lines.filter(lambda row: row != header)
    # RDD。如果需要转换成dataframe
    schema = StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
        StructField("age", LongType(), True),
        StructField("eyeColor", StringType(), True)
    ])
    lines_df = sqlContext.createDataFrame(csvRDD, schema)  # 指定schema。
    lines_df.show()

    # hdfs上的csv文件读取
    sqlContext = SQLContext(sc)
    sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('file')

    # Map、Reduce API 基本
    sc = SparkContext('local')
    # 第二个参数2代表的是分区数，默认为1
    old = sc.parallelize([1, 2, 3, 4, 5], 2)
    newMap = old.map(lambda x: (x, x ** 2))
    newReduce = old.reduce(lambda a, b: a + b)
    print(newMap.collect())
    print(newReduce)

    # flatMap、filter、distinc API 数据的拆分、过滤和去重
    sc = SparkContext('local')
    old = sc.parallelize([1, 2, 3, 4, 5])
    # 新的map里将原来的每个元素拆成了3个
    newFlatPartitions = old.flatMap(lambda x: (x, x + 1, x * 2))
    # 过滤，只保留小于6的元素
    newFilterPartitions = newFlatPartitions.filter(lambda x: x < 6)
    # 去重
    newDiscinctPartitions = newFilterPartitions.distinct()
    print(newFlatPartitions.collect())
    print(newFilterPartitions.collect())
    print(newDiscinctPartitions.collect())

    # Sample、taskSample、sampleByKey API 数据的抽样，在机器学习中十分实用的功能，而它们有的是传输有的是动作，需要留意这个区别
    sc = SparkContext('local')
    old = sc.parallelize(range(8))
    samplePartition = [old.sample(withReplacement=True, fraction=0.5) for i in range(5)]
    for num, element in zip(range(len(samplePartition)), samplePartition):
        print('sample: %s y=%s' % (str(num), str(element.collect())))
    taskSamplePartition = [old.takeSample(withReplacement=False, num=4) for i in range(5)]
    for num, element in zip(range(len(taskSamplePartition)), taskSamplePartition):
        # 注意因为是action，所以element是集合对象，而不是rdd的分区
        print('taskSample: %s y=%s' % (str(num), str(element)))
    mapRdd = sc.parallelize([('B', 1), ('A', 2), ('C', 3), ('D', 4), ('E', 5)])
    y = [mapRdd.sampleByKey(withReplacement=False,
                            fractions={'A': 0.5, 'B': 1, 'C': 0.2, 'D': 0.6, 'E': 0.8}) for i in range(5)]
    for num, element in zip(range(len(y)), y):
        # 注意因为是action，所以element是集合对象，而不是rdd的分区
        print('y: %s y=%s' % (str(num), str(element.collect())))

    # 交集intersection、并集union、排序sortBy API
    sc = SparkContext('local')
    rdd1 = sc.parallelize(['C', 'A', 'B', 'B'])
    rdd2 = sc.parallelize(['A', 'A', 'D', 'E', 'B'])
    rdd3 = rdd1.union(rdd2)
    rdd4 = rdd1.intersection(rdd2)
    print(rdd3.collect())
    print(rdd4.collect())
    print(rdd3.sortBy(lambda x: x[0]).collect())

    # flod折叠、aggregate聚合API
    sc = SparkContext('local')
    rdd1 = sc.parallelize([2, 4, 6, 1])
    rdd2 = sc.parallelize([2, 4, 6, 1], 4)
    zeroValue = 0
    foldResult = rdd1.fold(zeroValue, lambda element, accumulate: accumulate + element)
    zeroValue = (1, 2)
    seqOp = lambda accumulate, element: (accumulate[0] + element, accumulate[1] * element)
    combOp = lambda accumulate, element: (accumulate[0] + element[0], accumulate[1] * element[1])
    aggregateResult = rdd1.aggregate(zeroValue, seqOp, combOp)
    print(foldResult)
    print(aggregateResult)
    aggregateResult = rdd2.aggregate(zeroValue, seqOp, combOp)
    print(foldResult)
    print(aggregateResult)

    # reduceByKey、 reduceByKeyLocal API
    sc = SparkContext('local')
    oldRdd = sc.parallelize([('Key1', 1), ('Key3', 2), ('Key1', 3), ('Key2', 4), ('Key2', 5)])
    newRdd = oldRdd.reduceByKey(lambda accumulate, ele: accumulate + ele)
    newActionResult = oldRdd.reduceByKeyLocally(lambda accumulate, ele: accumulate + ele)
    print(newRdd.collect())
    print(newActionResult)

    # map、reduce
    # plan 1
    sc = SparkContext('local')
    # 第二个参数2代表的是分区数，默认为1
    old = sc.parallelize([1, 2, 3, 4, 5])
    newMapRdd = old.map(lambda x: (str(x), x ** 2))
    print(newMapRdd.collect())
    mergeRdd = newMapRdd.values()
    print(mergeRdd.sum())

    # plan 2
    sc = SparkContext('local')
    oldRdd = sc.parallelize([1, 2, 3, 4, 5])
    newListRdd = oldRdd.map(lambda x: x ** 2)
    newMapRdd = oldRdd.zip(newListRdd)
    print(newMapRdd.values().sum())


if __name__ == '__main__':
    main()
    print("end")
