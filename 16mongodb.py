# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

import pymongo
import pandas as pd
import logging
import os
import json
import time


class MongoDB:
    def __init__(self, conf):
        self.config = {
            'host': conf['host'],
            'port': 27017,
            'user': conf['user'],
            'password': conf['password'],
            'database': conf['database'],
            'col': conf['col'],
            'charset': 'utf8mb4',  # 支持1-4个字节字符
        }
        # Log.sql_log.(error|info|debug)
        self.myclient = pymongo.MongoClient("mongodb://%s:%s/" % (self.config["host"], self.config["port"]))
        self.mydb = self.myclient[self.config["database"]]
        self.mycol = self.mydb[self.config["col"]]

    def __del__(self):
        pass

    def exec_sort(self):
        # 排序，默认升序，-1为降序
        mydoc = self.mycol.find().sort("alexa", -1)
        for x in mydoc:
            print(x)

    def exec_add(self, strsql):
        # 插入单条
        mydict = {"name": "RUNOOB", "alexa": "10000", "url": "https://www.runoob.com"}
        x = self.mycol.insert_one(mydict)
        print(x.inserted_id)
        # 插入多条
        mylist = [
            {"name": "Taobao", "alexa": "100", "url": "https://www.taobao.com"},
            {"name": "QQ", "alexa": "101", "url": "https://www.qq.com"},
        ]
        x = self.mycol.insert_many(mylist)
        print(x.inserted_ids)

    def exec_del(self):
        # 删除
        myquery = {"name": "Taobao"}
        self.mycol.delete_one(myquery)
        # 删除后输出
        for x in self.mycol.find():
            print(x)
        myquery = {"name": {"$regex": "^F"}}
        x = self.mycol.delete_many(myquery)
        print(x.deleted_count, "个文档已删除")
        # 删表
        self.mycol.drop()

    def exec_upd(self, strsql):
        # 修改
        myquery = {"name": {"$regex": "^F"}}
        newvalues = {"$set": {"alexa": "123"}}
        x = self.mycol.update_one(myquery, newvalues)
        x = self.mycol.update_many(myquery, newvalues)
        print(x.modified_count, "文档已修改")

    def exec_check(self):
        # 显示库名
        # dblist = self.myclient.list_database_names()
        # print(dblist)
        # 查询
        # x = self.mycol.find_one()
        # print(x)
        question_obj = []
        print(999999)
        for i1, x in enumerate(self.mycol.find()):
            tmpjson = {}
            print(x)
            try:
                if x["preview"]["text"] is not "":
                    tmpjson["text"] = x["preview"]["text"]
                elif x["description"]["text"] is not "":
                    tmpjson["text"] = x["description"]["text"]
                else:
                    pass
            except Exception as e:
                if x["description"]["text"] is not "":
                    tmpjson["text"] = x["description"]["text"]
                else:
                    pass
            tmpjson["id"] = x["_id"]
            tmpjson["text"] = tmpjson["text"].replace("\n", "。")
            tmpjson["mainReviewPoints"] = x["mainReviewPoints"]
            tmpjson["reviewPoints"] = x["reviewPoints"]
            tmpjson["qType1"] = x["qType1"]
            tmpjson["keywords"] = x["keywords"]
            # tmpjson["preview"] = {}
            # tmpjson["steps"] = []

            try:
                if x["level"] is None:
                    tmpjson["level"] = 1
                else:
                    tmpjson["level"] = x["level"]
                    # print(x["level"])
            except Exception as e:
                tmpjson["level"] = 1
            question_obj.append(tmpjson)
        # 存文件
        question_pd = pd.DataFrame(question_obj)
        tmpo_path = os.path.join("..", "data", "thinking2", "question_obj.csv")
        question_pd.to_csv(tmpo_path, index=False, header=True, encoding="utf-8")
        # 存文件
        self.mycol2 = self.mydb["reviewpoints"]
        review_obj = [x for x in self.mycol2.find()]
        review_pd = pd.DataFrame(review_obj)
        tmpo_path = os.path.join("..", "data", "thinking2", "review_obj.csv")
        review_pd.to_csv(tmpo_path, index=False, header=True, encoding="utf-8")

        # for x in self.mycol.find({}, {"_id": 0}):
        #     print(x)
        # # 条件查询
        # myquery = {"name": "RUNOOB"}
        # mydoc = self.mycol.find(myquery)
        # for x in mydoc:
        #     print(x)
        # 正则查询
        # {"$regex": "^R"} 第一个字母为R
        # myquery = {"name": {"$gt": "H"}}  # 第一个ascii>H
        # mydoc = self.mycol.find(myquery)
        # for x in mydoc:
        #     print(x)


if __name__ == '__main__':
    config = {
        'host': "192.168.1.252",
        'port': 27017,
        'user': "root",
        'password': "root",
        'database': "thinking2",
        'col': "questions",
    }
    mongodb = MongoDB(config)
    mongodb.exec_check()
