# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

import pymongo
import threading
import logging
import os
import json
import time


class MongoDB:
    def __init__(self, conf):
        self.config = {
            'host': conf['host'],
            'port': 27017,
            # 'user': conf['user'],
            # 'password': conf['password'],
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

    def exec_require(self, quire_list):
        # 显示查询结果
        mydoc = {}
        for col in quire_list:
            mydoc[col] = self.mydb[col].find()
        return mydoc

    def exec_check(self):
        # 显示查询结果
        myquery = {"name": {"$gt": "H"}}  # 第一个ascii>H
        mydoc = self.mycol.find(myquery)
        return mydoc

    def exec_demo(self):
        # 显示库名
        dblist = self.myclient.list_database_names()
        print(dblist)
        # 显示 表 集合 名
        collist = self.myclient[dblist[0]].list_collection_names()
        print(collist)
        # 查询
        # x = self.mycol.find_one()
        # print(x)
        # for x in self.mycol.find():
        #     print(x)
        # # 返回字段置为1，非返还置零。
        # for x in self.mycol.find({}, {"_id": 0}):
        #     print(x)
        # # 条件查询
        # myquery = {"name": "RUNOOB"}
        # mydoc = self.mycol.find(myquery)
        # for x in mydoc:
        #     print(x)
        # 正则查询
        # {"$regex": "^R"} 第一个字母为R
        myquery = {"name": {"$gt": "H"}}  # 第一个ascii>H
        mydoc = self.mycol.find(myquery)
        for x in mydoc:
            print(x)


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
