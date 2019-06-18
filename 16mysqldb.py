# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

import pymysql
import logging
import threading
import os
import json
import time
import pandas as pd


class MysqlDB:
    def __init__(self, conf):
        self.config = {
            'host': conf['host'],
            'user': conf['user'],
            'password': conf['password'],
            'database': conf['database'],
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.conn = pymysql.connect(**self.config)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()
        # Log.sql_log.(error|info|debug)

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def exec_sql(self, strsql):
        """
        write data, such as insert, delete, update
        :param strsql: string sql
        :return: affected rows number
        return 0 when errors
        """
        try:
            self.lock.acquire()
            self.conn.ping(True)
            res = self.cursor.execute(strsql)
            res = self.cursor.fetchall()
            # if strsql.strip().lower().startswith("select"):
            #     res = self.cursor.fetchall()
            self.conn.commit()
            return res
        except Exception as ex:
            logging.error("exec sql error:")
            logging.error(strsql, exc_info=True)
            return 0
        finally:
            self.lock.release()


if __name__ == '__main__':
    config = {
        'host': "192.168.1.252",
        'user': "thinking",
        'password': "thinking2018",
        'database': "thinkingdb",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
        'cursorclass': pymysql.cursors.DictCursor
    }
    mysql = MysqlDB(config)

    service_count_sql = """SELECT id,description,`level` FROM `crawler_example_info`"""
    service_count = mysql.exec_sql(service_count_sql)
    predict_pd = pd.DataFrame(service_count)
    tmpo_path = os.path.join("..", "data", "thinking2", "predict_obj.csv")
    predict_pd.to_csv(tmpo_path, index=False, header=True, encoding="utf-8")
    print(predict_pd)

    # tt_sql = """desc `crawler_example_info`"""
    # repp = mysql.exec_sql(tt_sql)
    # print(pd.DataFrame(repp))
