# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

import pymysql
import threading
import logging
import os
import json
import time

def init_logging(log_path='/data/dcos/data/py_script', debug=False):
    # 检查是否存在目录
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    fmt = "%(asctime)-15s [%(pathname)s] [%(lineno)d] [%(process)d] [%(levelname)s] [%(message)s]"

    log_formatter = logging.Formatter(fmt)

    error_handler_path = os.path.join(log_path, 'error.log')
    info_handler_path = os.path.join(log_path, 'info.log')

    error_handler = logging.FileHandler(error_handler_path)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_formatter)

    info_handler = logging.FileHandler(info_handler_path)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(info_handler)
    root_logger.addHandler(error_handler)

    if debug:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(log_formatter)
        root_logger.addHandler(console)


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
            if strsql.strip().lower().startswith("select"):
                res = self.cursor.fetchall()
            self.conn.commit()
            return res
        except Exception as ex:
            logging.error("exec sql error:")
            logging.error(strsql, exc_info=True)
            return 0
        finally:
            self.lock.release()


if __name__ == '__main__':
    init_logging()
    while True:
        try:
            config = {
                'host': "127.0.0.1",
                'user': "root",
                'password': "root",
                'database': "dcos_cmdb",
                'charset': 'utf8mb4',  # 支持1-4个字节字符
                'cursorclass': pymysql.cursors.DictCursor
            }
            mysql = MysqlDB(config)
            service_count_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info`"""
            service_count_phy_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info` WHERE svr_type ='0';"""
            rack_countrack_count_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info` GROUP BY svr_rack_name;"""

            service_count = mysql.exec_sql(service_count_sql)
            service_count_phy = mysql.exec_sql(service_count_phy_sql)
            rack_countrack_count = mysql.exec_sql(rack_countrack_count_sql)
            data = {
                'service_count': service_count[0].get('cot', 0) if service_count else 3500,
                'service_count_phy': service_count_phy[0].get('cot', 0) if service_count_phy else 1000,
                'rack_countrack_count': len(rack_countrack_count) if rack_countrack_count else 50,
                'tenant_count': 24,
            }
        except:
            data = {
                'service_count': 3500,
                'service_count_phy': 1000,
                'rack_countrack_count':  50,
                'tenant_count': 24,
            }
            logging.error("unknown error", exc_info=True)

        try:
            with open('./cmdb_count.json', 'w') as fp:
                fp.write(json.dumps(data))
            logging.info("wrire data success")
        except:
            logging.error("wrire data fail", exc_info=True)

        time.sleep(600)


