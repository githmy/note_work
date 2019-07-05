# !/usr/bin/env python
import pymysql
import os
import logging
import datetime
import glob
import threading
import redis
from builtins import object
from threading import Lock
from typing import Text, List
from ai_sets.config import AisetsConfig

logger = logging.getLogger(__name__)


class Datadb(object):
    def __init__(self, config=None):
        self._config = {
            'host': config["db_ad"]['host'],
            'user': config["db_ad"]['user'],
            'password': config["db_ad"]['password'],
            'database': config["db_ad"]['database'],
            # 'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.conn = pymysql.connect(**self._config)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()

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

    def show_unit(self):
        get_module_id1_sql = """SELECT * from aidb.unit_tb;"""
        return self.exec_sql(get_module_id1_sql)

    def unit_config(self, config):
        get_module_id1_sql = """SELECT config from aidb.unitconfig_tb WHERE unitid =%s AND branch =%s AND userid =%s;""" % (
            str(config["project"]), str(config["branch"]), str(config["userid"]))
        # print(get_module_id1_sql)
        return self.exec_sql(get_module_id1_sql)

    def config_tree(self):
        get_module_id1_sql = """SELECT tree from aidb.config_tree_tb;"""
        return self.exec_sql(get_module_id1_sql)

    def config_rule(self, colum):
        get_module_id1_sql = """SELECT %s from aidb.config_rule_tb;""" % colum
        # print(get_module_id1_sql)
        return self.exec_sql(get_module_id1_sql)

    def func_env(self):
        get_module_id1_sql = """SELECT func_name,env_id from aidb.func_tb;"""
        # print(get_module_id1_sql)
        return self.exec_sql(get_module_id1_sql)

    def write_status(self, content, config):
        get_module_id1_sql = """update unitconfig_tb set status='%s' where unitid=%s and branch=%s and userid=%s;""" % (
            content, config["project"], config["branch"], config["userid"])
        return self.exec_sql(get_module_id1_sql)

    def write_perform(self, content, config):
        get_module_id1_sql = """update unitconfig_tb set performance='%s' where unitid=%s and branch=%s and userid=%s;""" % (
            content, config["project"], config["branch"], config["userid"])
        return self.exec_sql(get_module_id1_sql)

    def write_unit_info(self, nickname, content):
        get_module_id1_sql = """UPDATE func_tb set l1.structure_info='%s' WHERE tmpname='%s';""" % (
            content, nickname)
        # print(get_module_id1_sql)
        return self.exec_sql(get_module_id1_sql)

    def check_func_info(self, nickname):
        get_module_id1_sql = """select func_name from func_tb WHERE func_name='%s';""" % (nickname)
        # print(get_module_id1_sql)
        return self.exec_sql(get_module_id1_sql)

    def insert_func_info(self, nickname, envid, content):
        get_module_id1_sql = """INSERT INTO func_tb (func_name,env_id,structure_info) VALUES ('%s','%s','%s');""" % (
            nickname, envid, content)
        # print(get_module_id1_sql)
        return self.exec_sql(get_module_id1_sql)

    def update_func_info(self, nickname, envid, content):
        get_module_id1_sql = """UPDATE func_tb set structure_info='%s',env_id='%s' WHERE func_name='%s';""" % (
            content, envid, nickname)
        # print(get_module_id1_sql)
        return self.exec_sql(get_module_id1_sql)

    def testss(self):
        get_module_id1_sql = """show databases;"""
        return self.exec_sql(get_module_id1_sql)


class RDdb(object):
    def __init__(self, config=None):
        self._config = {
            'host': config["buffer_ad"]['host'],
            'db': config["buffer_ad"]['database'],
            'password': config["buffer_ad"]['password'],
            'port': config["buffer_ad"]['port'],
        }
        # self.conn = redis.Redis(host='localhost', password="eeee",port=6379, db=0)
        self.conn = redis.Redis(**self._config)

    def show_keys(self):
        a = self.conn.keys()
        print(a)

    def show_env(self):
        return self.conn.keys("env__*")

    def set_env(self, ip, port, env_id):
        a = self.conn.set("env__" + str(ip) + ":" + str(port), str(env_id))

    def del_env(self, env_obj):
        self.conn.delete(env_obj)

    def get_env(self, env_obj):
        return self.conn.get(env_obj)
