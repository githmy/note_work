# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy.exceptions import DropItem
import pandas as pd
import os
import urllib.request as librequest
import pymysql
import threading
from lele.items import LeleItem


data_path = os.path.join("..", "..", "..", "data", "spider", "data", "lele")
if not os.path.exists(data_path):
    os.makedirs(data_path)


class LelePipeline(object):
    def __init__(self):
        self.config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "333",
            'database': "leledb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.conn = pymysql.connect(**self.config)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def _exec_sql(self, strsql):
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
            print("exec sql error:")
            print(strsql)
            return 0
        finally:
            self.lock.release()

    def open_spider(self, spider):
        self.all_list = []
        write_sql = """TRUNCATE `main_class`"""
        have_num = self._exec_sql(write_sql)

    def close_spider(self, spider):
        pdobj = pd.DataFrame(self.all_list)
        # file_path = os.path.join("..", "..", "data", "spider", "data", "lele")
        file_name = os.path.join(data_path, "instruction.csv")
        pdobj.to_csv(file_name, encoding='utf-8', index=False)

    # 常规流程函数
    def process_item(self, item, spider):
        print("in file")
        # if isinstance(item, LeleItem):
        #     print(item)
        if item.get('url'):
            print(item)
            # 1. 写入列表
            self.all_list.append(item)
            # 2. 数据库记录
            write_sql = """insert into `main_class` (a_item_1,a_href_1,a_item_2,a_href_2,url) VALUES ("{}","{}","{}","{}","{}");""".format(
                item.get("a_item_1"), item.get("a_href_1"), item.get("a_item_2"), item.get("a_href_2"), item.get("url"))
            print(write_sql)
            have_num = self._exec_sql(write_sql)
            print(have_num)
            # 3. 下载
            # self._down_item(item.get("url"))
            return item
        else:
            raise DropItem("Missing price in %s" % item)
            # print(item)

    def _down_item(self, url):
        response = librequest.urlopen(url, timeout=60)
        cat_vido = response.read()
        print("downloading: ", url)
        seed_name = "_".join(url.split("/")[3:])
        # 2.1 种子保存
        with open(os.path.join(data_path, seed_name), 'wb') as f:
            f.write(cat_vido)
        have_num_sql = """UPDATE `main_class` SET had_get="{}" WHERE url="{}";""".format(1, url)
        print(have_num_sql)
        have_num = self._exec_sql(have_num_sql)
        print(have_num)
