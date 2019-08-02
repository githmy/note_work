import urllib
import base64
import uuid
import pandas as pd
from utils.connect_mysql import MysqlDB
# import requests
import pymysql
import json
import shutil
import os
import re
import numpy as np
import urllib.request as librequest
import urllib.parse
from utils.connect_mongo import MongoDB
from utils.connect_mysql import MysqlDB
import demjson
import ast


def get_paths(source_root, target_root):
    # 1. 遍历
    list_1 = os.listdir(source_root)
    for i1 in list_1:
        i1 = i1.strip()
        list_2 = os.listdir(os.path.join(source_root, i1))
        for i2 in list_2:
            i2 = i2.strip()
            list_3 = os.listdir(os.path.join(source_root, i1, i2))
            for i3 in list_3:
                i3 = i3.strip()
                if i3.startswith("pcM_"):
                    # 输入路径
                    in_content = os.path.join(source_root, i1, i2, i3)
                    # 输出绝对路径
                    out_content = os.path.join(target_root, i1, i2)
                    out_file_full_notail = os.path.join(out_content, i3)
                    yield in_content, out_file_full_notail, out_content
                else:
                    print("content_error", i1, i2, i3)
                    raise Exception("content_error", i1, i2, i3)


def get_paths_lele(source_root, target_root):
    # 1. 遍历
    list_1 = os.listdir(source_root)
    for i1 in list_1:
        i1 = i1.strip()
        list_2 = os.listdir(os.path.join(source_root, i1))
        for i2 in list_2:
            i2 = i2.strip()
            list_3 = os.listdir(os.path.join(source_root, i1, i2))
            for i3 in list_3:
                i3 = i3.strip()
                # if i3.startswith("pcM_"):
                if i3.endswith(".mp4"):
                    de_tail = ".".join(i3.split(".")[:-1])
                    # 输入路径
                    in_content = os.path.join(source_root, i1, i2, de_tail)
                    # 输出绝对路径
                    out_content = os.path.join(target_root)
                    out_file_full_notail = os.path.join(out_content, de_tail)
                    yield in_content, out_file_full_notail, out_content
                else:
                    print("content_error", i1, i2, i3)
                    raise Exception("content_error", i1, i2, i3)


def get_special_type_iter(source_root, target_root, file_type):
    for i in get_all_iter(source_root):
        if i.endswith(file_type):
            de_tail = ".".join(i.split(".")[:-1])
            # 输入路径
            in_content = os.path.join(source_root, de_tail)
            # 输出绝对路径
            out_content = os.path.join(target_root)
            out_file_full_notail = os.path.join(out_content, de_tail)
            yield in_content, out_file_full_notail, out_content


def get_all_iter(source_root):
    for root, dirs, files in os.walk(source_root):
        for i1 in files:
            yield os.path.join(root, i1)


def get_file_l2(source_root, target_root):
    # 1. 遍历
    list_1 = os.listdir(source_root)
    for i1 in list_1:
        i1 = i1.strip()
        list_2 = os.listdir(os.path.join(source_root, i1))
        for i2 in list_2:
            i2 = i2.strip()
            list_3 = os.listdir(os.path.join(source_root, i1, i2))
            for i3 in list_3:
                i3 = i3.strip()
                if i3.startswith("pcM_"):
                    # 输入路径
                    yield os.path.join(source_root, i1, i2, i3), os.path.join(target_root, i3)
                else:
                    print("content_error", i1, i2, i3)
                    # raise Exception("content_error", i1, i2, i3)


# 正则替换数据
class Replace_data(object):
    def get_file(self):
        self.source_root = os.path.join("E:\\", "project", "ht_data", "prod")
        self.target_root = os.path.join("E:\\", "prod_new")
        if not os.path.exists(self.target_root):
            os.makedirs(self.target_root)
        self.files = []
        for i1 in os.listdir(self.source_root):
            if i1.startswith("example"):
                self.files.append(i1)

    def get_mongo(self):
        config = {
            'host': "192.168.1.252",
            'port': 27017,
            'user': "root",
            'password': "root",
            'database': "thinking2",
            'col': "reviewpoints",
        }
        mongodb = MongoDB(config)
        res = mongodb.exec_check()
        pd_mongo = pd.DataFrame(list(res))
        self.pd_mongo = pd_mongo[['_id', 'name']]

    def get_mysql(self):
        config = {
            'host': "192.168.1.252",
            'user': "thinking",
            'password': "thinking2018",
            'database': "htdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)
        req_sql = """SELECT * FROM bus_point_info;"""
        res = mysql.exec_sql(req_sql)
        pd_sql = pd.DataFrame(res)
        self.pd_sql = pd_sql[['Code', 'Name']].rename(columns={"Name": "name"})

    def get_strs(self):
        self.resstrs = []
        for i1 in self.files:
            with open(os.path.join(self.source_root, i1), encoding="utf-8") as f:
                temp = f.readlines()
                tmp_list = []
                for index in temp:
                    tmp_list.append(index)
            self.resstrs.append("".join(tmp_list))

    def replace_data(self):
        new_pd = pd.merge(self.pd_mongo, self.pd_sql, on="name", how="left")
        self.nomatch_list = []
        tmp_strs_list = []
        counter = 0
        for id1, i1 in enumerate(self.resstrs):
            tmp_strs = i1
            for i2 in new_pd.index:
                nomatchjson = {}
                if tmp_strs.find(new_pd.loc[i2]["_id"]) != -1:
                    if not pd.isnull(new_pd.loc[i2]["Code"]):
                        tmp_strs = tmp_strs.replace(new_pd.loc[i2]["_id"], new_pd.loc[i2]["Code"])
                        print("replace:{} , {} , {} , {}".format(self.files[id1], new_pd.loc[i2]["_id"],
                                                                 new_pd.loc[i2]["Code"], new_pd.loc[i2]["name"]))
                        counter += 1
                        print(counter)
                    else:
                        nomatchjson["file"] = self.files[id1]
                        nomatchjson["old"] = new_pd.loc[i2]["_id"]
                        nomatchjson["new"] = new_pd.loc[i2]["Code"]
                        nomatchjson["name"] = new_pd.loc[i2]["name"]
                        self.nomatch_list.append(nomatchjson)
            tmp_strs_list.append(tmp_strs)
        self.resstrs = tmp_strs_list

    def write_data(self):
        for i1 in zip(self.files, self.resstrs):
            with open(os.path.join(self.target_root, i1[0]), mode='w', encoding="utf-8") as f:
                # with open(os.path.join(self.target_root, i1[0]), mode='w') as f:
                f.write(i1[1])
        pdobj = pd.DataFrame(self.nomatch_list)
        tmp_csv = "not_find_point.csv"
        pdobj.to_csv(tmp_csv, index=False, header=True, encoding="utf-8")

    def __call__(self, *args, **kwargs):
        self.get_file()
        self.get_mongo()
        self.get_mysql()
        self.get_strs()
        self.replace_data()
        self.write_data()


# 正则替换数据
class Replace_js_data(object):
    def get_file(self):
        self.source_root = os.path.join("E:\\", "project", "ht_data", "prod")
        self.target_root = os.path.join("E:\\", "prod_new")
        if not os.path.exists(self.target_root):
            os.makedirs(self.target_root)
        self.files = []
        for i1 in os.listdir(self.source_root):
            if i1.startswith("example"):
                self.files.append(i1)

    def get_mongo(self):
        config = {
            'host': "192.168.1.252",
            'port': 27017,
            'user': "root",
            'password': "root",
            'database': "thinking2",
            'col': "reviewpoints",
        }
        mongodb = MongoDB(config)
        res = mongodb.exec_check()
        pd_mongo = pd.DataFrame(list(res))
        self.pd_mongo = pd_mongo[['_id', 'name']]

    def get_mysql(self):
        config = {
            'host': "192.168.1.252",
            'user': "thinking",
            'password': "thinking2018",
            'database': "htdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)
        req_sql = """SELECT * FROM bus_point_info;"""
        res = mysql.exec_sql(req_sql)
        pd_sql = pd.DataFrame(res)
        self.pd_sql = pd_sql[['Code', 'Name']].rename(columns={"Name": "name"})

    def get_strs(self):
        # pattern = re.compile(r'^.*?module\.exports.*?=')
        pattern_cont = re.compile(r'^[\d\D]*?module\.exports[\d\D]*?=([\d\D]*$)')
        pattern_illu = re.compile('(//[\\s\\S]*?\n)')
        self.resstrs = []
        for i1 in self.files:
            print(os.path.join(self.source_root, i1))
            with open(os.path.join(self.source_root, i1), encoding="utf-8") as f:
                temp = f.readlines()
                tmp_list = []
                for index in temp:
                    tmp_list.append(index)
            one_str = pattern_cont.findall("".join(tmp_list))[0].replace(";", "")
            one_str = pattern_illu.sub("", one_str)
            print(one_str)
            # py_obj = demjson.decode(one_str)
            # py_obj = json.loads(_jsonnet.evaluate_snippet('snippet', one_str))
            py_obj = ast.literal_eval(one_str)
            print(py_obj)
            # js_tmp = json.loads(one_str, encoding="utf-8")
            # print(js_tmp)
            exit(0)
            self.resstrs.append(one_str)

    def replace_data(self):
        new_pd = pd.merge(self.pd_mongo, self.pd_sql, on="name", how="left")
        self.nomatch_list = []
        tmp_strs_list = []
        for id1, i1 in enumerate(self.resstrs):
            tmp_strs = i1
            for i2 in new_pd.index:
                nomatchjson = {}
                if tmp_strs.find(new_pd.loc[i2]["_id"]) != -1:
                    if not pd.isnull(new_pd.loc[i2]["Code"]):
                        tmp_strs = tmp_strs.replace(new_pd.loc[i2]["_id"], new_pd.loc[i2]["Code"])
                        print("replace:{} , {} , {}".format(self.files[id1], new_pd.loc[i2]["_id"],
                                                            new_pd.loc[i2]["Code"]))
                    else:
                        nomatchjson["file"] = self.files[id1]
                        nomatchjson["old"] = new_pd.loc[i2]["_id"]
                        nomatchjson["new"] = new_pd.loc[i2]["Code"]
                        self.nomatch_list.append(nomatchjson)
            tmp_strs_list.append(tmp_strs)
        self.resstrs = tmp_strs_list

    def write_data(self):
        for i1 in zip(self.files, self.resstrs):
            with open(os.path.join(self.target_root, i1[0]), mode='w', encoding="utf-8") as f:
                f.write(i1[1])
        pdobj = pd.DataFrame(self.nomatch_list)
        tmp_csv = "not_find_point.csv"
        pdobj.to_csv(tmp_csv, index=False, header=True, encoding="utf-8")

    def __call__(self, *args, **kwargs):
        self.get_file()
        self.get_strs()
        self.get_mongo()
        self.get_mysql()
        self.replace_data()
        self.write_data()


# 目录结构转化为uuid列表
class Contents2uuid(object):
    def __init__(self, source_root, target_root, file_type, index_name):
        self.source_root = source_root
        self.target_root = target_root
        self.file_type = file_type
        self.index_name = index_name

    def trans_info_iter(self):
        # 1. 定义输入输出
        if not os.path.exists(self.target_root):
            os.makedirs(self.target_root)
        reslist = []
        # # 2. 信息获取
        # 原始信息
        for i1 in get_special_type_iter(self.source_root, self.target_root, self.file_type):
            strs = i1[0].replace(self.source_root, "").lstrip("\\")
            uuid1 = self.strs2uuid(strs)
            tmplist = strs.split("\\")
            tmpjson = {}
            dirlenth = len(tmplist)
            for i2 in range(dirlenth - 1):
                tmpjson["dir" + str(i2 + 1)] = tmplist[i2]
            tmpjson["文件代号"] = uuid1
            tmpjson["知识点代号"] = tmplist[dirlenth - 1]
            tmpjson["我们的知识点"] = None
            reslist.append(tmpjson)
            if os._exists(i1[2] + uuid1 + self.file_type):
                raise Exception(i1[2] + uuid1 + self.file_type)
            self.copy_file2content(i1[0] + self.file_type, os.path.join(i1[2], uuid1 + self.file_type))
        # 3. 输出数据
        pdobj = pd.DataFrame(reslist)
        pdobj.to_excel(os.path.join(self.target_root, "..", self.index_name), sheet_name='Sheet1', index=False,
                       header=True, encoding="utf-8")

    def list_content2excel(self):
        # 1. 定义输入输出
        reslist = []
        # # 2. 信息获取
        for i1 in get_all_iter(self.source_root):
            print(self.source_root)
            print(i1)
            strs = i1.replace(self.source_root, "").lstrip("\\")
            print(strs)
            uuid1 = self.strs2uuid(strs)
            tmplist = strs.split("\\")
            tmpjson = {}
            dirlenth = len(tmplist)
            for i2 in range(dirlenth - 1):
                tmpjson["dir" + str(i2 + 1)] = tmplist[i2]
            tmpjson["文件代号"] = uuid1
            tmpjson["知识点代号"] = tmplist[dirlenth - 1]
            tmpjson["我们的知识点"] = None
            reslist.append(tmpjson)
            print(tmpjson)
        # 3. 输出数据
        pdobj = pd.DataFrame(reslist)
        # new_pd = pd.merge(pdobjhead, pdobj, on="video_urlid", how="right")
        pdobj.to_excel(os.path.join(self.source_root, "..", self.index_name), sheet_name='Sheet1', index=False,
                       header=True, encoding="utf-8")

    def trans_info(self):
        # 1. 定义输入输出
        # source_root = os.path.join("D:\\", "video_data", "headtail")
        # source_root = os.path.join("E:\\", "project", "data", "spider", "data", "down")
        source_root = os.path.join("D:\\", "video_data", "乐乐课堂小学奥数1-6年级")
        target_root = os.path.join("D:\\", "video_data", "乐乐小学奥数UUID")
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        # print(source_root)
        tmpo_path = "video_uuid.csv"
        reslist = []
        # # 2. 信息获取
        # tmpheadfile = os.path.join("E:\\", "project", "data", "spider", "data", "视频知识点罗列.xlsx")
        # pdobjhead = pd.read_excel(tmpheadfile, sheet_name='Sheet1', header=0)
        # pdobjhead.drop_duplicates(subset='video_urlid', keep='first', inplace=True)
        # 原始信息
        for i1 in get_paths_lele(source_root, target_root):
            strs = i1[0].replace(source_root, "").lstrip("\\")
            uuid1 = self.strs2uuid(strs)
            tmplist = strs.split("\\")
            tmpjson = {
                "一级目录": tmplist[0],
                "二级目录": tmplist[1],
                "视频内容": tmplist[2],
                "文件代号": uuid1,
                "我们的知识点": None,
            }
            reslist.append(tmpjson)
            if os._exists(i1[2] + uuid1 + ".mp4"):
                raise Exception(i1[2] + uuid1 + ".mp4")
            self.copy_file2content(i1[0] + ".mp4", os.path.join(i1[2], uuid1 + ".mp4"))
        # 3. 输出数据
        pdobj = pd.DataFrame(reslist)
        # new_pd = pd.merge(pdobjhead, pdobj, on="video_urlid", how="right")
        pdobj.to_excel(os.path.join(target_root, "..", '乐乐小学奥数uuid.xls'), sheet_name='Sheet1', index=False, header=True,
                       encoding="utf-8")

    def strs2uuid(self, strs):
        namespace = uuid.NAMESPACE_URL
        return str(uuid.uuid3(namespace, strs))

    def rename_ondict2uuid(self, strs):
        namespace = uuid.NAMESPACE_URL
        return str(uuid.uuid3(namespace, strs))

    def copy_file2content(self, sourcefile, targetfile):
        exec_str = 'copy "' + sourcefile + '" "' + targetfile + '"'
        print(exec_str)
        os.system(exec_str)

    def __call__(self, *args, **kwargs):
        # self.list_content2excel()
        self.trans_info_iter()
        # self.trans_info()
        # self.sql2pd()
        # self.replace_data()


# 三级目录去重到一级目录
class Deduplicate_video(object):
    def sql_unique_name(self):
        config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "333",
            'database': "ycdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)
        req_sql = """SELECT seed_urls, urldir FROM seed_point_all WHERE urldir LIKE "pcM_%" AND seed_urls LIKE "%.m3u8" GROUP BY seed_urls;"""
        res = mysql.exec_sql(req_sql)
        self.pd_sql = pd.DataFrame(res)
        print(self.pd_sql)

    def iter_dir_cope(self):
        source_root = os.path.join("D:\\", "video_data", "merged")
        target_root = os.path.join("D:\\", "video_data", "deduplicate")
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        res = get_file_l2(source_root, target_root)
        for i1 in res:
            exec_str = 'copy "' + i1[0] + '" "' + i1[1] + '"'
            print(exec_str)
            os.system(exec_str)

    def __call__(self, *args, **kwargs):
        self.sql_unique_name()
        self.iter_dir_cope()


# 关联知识点
class Relation_points(object):
    def sql_unique_name(self):
        config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "333",
            'database': "ycdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)
        req_sql = """
            SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get from (
                (
                    SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM (
                    (SELECT subjects,publisher,stage,semester,urlconnect FROM `main_class` WHERE subjects='数学' and stage='初中') l1 
                    LEFT JOIN 
                    (SELECT father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM `big_video_class`) r1 
                    ON l1.urlconnect=r1.father_url
                    )
                ) l2
                RIGHT JOIN 
                (SELECT chapter_url,seed_urls,urldir,points,had_get FROM `seed_point_all` WHERE seed_urls LIKE '%.m3u8' and urldir LIKE 'pcM_%' GROUP BY seed_urls) r2 
                ON l2.video_url=r2.chapter_url
            ) WHERE had_get=1;"""
        res = mysql.exec_sql(req_sql)
        self.pd_sql = pd.DataFrame(res)
        print(self.pd_sql)

    def iter_dir_cope(self):
        self.pd_sql.to_excel('video_uuid.xls', sheet_name='Sheet1', index=False, header=True, encoding="utf-8")

    def __call__(self, *args, **kwargs):
        self.sql_unique_name()
        self.iter_dir_cope()


class Find_not_down(object):
    def sql_unique_name(self):
        config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "333",
            'database': "ycdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)
        req_sql = """SELECT * FROM seed_point_all WHERE urldir LIKE "pcM_%" AND seed_urls LIKE "%.m3u8" GROUP BY seed_urls;"""
        res = mysql.exec_sql(req_sql)
        self.pd_sql = pd.DataFrame(res)

    def list_not_down(self):
        source_root = os.path.join("E:\\", "project", "data", "spider", "data", "down")
        target_root = os.path.join("D:\\", "video_datatt")

        res = get_paths(source_root, target_root)
        print(111)
        list_1 = [os.path.split(i1[0])[1] for i1 in res]
        # list_1 = os.listdir(target_root)
        print(list_1)

        def dusig(strs):
            for i1 in list_1:
                if i1.startswith(strs):
                    return 1

        self.pd_sql["dusig"] = self.pd_sql["urldir"].map(dusig)
        print(self.pd_sql[self.pd_sql["dusig"] != 1])

    def __call__(self, *args, **kwargs):
        self.sql_unique_name()
        self.list_not_down()


class Compare_not_have(object):
    def sql_unique_name(self):
        config = {
            'host': "127.0.0.1",
            'user': "root",
            'password': "333",
            'database': "ycdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        mysql = MysqlDB(config)
        req_sql = """
            SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get from (
                (
                    SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM (
                    (SELECT subjects,publisher,stage,semester,urlconnect FROM `main_class` WHERE subjects='数学' and stage='初中') l1 
                    LEFT JOIN 
                    (SELECT father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM `big_video_class`) r1 
                    ON l1.urlconnect=r1.father_url
                    )
                ) l2
                RIGHT JOIN 
                (SELECT chapter_url,seed_urls,urldir,points,had_get FROM `seed_point_all` WHERE seed_urls LIKE '%.m3u8' and urldir LIKE 'pcL_%' and had_get=0 GROUP BY seed_urls) r2 
                ON l2.video_url=r2.chapter_url
            ) WHERE had_get IS NOT NULL AND video_url IS NOT NULL ;
        """
        res = mysql.exec_sql(req_sql)
        self.pd_sql = pd.DataFrame(res)

    def content_notin_filenames(self):
        source_root = os.path.join("D:\\", "video_data", "deduplicate")
        target_root = os.path.join("E:\\", "project", "data", "spider", "data", "down_append_pcl")

        list_s = os.listdir(source_root)
        names_old = []
        for i1 in list_s:
            names_old.append(re.split(r'[_\.]', i1)[1])
        list_t = os.listdir(target_root)
        names_new = []
        for i1 in list_t:
            names_new.append(i1.split("_")[1])
        self.new_list = [i1 for i1 in names_new if i1 not in names_old]

    def move_not_in2(self):
        source_root = os.path.join("E:\\", "project", "data", "spider", "data", "down_append_pcl")
        target_root = os.path.join("D:\\", "video_data", "append_pcl")
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        for i1 in self.new_list:
            print(os.path.join(source_root, "pcL_" + i1), target_root)
            shutil.copytree(os.path.join(source_root, "pcL_" + i1), os.path.join(target_root, "pcL_" + i1))

    def __call__(self, *args, **kwargs):
        self.content_notin_filenames()
        self.move_not_in2()


def main():
    # # inst = Replace_js_data()
    # # inst = Replace_data()
    # filetype = "mp4"
    # contentname = "初中数学同步视频"
    # source_root = os.path.join("D:\\", "video_data", contentname)
    # target_root = os.path.join("D:\\", "video_data", "{}{}".format(contentname, filetype))
    # index_name = '{}{}.xls'.format(contentname, filetype)
    # inst = Contents2uuid(source_root, target_root, ".{}".format(filetype), index_name)
    # # inst = Deduplicate_video()
    # # inst = Find_not_down()
    #
    inst = Relation_points()
    # inst = Compare_not_have()
    inst()
    print("转换结束！")


# 替换知识点
if __name__ == '__main__':
    main()
