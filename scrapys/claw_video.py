import copy
import itertools
import json
import pymysql
import os
import time
import pandas as pd
import simplejson
from urllib import request
import urllib.request as librequest
from utils.log_tool_claw import logger, data_path
from utils.connect_mysql import MysqlDB


class Yangcong():
    def __init__(self, mysql_ins):
        self.db_conn = mysql_ins
        self._aheads = {}
        self.url_list = {
            "log_url": "https://api-v5-0.yangcong345.com/login",
            "main_class": "https://school-api.yangcong345.com/course/subjects",
            # "user_current": "https://api-v5-0.yangcong345.com/user-current-textbook",  # 临时忽略
            # 包含 pre_video 的key
            # 视频网页的显示要点
            "current_chapters": "https://school-api.yangcong345.com/course/chapters-with-section/publisher/{}/semester/{}/subject/{}/stage/{}",
            # # 视屏里的穿插话题。
            # "theme": "https://school-api.yangcong345.com/course/problems/fc96aa6c-6c28-11e7-92b1-7f399bce3897/1",
            # # 视频网页的纯key映射不含要点，可以不管
            # "video_summary": "https://api-v5-0.yangcong345.com/progresses?subjectId=1&publisherId=1&semesterId=13&stageId=2",
            # # # 参数
            # # subjectId: 1
            # # publisherId: 1
            # # semesterId: 13
            # # stageId: 2
            # # 返回 课题类型 概念题 id
            # 返回gene_seed的url地址。
            # "pre_video": "https://school-api.yangcong345.com/course/course-tree/themes/0b315a90-57f7-11e7-aca6-c3badc9742d1",
            "pre_video": "https://school-api.yangcong345.com/course/course-tree/themes/{}",
            "seed_list": [],
            "gene_seed": "https://hls.media.yangcong345.com/pcM/pcM_58c26cbb36eaf35866aae116.m3u8",
        }
        self._iplist = [
            "",
            "",
        ]
        self._agentlist = [
            "Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Safari/535.19",
            "Mozilla/5.0 (Linux; U; Android 4.0.4; en-gb; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30",
            "Mozilla/5.0 (Linux; U; Android 2.2; en-gb; GT-P1000 Build/FROYO) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
            "Mozilla/5.0 (Android; Mobile; rv:14.0) Gecko/14.0 Firefox/14.0",
            "Mozilla/5.0 (Windows NT 6.2; WOW64; rv:21.0) Gecko/20100101 Firefox/21.0",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36",
            "Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19",
            "Mozilla/5.0 (iPod; U; CPU like Mac OS X; en) AppleWebKit/420.1 (KHTML, like Gecko) Version/3.0 Mobile/3A101a Safari/419.3",
            "Mozilla/5.0 (iPad; CPU OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A334 Safari/7534.48.3",
        ]

    # 获取视频基本信息
    def get_basic_info2db(self):
        # 当前主列表集合
        self._main_current_urls = []
        # 当前页面视频前页集合
        self._pre_video_urls = []
        self._login()
        self._get_main_list()
        self._get_current_page()
        self._get_seeds_page()

    # 获取视频基本信息
    def download_normal_usedb(self):
        # 1. 读取下载地址
        self._download_urls_and_info = []
        self._readdb_seed_all_format()
        for id1, i1 in enumerate(self._download_urls_and_info):
            # if id1 == 3000:
            #     exit(0)
            print()
            print(id1, i1)
            self._download_ke(i1)

    def _download_ke(self, info_array):
        # subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get
        head_urls = "https://hls.media.yangcong345.com/pcM/"
        # link_demo = "https://hls.media.yangcong345.com/pcM/pcM_58c26cbb36eaf35866aae1161.ts"
        # 1. 如果不存在则创建目录
        video_path = os.path.join(data_path, "down", info_array["video_urlid"],
                                  copy.deepcopy(info_array["points"]).replace("|", "（竖线）").strip(" "),
                                  info_array["urldir"])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        request_list = []
        f_list = []
        # 2. 下载种子
        logger.info("downloading: {}".format(info_array["seed_urls"]))
        response = librequest.urlopen(info_array["seed_urls"])
        cat_vido = response.read()
        seed_name = info_array["urldir"] + ".m3u8"
        # 2.1 种子保存
        with open(os.path.join(video_path, seed_name), 'wb') as f:
            f.write(cat_vido)
        # 2.2 种子读取
        with open(os.path.join(video_path, seed_name), "r") as f:
            contentlists = f.readlines()
        # 2.3 种子解析
        for i1 in contentlists:
            if not i1.startswith("#") and i1.strip() != "":
                f_list.append(i1.strip())
                request_list.append(head_urls + i1.strip())
        # 3. 下载
        try:
            for i1 in zip(request_list, f_list):
                response = librequest.urlopen(i1[0], timeout=60)
                cat_vido = response.read()
                # 5. 写入本地文件
                with open(os.path.join(video_path, i1[1]), 'wb') as f:
                    f.write(cat_vido)
            # 5. 数据库状态写入
            self.write2db_down_success(info_array["video_url"], info_array["points"], info_array["urldir"])
        except Exception as e:
            logger.info(e)

    def _login(self):
        post_data = dict(
            name="18721986267",
            password="fff111QQQ",
        )
        endata = bytes(json.dumps(post_data), "utf-8")
        request_headers = {"content-type": "application/json"}
        req = librequest.Request(url=self.url_list["log_url"], data=endata, method='POST', headers=request_headers)
        with librequest.urlopen(req) as response:
            ori_page = response.read().decode('utf-8')
            self._aheads["Authorization"] = response.headers["Authorization"]
            the_page0 = simplejson.loads(ori_page)
            logger.info("login json: %s" % the_page0)
            self.headers = {
                "Accept": "application/json",
                "Authorization": self._aheads["Authorization"],
                "client-type": "pc",
                "client-version": "6.7.4",
                "device": 2630311005440102,
                "Referer": "https://yangcong345.com/",
                "Sec-Fetch-Mode": "cors",
                "User-Agent": "Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1",
            }

    def _get_main_list(self):
        req = librequest.Request(url=self.url_list["main_class"], method='GET', headers=self.headers)
        with librequest.urlopen(req) as response:
            ori_page = response.read().decode('utf-8')
            the_page0 = simplejson.loads(ori_page)

            # subjects,subjectsid,stage,stageid,publisher,publisherid,semester,semesterid
            def json2list(json):
                listres = []
                for i1 in json:
                    th1 = [i1["name"], i1["id"]]
                    for i2 in i1["stages"]:
                        th2 = copy.deepcopy(th1)
                        th2.extend([i2["name"], i2["id"]])
                        for i3 in i2["publishers"]:
                            th3 = copy.deepcopy(th2)
                            th3.extend([i3["name"], i3["id"]])
                            for i4 in i3["semesters"]:
                                th4 = copy.deepcopy(th3)
                                th4.extend([i4["name"], i4["id"]])
                                listres.append(th4)
                return listres

            reslist = json2list(the_page0)
            self.write2db_main_format(reslist)
            logger.info("main class json: %s" % the_page0)

    def _get_current_page(self):
        self._readdb_main_format()
        for i1 in self._main_current_urls:
            req = librequest.Request(url=i1, method='GET', headers=self.headers)
            with librequest.urlopen(req) as response:
                ori_page = response.read().decode('utf-8')
                the_page0 = simplejson.loads(ori_page)

                # father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid
                def json2list(fatherurl, url_head, json):
                    listres = []
                    for i1 in json:
                        th1 = [fatherurl, i1["name"]]
                        for i2 in i1["sections"]:
                            th2 = copy.deepcopy(th1)
                            th2.extend([i2["name"]])
                            for i3 in i2["subsections"]:
                                th3 = copy.deepcopy(th2)
                                th3.extend([i3["name"]])
                                for i4 in i3["themes"]:
                                    th4 = copy.deepcopy(th3)
                                    th4.extend([i4["name"], url_head.format(i4["id"]), i4["id"]])
                                    listres.append(th4)
                    return listres

                reslist = json2list(i1, self.url_list["pre_video"], the_page0)
                self.write2db_bigvideo_class_format(reslist)
                logger.info("get_current_page: %s" % the_page0)
                # exit(0)

    def _get_seeds_page(self):
        # 知识点从这里面读取。
        self._readdb_bigvideo_class_format()
        # print(len(self._pre_video_urls))
        for i1 in self._pre_video_urls:
            req = librequest.Request(url=i1, method='GET', headers=self.headers)
            with librequest.urlopen(req) as response:
                ori_page = response.read().decode('utf-8')
                the_page0 = simplejson.loads(ori_page)

                # chapter_url, points, pointsid, seed_urls, urldir, had_get
                def json2list(fatherurl, json):
                    listres = []
                    for i1 in json["topics"]:
                        for i2 in i1["video"]["addresses"]:
                            listres.append(
                                [fatherurl, i1["name"], i1["id"], i2["url"], i2["url"].split("/")[-1].split(".")[0], 0])
                    return listres

                reslist = json2list(i1, the_page0)
                self.write2db_seed_all_format(reslist)
                logger.info("get_current_page: %s" % the_page0)

    def write2db_ori(self, keyname, strs):
        have_num_sql = """SELECT COUNT(1) as a FROM `json_buff` WHERE jsontype="{}";""".format(keyname)
        print(have_num_sql)
        have_num = self.db_conn.exec_sql(have_num_sql)
        print(have_num)
        if have_num[0]["a"] == 0:
            write_sql = """insert into `json_buff` (json,jsontype) VALUES ("{}","{}");""".format(strs, keyname)
            print(write_sql)
            write_res = self.db_conn.exec_sql(write_sql)
        else:
            write_sql = """UPDATE `json_buff` SET json="{}" WHERE jsontype="{}";""".format(strs, keyname)
            print(write_sql)
            write_res = self.db_conn.exec_sql(write_sql)
        logger.info("write2db_ori: %s" % write_res)

    def _readdb_main_format(self):
        have_num_sql = """SELECT subjectsid, stageid, publisherid, semesterid FROM `main_class`;"""
        # print(have_num_sql)
        have_res = self.db_conn.exec_sql(have_num_sql)
        # print(have_res)
        self._main_current_urls = []
        for i1 in have_res:
            tmp_current_url = self.url_list["current_chapters"].format(i1["publisherid"], i1["semesterid"],
                                                                       i1["subjectsid"], i1["stageid"])
            self._main_current_urls.append(tmp_current_url)

    def _readdb_bigvideo_class_format(self):
        # have_num_sql = """SELECT video_urlid FROM `big_video_class`;"""
        have_num_sql = """SELECT video_url FROM big_video_class WHERE video_url not in (SELECT chapter_url from seed_point_all GROUP BY chapter_url);"""
        # print(have_num_sql)
        have_res = self.db_conn.exec_sql(have_num_sql)
        # print(have_res)
        self._pre_video_urls = []
        for i1 in have_res:
            # tmp_pre_video_url = self.url_list["pre_video"].format(i1["video_urlid"])
            self._pre_video_urls.append(i1["video_url"])

    def _readdb_seed_all_format(self):
        file_name = os.path.join(data_path, "down", "instruction.csv")
        if 0:
            # if os.path.isfile(file_name):
            print("from pandas")
            pdobj = pd.read_csv(file_name)
            str_res = pdobj.to_json(orient='records', force_ascii=False)
            have_res = json.loads(str_res, encoding="utf-8")
        else:
            print("from sql")
            download_sql = """
            SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get from (
                (
                    SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM (
                    (SELECT subjects,publisher,stage,semester,urlconnect FROM `main_class` WHERE subjects='数学' and stage='初中') l1 
                    LEFT JOIN 
                    (SELECT father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM `big_video_class`) r1 
                    ON l1.urlconnect=r1.father_url
                    )
                ) l2
                LEFT JOIN 
                (SELECT chapter_url,seed_urls,urldir,points,had_get FROM `seed_point_all` WHERE seed_urls LIKE '%.m3u8' and urldir LIKE 'pcM_%' and had_get=0) r2 
                ON l2.video_url=r2.chapter_url
            ) WHERE had_get IS NOT NULL;
            """
            have_res = self.db_conn.exec_sql(download_sql)
            pdobj = pd.DataFrame(have_res)
            # pdobj.to_csv(file_name, encoding='utf-8', index=False)
            pdobj.to_excel(file_name, sheet_name='Sheet1', index=False)
        self._download_urls_and_info = have_res

    def write2db_main_format(self, lists):
        for i1 in lists:
            tmp_page_url = self.url_list["current_chapters"].format(i1[5], i1[7], i1[1], i1[3])
            have_num_sql = """SELECT COUNT(1) as a FROM `main_class` WHERE urlconnect="{}";""".format(tmp_page_url)
            # print(have_num_sql)
            have_num = self.db_conn.exec_sql(have_num_sql)
            # print(have_num)
            if have_num[0]["a"] == 0:
                write_sql = """insert into `main_class` (subjects,subjectsid,stage,stageid,publisher,publisherid,semester,semesterid,urlconnect) VALUES ("{}","{}","{}","{}","{}","{}","{}","{}","{}");""".format(
                    i1[0], i1[1], i1[2], i1[3], i1[4], i1[5], i1[6], i1[7], tmp_page_url)
                # print(write_sql)
                write_res = self.db_conn.exec_sql(write_sql)
            else:
                continue
                # write_sql = """UPDATE `main_class` SET subjectsid={},stageid={},publisherid={},semesterid={} WHERE urlconnect="{}";""".format(
                #     i1[1], i1[3], i1[5], i1[7], i1[0], i1[2], i1[4], i1[6], tmp_page_url)
                # # print(write_sql)
                # write_res = self.db_conn.exec_sql(write_sql)
            logger.info("write2db_main_format: %s" % write_res)

    def write2db_bigvideo_class_format(self, lists):
        # father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid
        for i1 in lists:
            have_num_sql = """SELECT COUNT(1) as a FROM `big_video_class` WHERE video_urlid="{}";""".format(i1[6])
            # print(have_num_sql)
            have_num = self.db_conn.exec_sql(have_num_sql)
            # print(have_num)
            if have_num[0]["a"] == 0:
                write_sql = """insert into `big_video_class` (father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid) VALUES ("{}","{}","{}","{}","{}","{}","{}");""".format(
                    i1[0], i1[1], i1[2], i1[3], i1[4], i1[5], i1[6])
                # print(write_sql)
                write_res = self.db_conn.exec_sql(write_sql)
            else:
                continue
                # write_sql = """UPDATE `big_video_class` SET subjectsid={},stageid={},publisherid={},semesterid={} WHERE subjects="{}" and stage="{}" and publisher="{}" and semester="{}";""".format(
                #     i1[1], i1[3], i1[5], i1[7], i1[0], i1[2], i1[4], i1[6])
                # # print(write_sql)
                # write_res = self.db_conn.exec_sql(write_sql)
            logger.info("write2db_bigvideo_class_format: %s" % write_res)

    def write2db_seed_all_format(self, lists):
        # chapter_url, points, pointsid, seed_urls, urldir, had_get
        for i1 in lists:
            have_num_sql = """SELECT COUNT(1) as a FROM `seed_point_all` WHERE chapter_url="{}" and urldir="{}";""".format(
                i1[0], i1[4])
            # print(have_num_sql)
            have_num = self.db_conn.exec_sql(have_num_sql)
            # print(have_num)
            if have_num[0]["a"] == 0:
                write_sql = """insert into `seed_point_all` (chapter_url, points, pointsid, seed_urls, urldir, had_get) VALUES ("{}","{}","{}","{}","{}","{}");""".format(
                    i1[0], i1[1], i1[2], i1[3], i1[4], i1[5])
                # print(write_sql)
                write_res = self.db_conn.exec_sql(write_sql)
            else:
                continue
                # write_sql = """UPDATE `big_video_class` SET subjectsid={},stageid={},publisherid={},semesterid={} WHERE subjects="{}" and stage="{}" and publisher="{}" and semester="{}";""".format(
                #     i1[1], i1[3], i1[5], i1[7], i1[0], i1[2], i1[4], i1[6])
                # # print(write_sql)
                # write_res = self.db_conn.exec_sql(write_sql)
            logger.info("write2db_bigvideo_class_format: %s" % write_res)

    def write2db_down_success(self, video_url, points, urldir):
        # chapter_url, points, pointsid, seed_urls, urldir, had_get
        have_num_sql = """UPDATE `seed_point_all` set had_get=1 WHERE chapter_url="{}" and points="{}" and urldir="{}";""".format(
            video_url, points, urldir)
        logger.info(have_num_sql)
        have_num = self.db_conn.exec_sql(have_num_sql)
        logger.info(have_num)

    def export_content(self):
        file_name = os.path.join(data_path, "down", "instruction.csv")
        download_sql = """
        SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get from (
            (
                SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM (
                (SELECT subjects,publisher,stage,semester,urlconnect FROM `main_class` WHERE subjects='数学' and stage='初中') l1 
                LEFT JOIN 
                (SELECT father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM `big_video_class`) r1 
                ON l1.urlconnect=r1.father_url
                )
            ) l2
            LEFT JOIN 
            (SELECT chapter_url,seed_urls,urldir,points,had_get FROM `seed_point_all` WHERE seed_urls LIKE '%.m3u8' and urldir LIKE 'pcM_%') r2 
            ON l2.video_url=r2.chapter_url
        ) WHERE had_get IS NOT NULL;
        """
        have_res = self.db_conn.exec_sql(download_sql)
        pdobj = pd.DataFrame(have_res)
        pdobj.to_csv(file_name, encoding='utf-8', index=False)


class Yangcong_append():
    def __init__(self, mysql_ins):
        self.db_conn = mysql_ins
        self._aheads = {}
        self.url_list = {
            "log_url": "https://api-v5-0.yangcong345.com/login",
            "main_class": "https://school-api.yangcong345.com/course/subjects",
            # "user_current": "https://api-v5-0.yangcong345.com/user-current-textbook",  # 临时忽略
            # 包含 pre_video 的key
            # 视频网页的显示要点
            "current_chapters": "https://school-api.yangcong345.com/course/chapters-with-section/publisher/{}/semester/{}/subject/{}/stage/{}",
            # # 视屏里的穿插话题。
            # "theme": "https://school-api.yangcong345.com/course/problems/fc96aa6c-6c28-11e7-92b1-7f399bce3897/1",
            # # 视频网页的纯key映射不含要点，可以不管
            # "video_summary": "https://api-v5-0.yangcong345.com/progresses?subjectId=1&publisherId=1&semesterId=13&stageId=2",
            # # # 参数
            # # subjectId: 1
            # # publisherId: 1
            # # semesterId: 13
            # # stageId: 2
            # # 返回 课题类型 概念题 id
            # 返回gene_seed的url地址。
            # "pre_video": "https://school-api.yangcong345.com/course/course-tree/themes/0b315a90-57f7-11e7-aca6-c3badc9742d1",
            "pre_video": "https://school-api.yangcong345.com/course/course-tree/themes/{}",
            "seed_list": [],
            "gene_seed": "https://hls.media.yangcong345.com/pcM/pcM_58c26cbb36eaf35866aae116.m3u8",
        }
        self._iplist = [
            "",
            "",
        ]
        self._agentlist = [
            "Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Safari/535.19",
            "Mozilla/5.0 (Linux; U; Android 4.0.4; en-gb; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30",
            "Mozilla/5.0 (Linux; U; Android 2.2; en-gb; GT-P1000 Build/FROYO) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
            "Mozilla/5.0 (Android; Mobile; rv:14.0) Gecko/14.0 Firefox/14.0",
            "Mozilla/5.0 (Windows NT 6.2; WOW64; rv:21.0) Gecko/20100101 Firefox/21.0",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36",
            "Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19",
            "Mozilla/5.0 (iPod; U; CPU like Mac OS X; en) AppleWebKit/420.1 (KHTML, like Gecko) Version/3.0 Mobile/3A101a Safari/419.3",
            "Mozilla/5.0 (iPad; CPU OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A334 Safari/7534.48.3",
        ]

    # 获取视频基本信息
    def get_basic_info2db(self):
        # 当前主列表集合
        self._main_current_urls = []
        # 当前页面视频前页集合
        self._pre_video_urls = []
        self._login()
        self._get_main_list()
        self._get_current_page()
        self._get_seeds_page()

    # 获取视频基本信息
    def download_normal_usedb(self):
        # 1. 读取下载地址
        self._download_urls_and_info = []
        self._readdb_seed_all_format()
        leth = len(self._download_urls_and_info)
        for id1, i1 in enumerate(self._download_urls_and_info):
            print(id1, leth, i1)
            self._download_ke(i1)

    def _download_ke(self, info_array):
        # subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get
        head_urls = "https://hls.media.yangcong345.com/pcL/"
        # link_demo = "https://hls.media.yangcong345.com/pcM/pcM_58c26cbb36eaf35866aae1161.ts"
        # 1. 如果不存在则创建目录
        # video_path = os.path.join(data_path, "down", info_array["video_urlid"],
        #                           copy.deepcopy(info_array["points"]).replace("|", "（竖线）").strip(" "),
        #                           info_array["urldir"])
        video_path = os.path.join(data_path, "down_math_verse", info_array["urldir"])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        request_list = []
        f_list = []
        # 2. 下载种子
        logger.info("downloading: {}".format(info_array["seed_urls"]))
        response = librequest.urlopen(info_array["seed_urls"])
        cat_vido = response.read()
        seed_name = info_array["urldir"] + ".m3u8"
        # 2.1 种子保存
        with open(os.path.join(video_path, seed_name), 'wb') as f:
            f.write(cat_vido)
        # 2.2 种子读取
        with open(os.path.join(video_path, seed_name), "r") as f:
            contentlists = f.readlines()
        # 2.3 种子解析
        for i1 in contentlists:
            if not i1.startswith("#") and i1.strip() != "":
                f_list.append(i1.strip())
                request_list.append(head_urls + i1.strip())
        # 3. 下载
        try:
            for i1 in zip(request_list, f_list):
                response = librequest.urlopen(i1[0], timeout=60)
                cat_vido = response.read()
                # 5. 写入本地文件
                with open(os.path.join(video_path, i1[1]), 'wb') as f:
                    f.write(cat_vido)
            # 5. 数据库状态写入
            self.write2db_down_success(info_array["video_url"], info_array["seed_urls"], info_array["urldir"])
        except Exception as e:
            logger.info(e)

    def _login(self):
        post_data = dict(
            name="18721986267",
            password="fff111QQQ",
        )
        endata = bytes(json.dumps(post_data), "utf-8")
        request_headers = {"content-type": "application/json"}
        req = librequest.Request(url=self.url_list["log_url"], data=endata, method='POST', headers=request_headers)
        with librequest.urlopen(req) as response:
            ori_page = response.read().decode('utf-8')
            self._aheads["Authorization"] = response.headers["Authorization"]
            the_page0 = simplejson.loads(ori_page)
            logger.info("login json: %s" % the_page0)
            self.headers = {
                "Accept": "application/json",
                "Authorization": self._aheads["Authorization"],
                "client-type": "pc",
                "client-version": "6.7.4",
                "device": 2630311005440102,
                "Referer": "https://yangcong345.com/",
                "Sec-Fetch-Mode": "cors",
                "User-Agent": "Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1",
            }

    def _get_main_list(self):
        req = librequest.Request(url=self.url_list["main_class"], method='GET', headers=self.headers)
        with librequest.urlopen(req) as response:
            ori_page = response.read().decode('utf-8')
            the_page0 = simplejson.loads(ori_page)

            # subjects,subjectsid,stage,stageid,publisher,publisherid,semester,semesterid
            def json2list(json):
                listres = []
                for i1 in json:
                    th1 = [i1["name"], i1["id"]]
                    for i2 in i1["stages"]:
                        th2 = copy.deepcopy(th1)
                        th2.extend([i2["name"], i2["id"]])
                        for i3 in i2["publishers"]:
                            th3 = copy.deepcopy(th2)
                            th3.extend([i3["name"], i3["id"]])
                            for i4 in i3["semesters"]:
                                th4 = copy.deepcopy(th3)
                                th4.extend([i4["name"], i4["id"]])
                                listres.append(th4)
                return listres

            reslist = json2list(the_page0)
            self.write2db_main_format(reslist)
            logger.info("main class json: %s" % the_page0)

    def _get_current_page(self):
        self._readdb_main_format()
        for i1 in self._main_current_urls:
            req = librequest.Request(url=i1, method='GET', headers=self.headers)
            with librequest.urlopen(req) as response:
                ori_page = response.read().decode('utf-8')
                the_page0 = simplejson.loads(ori_page)

                # father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid
                def json2list(fatherurl, url_head, json):
                    listres = []
                    for i1 in json:
                        th1 = [fatherurl, i1["name"]]
                        for i2 in i1["sections"]:
                            th2 = copy.deepcopy(th1)
                            th2.extend([i2["name"]])
                            for i3 in i2["subsections"]:
                                th3 = copy.deepcopy(th2)
                                th3.extend([i3["name"]])
                                for i4 in i3["themes"]:
                                    th4 = copy.deepcopy(th3)
                                    th4.extend([i4["name"], url_head.format(i4["id"]), i4["id"]])
                                    listres.append(th4)
                    return listres

                reslist = json2list(i1, self.url_list["pre_video"], the_page0)
                self.write2db_bigvideo_class_format(reslist)
                logger.info("get_current_page: %s" % the_page0)
                # exit(0)

    def _get_seeds_page(self):
        # 知识点从这里面读取。
        self._readdb_bigvideo_class_format()
        # print(len(self._pre_video_urls))
        for i1 in self._pre_video_urls:
            req = librequest.Request(url=i1, method='GET', headers=self.headers)
            with librequest.urlopen(req) as response:
                ori_page = response.read().decode('utf-8')
                the_page0 = simplejson.loads(ori_page)

                # chapter_url, points, pointsid, seed_urls, urldir, had_get
                def json2list(fatherurl, json):
                    listres = []
                    for i1 in json["topics"]:
                        for i2 in i1["video"]["addresses"]:
                            listres.append(
                                [fatherurl, i1["name"], i1["id"], i2["url"], i2["url"].split("/")[-1].split(".")[0], 0])
                    return listres

                reslist = json2list(i1, the_page0)
                self.write2db_seed_all_format(reslist)
                logger.info("get_current_page: %s" % the_page0)

    def write2db_ori(self, keyname, strs):
        have_num_sql = """SELECT COUNT(1) as a FROM `json_buff` WHERE jsontype="{}";""".format(keyname)
        print(have_num_sql)
        have_num = self.db_conn.exec_sql(have_num_sql)
        print(have_num)
        if have_num[0]["a"] == 0:
            write_sql = """insert into `json_buff` (json,jsontype) VALUES ("{}","{}");""".format(strs, keyname)
            print(write_sql)
            write_res = self.db_conn.exec_sql(write_sql)
        else:
            write_sql = """UPDATE `json_buff` SET json="{}" WHERE jsontype="{}";""".format(strs, keyname)
            print(write_sql)
            write_res = self.db_conn.exec_sql(write_sql)
        logger.info("write2db_ori: %s" % write_res)

    def _readdb_main_format(self):
        have_num_sql = """SELECT subjectsid, stageid, publisherid, semesterid FROM `main_class`;"""
        # print(have_num_sql)
        have_res = self.db_conn.exec_sql(have_num_sql)
        # print(have_res)
        self._main_current_urls = []
        for i1 in have_res:
            tmp_current_url = self.url_list["current_chapters"].format(i1["publisherid"], i1["semesterid"],
                                                                       i1["subjectsid"], i1["stageid"])
            self._main_current_urls.append(tmp_current_url)

    def _readdb_bigvideo_class_format(self):
        # have_num_sql = """SELECT video_urlid FROM `big_video_class`;"""
        have_num_sql = """SELECT video_url FROM big_video_class WHERE video_url not in (SELECT chapter_url from seed_point_all GROUP BY chapter_url);"""
        # print(have_num_sql)
        have_res = self.db_conn.exec_sql(have_num_sql)
        # print(have_res)
        self._pre_video_urls = []
        for i1 in have_res:
            # tmp_pre_video_url = self.url_list["pre_video"].format(i1["video_urlid"])
            self._pre_video_urls.append(i1["video_url"])

    def _readdb_seed_all_format(self):
        # file_name = os.path.join(data_path, "down", "instruction.csv")
        file_name = os.path.join(data_path, "instruction.xls")
        if 0:
            # if os.path.isfile(file_name):
            print("from pandas")
            pdobj = pd.read_csv(file_name)
            str_res = pdobj.to_json(orient='records', force_ascii=False)
            have_res = json.loads(str_res, encoding="utf-8")
        else:
            print("from sql")
            download_sql = """
            SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get from (
                (
                    SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM (
                    (SELECT subjects,publisher,stage,semester,urlconnect FROM `main_class` WHERE subjects='数学' and stage<>'初中') l1 
                    LEFT JOIN 
                    (SELECT father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM `big_video_class`) r1 
                    ON l1.urlconnect=r1.father_url
                    )
                ) l2
                RIGHT JOIN 
                (SELECT chapter_url,seed_urls,urldir,points,had_get FROM `seed_point_all` WHERE seed_urls LIKE '%.m3u8' and urldir LIKE 'pcL_%' and had_get=0 GROUP BY seed_urls) r2 
                ON l2.video_url=r2.chapter_url
            ) WHERE had_get IS NOT NULL AND video_url IS NOT NULL;
            """
            have_res = self.db_conn.exec_sql(download_sql)
            pdobj = pd.DataFrame(have_res)
            # pdobj.to_csv(file_name, encoding='utf-8', index=False)
            pdobj.to_excel(file_name, sheet_name='Sheet1', index=False)
        self._download_urls_and_info = have_res

    def write2db_main_format(self, lists):
        for i1 in lists:
            tmp_page_url = self.url_list["current_chapters"].format(i1[5], i1[7], i1[1], i1[3])
            have_num_sql = """SELECT COUNT(1) as a FROM `main_class` WHERE urlconnect="{}";""".format(tmp_page_url)
            # print(have_num_sql)
            have_num = self.db_conn.exec_sql(have_num_sql)
            # print(have_num)
            if have_num[0]["a"] == 0:
                write_sql = """insert into `main_class` (subjects,subjectsid,stage,stageid,publisher,publisherid,semester,semesterid,urlconnect) VALUES ("{}","{}","{}","{}","{}","{}","{}","{}","{}");""".format(
                    i1[0], i1[1], i1[2], i1[3], i1[4], i1[5], i1[6], i1[7], tmp_page_url)
                # print(write_sql)
                write_res = self.db_conn.exec_sql(write_sql)
            else:
                continue
                # write_sql = """UPDATE `main_class` SET subjectsid={},stageid={},publisherid={},semesterid={} WHERE urlconnect="{}";""".format(
                #     i1[1], i1[3], i1[5], i1[7], i1[0], i1[2], i1[4], i1[6], tmp_page_url)
                # # print(write_sql)
                # write_res = self.db_conn.exec_sql(write_sql)
            logger.info("write2db_main_format: %s" % write_res)

    def write2db_bigvideo_class_format(self, lists):
        # father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid
        for i1 in lists:
            have_num_sql = """SELECT COUNT(1) as a FROM `big_video_class` WHERE video_urlid="{}";""".format(i1[6])
            # print(have_num_sql)
            have_num = self.db_conn.exec_sql(have_num_sql)
            # print(have_num)
            if have_num[0]["a"] == 0:
                write_sql = """insert into `big_video_class` (father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid) VALUES ("{}","{}","{}","{}","{}","{}","{}");""".format(
                    i1[0], i1[1], i1[2], i1[3], i1[4], i1[5], i1[6])
                # print(write_sql)
                write_res = self.db_conn.exec_sql(write_sql)
            else:
                continue
                # write_sql = """UPDATE `big_video_class` SET subjectsid={},stageid={},publisherid={},semesterid={} WHERE subjects="{}" and stage="{}" and publisher="{}" and semester="{}";""".format(
                #     i1[1], i1[3], i1[5], i1[7], i1[0], i1[2], i1[4], i1[6])
                # # print(write_sql)
                # write_res = self.db_conn.exec_sql(write_sql)
            logger.info("write2db_bigvideo_class_format: %s" % write_res)

    def write2db_seed_all_format(self, lists):
        # chapter_url, points, pointsid, seed_urls, urldir, had_get
        for i1 in lists:
            have_num_sql = """SELECT COUNT(1) as a FROM `seed_point_all` WHERE chapter_url="{}" and urldir="{}";""".format(
                i1[0], i1[4])
            # print(have_num_sql)
            have_num = self.db_conn.exec_sql(have_num_sql)
            # print(have_num)
            if have_num[0]["a"] == 0:
                write_sql = """insert into `seed_point_all` (chapter_url, points, pointsid, seed_urls, urldir, had_get) VALUES ("{}","{}","{}","{}","{}","{}");""".format(
                    i1[0], i1[1], i1[2], i1[3], i1[4], i1[5])
                # print(write_sql)
                write_res = self.db_conn.exec_sql(write_sql)
            else:
                continue
                # write_sql = """UPDATE `big_video_class` SET subjectsid={},stageid={},publisherid={},semesterid={} WHERE subjects="{}" and stage="{}" and publisher="{}" and semester="{}";""".format(
                #     i1[1], i1[3], i1[5], i1[7], i1[0], i1[2], i1[4], i1[6])
                # # print(write_sql)
                # write_res = self.db_conn.exec_sql(write_sql)
            logger.info("write2db_bigvideo_class_format: %s" % write_res)

    def write2db_down_success(self, video_url, seed_urls, urldir):
        # chapter_url, points, pointsid, seed_urls, urldir, had_get
        have_num_sql = """UPDATE `seed_point_all` set had_get=1 WHERE chapter_url="{}" and seed_urls="{}" and urldir="{}";""".format(
            video_url, seed_urls, urldir)
        logger.info(have_num_sql)
        have_num = self.db_conn.exec_sql(have_num_sql)
        logger.info(have_num)

    def export_content(self):
        # file_name = os.path.join(data_path, "down", "instruction.xls")
        subject_name = "数学"
        section_name = "高中"
        file_name = os.path.join("instruction{}{}.xls".format(section_name, subject_name))
        # download_sql = """
        # SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get from (
        #     (
        #         SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM (
        #         (SELECT subjects,publisher,stage,semester,urlconnect FROM `main_class` WHERE subjects='{}' and stage='{}') l1
        #         LEFT JOIN
        #         (SELECT father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM `big_video_class`) r1
        #         ON l1.urlconnect=r1.father_url
        #         )
        #     ) l2
        #     LEFT JOIN
        #     (SELECT chapter_url,seed_urls,urldir,points,had_get FROM `seed_point_all` WHERE seed_urls LIKE '%.m3u8' and urldir LIKE 'pcM_%') r2
        #     ON l2.video_url=r2.chapter_url
        # ) WHERE had_get is NOT NULL ;
        # """
        download_sql = """
        SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid,seed_urls,urldir,points,had_get from (
            (
                SELECT subjects,publisher,stage,semester,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM (
                (SELECT subjects,publisher,stage,semester,urlconnect FROM `main_class` WHERE subjects='{}' and stage='{}') l1 
                LEFT JOIN 
                (SELECT father_url,chapter_1,chapter_2,chapter_3,lesson_type,video_url,video_urlid FROM `big_video_class`) r1 
                ON l1.urlconnect=r1.father_url
                )
            ) l2
            RIGHT JOIN 
            (SELECT chapter_url,seed_urls,urldir,points,had_get FROM `seed_point_all` WHERE seed_urls LIKE '%.m3u8' and urldir LIKE 'pcM_%' GROUP BY seed_urls ) r2 
            ON l2.video_url=r2.chapter_url
        ) WHERE had_get is NOT NULL and chapter_1 is NOT NULL ;
        """.format(subject_name, section_name)
        print(download_sql)
        have_res = self.db_conn.exec_sql(download_sql)
        pdobj = pd.DataFrame(have_res)
        # pdobj.to_csv(file_name, encoding='utf-8', index=False)
        pdobj.to_excel(file_name, sheet_name='Sheet1', index=False)


def main():
    config = {
        'host': "127.0.0.1",
        'user': "root",
        'password': "333",
        'database': "ycdb",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
        'cursorclass': pymysql.cursors.DictCursor
    }
    mysql_ins = MysqlDB(config)
    yangcong_ins = Yangcong_append(mysql_ins)
    # 获取视频基本信息
    # yangcong_ins.get_basic_info2db()
    # # 下载视频基本信息
    # yangcong_ins.download_normal_usedb()
    # 导出视频目录
    yangcong_ins.export_content()


def test_proxy():
    # 访问网址
    url = 'https://www.whatismyip.com/my-ip-information/?iref=home'
    # url = 'https://www.baidu.com'
    # 这是代理IP
    proxy = {'https': '113.247.252.114:9090'}
    # 创建ProxyHandler
    proxy_support = request.ProxyHandler(proxy)
    # 创建Opener
    opener = request.build_opener(proxy_support)
    # 添加User Angent
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36')]
    # 安装OPener
    request.install_opener(opener)
    # 使用自己安装好的Opener
    response = request.urlopen(url)
    # 读取相应信息并解码
    html = response.read().decode("utf-8")
    # 打印信息
    print(html)


if __name__ == '__main__':
    logger.info("".center(100, "*"))
    logger.info("welcome to claw".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    logger.info("")
    main()
    # test_proxy()
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
