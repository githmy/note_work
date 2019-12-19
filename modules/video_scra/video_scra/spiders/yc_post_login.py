# -*- coding: utf-8 -*-
import scrapy
# from scrapy.http.headers import Headers
import re


class YcPostLoginSpider(scrapy.Spider):
    name = 'yc_post_login'
    allowed_domains = ['yangcong345.com']
    start_urls = ["https://yangcong345.com/#/login?type=login"]

    def __init__(self, *a, **kw):
        super(YcPostLoginSpider, self).__init__(*a, **kw)
        self._aheads = {}
        self.url_list = {
            "log_url": "https://api-v5-0.yangcong345.com/login",
            "main_class": "https://school-api.yangcong345.com/course/subjects",
            # "user_current": "https://api-v5-0.yangcong345.com/user-current-textbook",  # 临时忽略
            # 包含 pre_video 的key
            # 视频网页的显示要点
            "current_chapters": "https://school-api.yangcong345.com/course/chapters-with-section/publisher/1/semester/13/subject/1/stage/2",
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
            "pre_video": "https://school-api.yangcong345.com/course/course-tree/themes/0b315a90-57f7-11e7-aca6-c3badc9742d1",
            "gene_seed": "https://hls.media.yangcong345.com/pcM/pcM_58c26cbb36eaf35866aae116.m3u8",
        }

    def parse(self, response):
        post_data = dict(
            name="18721986267",
            password="fff111QQQ",
        )
        yield scrapy.FormRequest(
            self.url_list["log_url"],
            formdata=post_data,
            callback=self.main_class
        )

    def main_class(self, response):
        # 获取大类
        # request.headers.setdefault('User-Agent', get_ua())
        self._aheads["Authorization"] = response.headers["Authorization"].decode()
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
        print("main_class".center(30, " ").center(100, "*"))
        print(response.body.decode())
        print("-" * 100)
        print(response.meta)
        # print(response.headers)
        # print(response.headers["Authorization"].decode())
        # main_json = response.headers["Authorization"].decode()
        # print(type(response.url))
        # print(type(response.status))
        return scrapy.Request(
            self.url_list["main_class"],
            # self.url_list["pre_video"],
            headers=self.headers,
            # callback=self.user_current,
            callback=self.current_chapters,
            meta={"main_class_json": response.body}
        )

    def user_current(self, response):
        # 过滤客户的可用内容
        print("filter_custorm_content".center(30, " ").center(100, "*"))
        print(response.body.decode())
        print("-" * 100)
        print(response.meta)
        return scrapy.Request(
            self.url_list["user_current"],
            headers=self.headers,
            callback=self.current_chapters,
            meta={"user_current_json": response.body}
        )

    def current_chapters(self, response):
        # 当前章节
        print("current_chapters".center(30, " ").center(100, "*"))
        print(response.body.decode())
        print("-" * 100)
        print(response.meta)
        return scrapy.Request(
            self.url_list["current_chapters"],
            headers=self.headers,
            callback=self.theme,
            meta={"current_chapters_json": response.body}
        )

    def theme(self, response):
        # 视频概览
        print("theme".center(30, " ").center(100, "*"))
        print(response.body.decode())
        print("-" * 100)
        print(response.meta)
        return scrapy.Request(
            self.url_list["theme"],
            headers=self.headers,
            callback=self.video_summary,
            meta={"theme_json": response.body}
        )

    def video_summary(self, response):
        # 视频概览
        print("video_summary".center(30, " ").center(100, "*"))
        print(response.body.decode())
        print("-" * 100)
        print(response.meta)
        return scrapy.Request(
            self.url_list["video_summary"],
            headers=self.headers,
            callback=self.pre_video,
            meta={"video_summary_json": response.body}
        )

    def pre_video(self, response):
        # 视频种子
        print("pre_video".center(30, " ").center(100, "*"))
        print(response.body.decode())
        print("-" * 100)
        print(response.meta)
        return scrapy.Request(
            self.url_list["pre_video"],
            headers=self.headers,
            callback=self.final4pipeline,
            meta={"pre_video_json": response.body}
        )

    def final4pipeline(self, response):
        # 返还所有结果，item导入final4pipeline
        print("final4pipeline".center(30, " ").center(100, "*"))
        print(response)
        print(response.body.decode())
        print("-" * 100)
        print(response.meta)
        # 内部是每一个频频的不同清晰度地址
        # 单视频前页地址列表
        for i1 in response.meta["pre_video_json"]["topics"]:
            addreslist = [i2["url"] for i2 in i1["video"]["addresses"]]
            # 每个种子用","分割,默认的以 "https://hls.media.yangcong345.com/pcM/pcM_"开头，".m3u8"结尾。

        """
        response.meta = {
            'main_json': "。。。",
            'item': "。。。",
            'depth': 2,
            'download_timeout': 180.0,
            'download_slot': 'school-api.yangcong345.com',
            'download_latency': 0.07305908203125
        }
        """
        print("-" * 100)
        print(response.meta["main_json"].decode())
        print(response.headers)
        print("*" * 100)
        print("")
