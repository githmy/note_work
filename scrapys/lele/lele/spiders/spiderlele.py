# -*- coding: utf-8 -*-
import scrapy
import urllib.request
import json
from bs4 import BeautifulSoup
from lxml import etree
from copy import deepcopy
import re


class SpiderleleSpider(scrapy.Spider):
    name = 'spiderlele'
    allowed_domains = ['www.leleketang.com']
    start_urls = ['http://www.leleketang.com/cr/categories.php']

    # 重载start_requests方法
    def start_requests(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:59.0) Gecko/20100101 Firefox/59.0"}
        # 指定cookies
        self.cookies = {
            'gs_browser': 'pc',
            'gs_browse_mode': 'desktop',
            # encode --> O:4:"User":33:{s:2:"id";i:1355491920;s:6:"fromId";s:5:"anony";s:5:"snsId";i:0;s:8:"nickname";s:16:"游客1355491920";s:6:"gender";i:0;s:6:"avatar";s:23:"/login/image/anony2.png";s:5:"space";s:0:"";s:5:"email";s:0:"";s:6:"mobile";s:0:"";s:8:"realName";s:0:"";s:8:"location";s:0:"";s:5:"brief";s:0:"";s:9:"signature";s:0:"";s:10:"createTime";s:0:"";s:4:"fans";i:0;s:7:"follows";i:0;s:8:"articles";i:0;s:4:"role";i:0;s:8:"roleName";s:0:"";s:6:"skinId";i:0;s:8:"birthday";s:0:"";s:10:"locationId";i:0;s:9:"famillyId";i:0;s:6:"visits";i:0;s:13:"lastVisitTime";N;s:4:"days";i:0;s:8:"lastDays";i:0;s:5:"coins";i:0;s:4:"gems";i:0;s:11:"experiences";i:0;s:19:"current_experiences";i:0;s:5:"level";i:0;s:6:"awards";i:0;}
            'gs_anony': 'O%3A4%3A%22User%22%3A33%3A%7Bs%3A2%3A%22id%22%3Bi%3A1355491920%3Bs%3A6%3A%22fromId%22%3Bs%3A5%3A%22anony%22%3Bs%3A5%3A%22snsId%22%3Bi%3A0%3Bs%3A8%3A%22nickname%22%3Bs%3A16%3A%22%E6%B8%B8%E5%AE%A21355491920%22%3Bs%3A6%3A%22gender%22%3Bi%3A0%3Bs%3A6%3A%22avatar%22%3Bs%3A23%3A%22%2Flogin%2Fimage%2Fanony2.png%22%3Bs%3A5%3A%22space%22%3Bs%3A0%3A%22%22%3Bs%3A5%3A%22email%22%3Bs%3A0%3A%22%22%3Bs%3A6%3A%22mobile%22%3Bs%3A0%3A%22%22%3Bs%3A8%3A%22realName%22%3Bs%3A0%3A%22%22%3Bs%3A8%3A%22location%22%3Bs%3A0%3A%22%22%3Bs%3A5%3A%22brief%22%3Bs%3A0%3A%22%22%3Bs%3A9%3A%22signature%22%3Bs%3A0%3A%22%22%3Bs%3A10%3A%22createTime%22%3Bs%3A0%3A%22%22%3Bs%3A4%3A%22fans%22%3Bi%3A0%3Bs%3A7%3A%22follows%22%3Bi%3A0%3Bs%3A8%3A%22articles%22%3Bi%3A0%3Bs%3A4%3A%22role%22%3Bi%3A0%3Bs%3A8%3A%22roleName%22%3Bs%3A0%3A%22%22%3Bs%3A6%3A%22skinId%22%3Bi%3A0%3Bs%3A8%3A%22birthday%22%3Bs%3A0%3A%22%22%3Bs%3A10%3A%22locationId%22%3Bi%3A0%3Bs%3A9%3A%22famillyId%22%3Bi%3A0%3Bs%3A6%3A%22visits%22%3Bi%3A0%3Bs%3A13%3A%22lastVisitTime%22%3BN%3Bs%3A4%3A%22days%22%3Bi%3A0%3Bs%3A8%3A%22lastDays%22%3Bi%3A0%3Bs%3A5%3A%22coins%22%3Bi%3A0%3Bs%3A4%3A%22gems%22%3Bi%3A0%3Bs%3A11%3A%22experiences%22%3Bi%3A0%3Bs%3A19%3A%22current_experiences%22%3Bi%3A0%3Bs%3A5%3A%22level%22%3Bi%3A0%3Bs%3A6%3A%22awards%22%3Bi%3A0%3B%7D',
            'grade_id': 20,
            'course_id': 2,
        }
        # 再次请求到详情页，并且声明回调函数callback，dont_filter=True 不进行域名过滤，meta给回调函数传递数据
        for url in self.start_urls:
            yield scrapy.Request(url, headers=self.headers, cookies=self.cookies, callback=self.parse,
                                 meta={'myItem': ""}, dont_filter=True)

    def parse(self, response):
        # print("parse".center(30, " ").center(100, "*"))
        # 1. 加入cookie
        newcookie = self._2cookie(response, self.cookies)
        # 2. 规范返回对象
        soup = BeautifulSoup(response.body.decode(), "html.parser")
        html = etree.HTML(soup.prettify())
        a_list = html.xpath("//a[contains(@class,'kc_item')]")
        # div_list = response.xpath("//div[contains(@class,'categories')]")
        for a in a_list:
            item = {}
            item["a_href_1"] = a.xpath("./@href")[0]
            item["a_item_1"] = a.xpath("./@title")[0]
            # print(item)
            # print(response.urljoin(item["a_href_1"]))
            # if item["s_href"] is not None:
            yield scrapy.Request(
                response.urljoin(item["a_href_1"]),
                cookies=newcookie,
                callback=self.parse_url1_list,
                meta={"item_1": deepcopy(item), "bcookie": deepcopy(newcookie)}
            )

            # print(scrapy.Request.headers)
            # print(scrapy.Request.cookies)

            # post_data = dict(
            #     name="18721986267",
            #     password="fff111QQQ",
            # )
            # yield scrapy.FormRequest(
            #     self.url_list["log_url"],
            #     formdata=post_data,
            #     callback=self.main_class
            # )

    def parse_url1_list(self, response):
        # print("parse_url1_list".center(30, " ").center(100, "*"))
        # 1. 加入cookie
        oldcookie = response.meta["bcookie"]
        newcookie = self._2cookie(response, oldcookie)
        # 2. 规范返回对象
        soup = BeautifulSoup(response.body.decode(), "html.parser")
        html = etree.HTML(soup.prettify())
        a_list = html.xpath("//a[contains(@class,'kn_one')]")
        for a in a_list:
            item = deepcopy(response.meta["item_1"])
            tmph = a.xpath("./@href")
            item["a_href_2"] = tmph[0] if len(tmph) > 0 else None
            tmpt = a.xpath("./@title")
            if len(tmpt) > 0:
                item["a_item_2"] = tmpt[0]
            else:
                ttp = a.xpath("./div[contains(@class,'kn_o_name')]/text()")
                if len(ttp) > 0:
                    item["a_item_2"] = ttp[0].strip()
                else:
                    item["a_item_2"] = None
            if not item["a_href_2"].endswith(":;"):
                yield scrapy.Request(
                    response.urljoin(item["a_href_2"]),
                    cookies=newcookie,
                    callback=self.parse_url2_list,
                    meta={"item_2": deepcopy(item), "bcookie": deepcopy(newcookie)}
                )

    def parse_url2_list(self, response):
        exit(0)
        # print("parse_url2_list".center(30, " ").center(100, "*"))
        # # 1. 加入cookie
        # oldcookie = deepcopy(response.meta["bcookie"])
        # newcookie = self._2cookie(response, oldcookie)
        # 2. 规范返回对象
        # print(response.headers)
        # print(response.body.decode())
        item = response.meta["item_2"]
        tmpu = re.findall(r'ogv: "(//.*?\.ogv)",', response.body.decode())
        if len(tmpu) > 0:
            if len(tmpu) > 1:
                print("地址超量错误！")
                exit(0)
            item["url"] = response.urljoin(tmpu[0])
        return item

    def _2cookie(self, response, oldcookie):
        newcookie = deepcopy(oldcookie)
        # 1. 取出头部返回字典
        bb = response.headers.items()
        cookitems = [i1[1] for i1 in bb if i1[0].decode() == "Set-Cookie"]
        if len(cookitems) > 0:
            cookitem = cookitems[0]
            # 2. 加入cookie
            for i1 in cookitem:
                for i2 in i1.decode().split(";"):
                    i2 = i2.strip()
                    if i2.startswith("grade_id") or i2.startswith("course_id"):
                        kvlist = i2.split("=")
                        newcookie[kvlist[0]] = kvlist[1]
        return newcookie
