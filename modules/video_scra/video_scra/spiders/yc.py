# -*- coding: utf-8 -*-
import scrapy
import re
from video_scra.log_tool import logger


class YcSpider(scrapy.Spider):
    name = 'yc'
    allowed_domains = ['yangcong345.com']
    # start_urls = ['http://yangcong345.com/']
    # start_urls = ["https://yangcong345.com/#/login?type=login"]
    start_urls = ["https://api-v5-0.yangcong345.com/login"]

    # start_urls = ["https://yangcong345.com/#/teaching/course"]

    def start_requests(self):
        cookies = "anonymid=jcokuqturos8ql; depovince=GW; jebecookies=f90c9e96-78d7-4f74-b1c8-b6448492995b|||||; _r01_=1; JSESSIONID=abcx4tkKLbB1-hVwvcyew; ick_login=ff436c18-ec61-4d65-8c56-a7962af397f4; _de=BF09EE3A28DED52E6B65F6A4705D973F1383380866D39FF5; p=90dea4bfc79ef80402417810c0de60989; first_login_flag=1; ln_uact=mr_mao_hacker@163.com; ln_hurl=http://hdn.xnimg.cn/photos/hdn421/20171230/1635/main_JQzq_ae7b0000a8791986.jpg; t=24ee96e2e2301bf2c350d7102956540a9; societyguester=24ee96e2e2301bf2c350d7102956540a9; id=327550029; xnsid=e7f66e0b; loginfrom=syshome; ch_id=10016"
        cookies = {i.split("=")[0]: i.split("=")[1] for i in cookies.split("; ")}
        # headers = {"Cookie":cookies}
        yield scrapy.Request(
            self.start_urls[0],
            callback=self.parse,
            cookies=cookies
            # headers = headers
        )

    def parse(self, response):
        print("")
        print("*" * 100)
        print(response.body.decode())
        print("*" * 100)
        print("")
        # password = response.xpath("//input[@id='password']/@value").extract_first()
        username = response.xpath("//input[@id='username']/@value").extract_first()
        password = response.xpath("//input[@id='password']/@value").extract_first()
        # commit = response.xpath("//input[@name='commit']/@value").extract_first()
        post_data = dict(
            username="dfdf",
            password="fff111QQQ",
            # authenticity_token=authenticity_token,
            # utf8=utf8,
            # commit=commit
        )
        yield scrapy.FormRequest(
            "https://github.com/session",
            formdata=post_data,
            callback=self.after_login
        )

        # def parse(self, response):
        #     username = response.xpath("//input[@id='username']/@value").extract_first()
        #     password = response.xpath("//input[@id='password']/@value").extract_first()
        #     # commit = response.xpath("//input[@name='commit']/@value").extract_first()
        #     post_data = dict(
        #         username="dfdf",
        #         password="fff111QQQ",
        #         # authenticity_token=authenticity_token,
        #         # utf8=utf8,
        #         # commit=commit
        #     )
        #     yield scrapy.FormRequest(
        #         "https://github.com/session",
        #         formdata=post_data,
        #         callback=self.after_login
        #     )

    def after_login(self, response):
        # with open("a.html","w",encoding="utf-8") as f:
        #     f.write(response.body.decode())
        print(re.findall("物理|数学", response.body.decode()))
