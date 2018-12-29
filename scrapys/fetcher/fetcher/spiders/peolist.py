# -*- coding: utf-8 -*-
import re
import scrapy
import codecs
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy.selector import Selector
from scrapy.conf import settings
import urllib
from scrapy.http.cookies import CookieJar  # 该模块继承自内置的http.cookiejar,操作类似
# from utils.trans_cookie import Cookieutil

# 实例化一个cookiejar对象
cookie_jar = CookieJar()

import sys

default_encoding = 'utf-8'
reload(sys)
sys.setdefaultencoding(default_encoding)


class PeolistSpider(scrapy.Spider):
    name = 'peolist'
    # allowed_domains = ['search.jiayuan.com']
    start_urls = ['http://search.jiayuan.com/v2/search_v2.php']
    # allowed_domains = ['baidu.com']
    # start_urls = ['https://www.baidu.com']
    cookiestr = {
        "_HASH": "878b45cc9ecd6579ce0a560a16f36a68",
        "IM_CS": "1",
        "IM_ID": "3",
        "IM_S": "%7B%22IM_CID%22%3A2138176%2C%22svc%22%3A%7B%22code%22%3A0%2C%22nps%22%3A0%2C%22unread_count%22%3A%2220%22%2C%22ocu%22%3A0%2C%22ppc%22%3A0%2C%22jpc%22%3A0%2C%22regt%22%3A%221490513353%22%2C%22using%22%3A%22%22%2C%22user_type%22%3A%2210%22%2C%22uid%22%3A162574980%7D%7D",
        "PHPSESSID": "893435dc9ba3fc39c70a7b4257317a53",
        "PROFILE": "162574980%3Aabc%3Am%3Aat1.jyimg.com%2F87%2F68%2F8b45cc9ecd6579ce0a560a16f36a%3A1%3A%3A1%3A8b45cc9ec_1_avatar_p.jpg%3A1%3A1%3A50%3A10",
        "RAW_HASH": "nmvxiJ9gMVnJnoSPAbgPbnY4qrsqYkEDTYWVJ2KsrRZV0Mx8t0%2AcxewFImeK7Wf1IVWuF0QItYZSXGK0TKRKFF9y4CEUHp3kazHmet4wsj%2A9Pbw.",
        "REG_REF_URL": "http://login.jiayuan.com/logout2.php",
        "SESSION_HASH": "497f12e805220d748d11d43e3e746434c78bd47b",
        "guider_quick_search": "on",
        "last_login_time": "1519780052",
        "main_search:162574980": "%7C%7C%7C00",
        "myage": "35",
        "myincome": "30",
        "myloc": "37%7C3707",
        "mysex": "m",
        "myuid": "161574980",
        "pclog": "%7B%22162574980%22%3A%221519780059185%7C1%7C0%22%7D",
        "pop_time": "1.51978E+12",
        "save_jy_login_name": "15711340912",
        "stadate1": "161574980",
        "upt": "PDoPEsJZ4SrjQPpoEsjDJkiBxz3EFtGfH6XHpEri8SlvJner5IPZX9QpgM88LIFHDFGc-pDqjxqrdfqslA..",
        "user_access": "1",
        "user_attr": "0",
        "Response Cookies": "",
        "is_searchv2": "1",
        "sk": "deleted"
    }

    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Content-Length": "129",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        # "Cookie": cookiestr,
        # "Cookie": "guider_quick_search=on; save_jy_login_name=15711340912; myuid=161574980; PHPSESSID=893435dc9ba3fc39c70a7b4257317a53; main_search:162574980=%7C%7C%7C00; is_searchv2=1; REG_REF_URL=http://login.jiayuan.com/logout2.php; stadate1=161574980; myloc=37%7C3707; myage=35; mysex=m; myincome=30; user_attr=000000; IM_S=%7B%22IM_CID%22%3A2138176%2C%22svc%22%3A%7B%22code%22%3A0%2C%22nps%22%3A0%2C%22unread_count%22%3A%2220%22%2C%22ocu%22%3A0%2C%22ppc%22%3A0%2C%22jpc%22%3A0%2C%22regt%22%3A%221490513353%22%2C%22using%22%3A%22%22%2C%22user_type%22%3A%2210%22%2C%22uid%22%3A162574980%7D%7D; user_access=1; PROFILE=162574980%3Aabc%3Am%3Aat1.jyimg.com%2F87%2F68%2F8b45cc9ecd6579ce0a560a16f36a%3A1%3A%3A1%3A8b45cc9ec_1_avatar_p.jpg%3A1%3A1%3A50%3A10; COMMON_HASH=878b45cc9ecd6579ce0a560a16f36a68; last_login_time=1519780052; upt=PDoPEsJZ4SrjQPpoEsjDJkiBxz3EFtGfH6XHpEri8SlvJner5IPZX9QpgM88LIFHDFGc-pDqjxqrdfqslA..; pclog=%7B%22162574980%22%3A%221519780059185%7C1%7C0%22%7D; RAW_HASH=nmvxiJ9gMVnJnoSPAbgPbnY4qrsqYkEDTYWVJ2KsrRZV0Mx8t0%2AcxewFImeK7Wf1IVWuF0QItYZSXGK0TKRKFF9y4CEUHp3kazHmet4wsj%2A9Pbw.; pop_time=1519784955790; SESSION_HASH=497f12e805220d748d11d43e3e746434c78bd47b; IM_CS=1; IM_ID=3",
        "Host": "search.jiayuan.com",
        "Origin": "http://search.jiayuan.com",
        "Proxy-Connection": "keep-alive",
        "Referer": "http://search.jiayuan.com/v2/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    }

    # cookiestr="SESSION_HASH=5f3cad9069499bbcd814b0194795569232ba618f; save_jy_login_name=15711340912; myuid=161574980; PHPSESSID=4224fa8945ea806e352b227e8e738130; main_search:162574980=%7C%7C%7C00; jy_safe_tips_new=xingfu; REG_REF_URL=http://login.jiayuan.com/logout2.php; user_access=1; stadate1=161574980; myloc=37%7C3707; myage=35; PROFILE=162574980%3Aabc%3Am%3Aat1.jyimg.com%2F87%2F68%2F8b45cc9ecd6579ce0a560a16f36a%3A1%3A%3A1%3A8b45cc9ec_1_avatar_p.jpg%3A1%3A1%3A50%3A10; mysex=m; myincome=30; COMMON_HASH=878b45cc9ecd6579ce0a560a16f36a68; sl_jumper=%26cou%3D17%26omsg%3D0%26dia%3D0%26lst%3D2018-02-27; last_login_time=1519724669; user_attr=000000; RAW_HASH=l57JHxgvPzk%2ASFevXugiYJLw25M6yiT4NfhFGRvJHnioVyE7kXI8d3D-p6nqnrV9bT0QQz49dxY-SufFmz0DrpnlVOOw7Nj34lcninfrv1cofFs."
    # print cookiestr
    # cookiestr = urllib.unquote(cookiestr)
    # print cookiestr
    # headers['Cookie'] = Cookieutil(start_urls[0]).getCookie()
    # print(headers['Cookie'])
    # exit(0)

    def start_requests(self):
        # aab = scrapy.Request(self.start_urls[0], callback=self.request_captcha(self), cookies=self.get_txtcookie(self))
        aab = scrapy.FormRequest(
            dont_filter=True,
            url=self.start_urls[0],
            formdata={
                "sex": "f",
                "key": "",
                "stc": "1:37,2:30.40,3:165.175,23:1",
                "sn": "default",
                "sv": "1",
                "p": "1",
                "f": "",
                "listStyle": "bigPhoto",
                "pri_uid": "162574980",
                "jsversion": "v5",
            },
            meta={
                'cookiejar': 1
            },
            callback=self.request_callback,
            # cookies=self.cookiestr,
            headers=self.headers
        )
        print(aab)
        # print '\n'.join(['%s:%s' % item for item in aab.__dict__.items()])
        yield aab
        # 这里带着cookie发出请求
        # yield scrapy.Request(
        #     url=self.start_urls,
        #     headers=self.headers,
        #     meta={
        #         'cookiejar': 1
        #     },
        #     callback=self.request_captcha
        # )

    def request_callback(self, response):
        print "2222222222222"
        print response
        sel = Selector(response)
        print "callbacked"
        pass

    def parse(self, response):
        pass
