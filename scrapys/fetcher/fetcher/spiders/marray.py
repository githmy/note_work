# -*- coding: utf-8 -*-
import re
import scrapy
import codecs
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy.selector import Selector
# from scrapy import log

import sys

default_encoding = 'utf-8'
reload(sys)
sys.setdefaultencoding(default_encoding)


class MarraySpider(scrapy.Spider):
    name = 'marray'
    allowed_domains = ['www.jiayuan.com']
    start_urls = ['http://www.jiayuan.com']

    def parse_with_cookie(self, response):
        file = codecs.open('page.html', 'w', encoding='utf-8')
        file.write(response.body)
        file.close()

    def start_requests(self):
        Request("http://usercp.jiayuan.com/?from=login",
                      cookies={'viewed': '"1083428"', '__utmv': '30149280.3975'},
                      callback=self.parse_with_cookie)
        # yield Request("http://usercp.jiayuan.com/?from=login",
        #               cookies={'viewed': '"1083428"', '__utmv': '30149280.3975'},
        #               callback=self.parse_with_cookie)
        pages = []
        for i in range(0, 2999):
            url = 'http://finance.sina.com.cn/realstock/company/sz00%s/nc.shtml' % str(i).rjust(4, "0")
            page = scrapy.Request(url)
            pages.append(page)
        return pages

    def parse(self, response):
        pass
