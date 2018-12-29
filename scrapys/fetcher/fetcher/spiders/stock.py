# -*- coding: utf-8 -*-
import scrapy
import json
from scrapy.selector import Selector
from ..items import FetcherItem

class StockSpider(scrapy.Spider):
    name = 'stock'
    allowed_domains = ['finance.sina.com.cn']
    # start_urls = ['http://finance.sina.com.cn/']

    # 自定义url
    def start_requests(self):
        pages = []
        # for i in range(0, 3999):
        for i in range(0, 3):
            url = 'http://finance.sina.com.cn/realstock/company/sh60%s/nc.shtml' % str(i).rjust(4, "0")
            page = scrapy.Request(url)
            pages.append(page)
        # for i in range(0, 2999):
        for i in range(0, 2):
            url = 'http://finance.sina.com.cn/realstock/company/sz00%s/nc.shtml' % str(i).rjust(4, "0")
            page = scrapy.Request(url)
            pages.append(page)
        return pages

    # 处理每个url文件
    def parse(self, response):
        node_list = Selector(response).xpath('//div[@id="picContainer"]//div[@class="tab"]/text()')
        print(node_list)

        for node in node_list:
            item = FetcherItem()
            # extract() 转为uncode字符串
            name = node.extract()
            # name = node.xpath("./h3/text()").extract()
            # title = node.xpath("./h4/text()").extract()
            # info = node.xpath("./p/text()").extract()
            print(name)

            item['id'] = name[0]
            item['na'] = name[1]
            item['op'] = name[2]

            # 返回给管道，引擎判断是item还是列表
            yield item
            # # 返回给调度器，引擎判断是url
        if len(response.xpath("//a[@class='noactive' and @id='next']")) == 0:
            url = response.xpath("//a[@id='next']/@href").extract()[0]
            yield scrapy.Request(url, callback=self.parse)

            # item = FetcherItem()

    # 处理每个url文件
    def parseimage(self, response):
        data_list = json.loads(response.body)['data']
        if len(data_list) == 0:
            return

        for data in data_list:
            item = FetcherItem()
            item['nickname'] = data['nickname']
            item['imagelink'] = data['imagelink']
            yield item

        self.offset = 20
        yield scrapy.Request(self.baseurl + str(self.offset), callback=self.parseimage)
