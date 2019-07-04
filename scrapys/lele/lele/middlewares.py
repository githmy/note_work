# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/spider-middleware.html
import random
import scrapy
from scrapy.conf import settings
from scrapy.http import HtmlResponse
from scrapy import signals
from selenium import webdriver
from selenium.webdriver import Chrome
import json
import time
import base64
import urllib
import urllib.request


# 默认主爬虫插件
class LeleSpiderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    # download 中间件后， 进入spiders的 parse_XX之前。
    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    # 爬虫运行yield item或者yield scrapy.Request()的时候调用。
    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, dict or Item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Response, dict
        # or Item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


# 下载设置插件
class LeleDownloaderMiddleware(object):
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    # 当每个request通过下载中间件时，该方法被调用。
    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    # 当下载器完成http请求，传递响应给引擎的时候调用
    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


# ip代理插件
class ProxyMiddleware(object):
    def process_request(self, request, spider):
        proxy = random.choice(settings['PROXIES'])
        request.meta['proxy'] = proxy


# ip test proxy func
class ProxyMiddleware(object):
    def flush_ip_pool(self):
        seed_url = "https://www.xicidaili.com/nn"
        test_url = "https://www.whatismyip.com/my-ip-information/?iref=home"
        # 1. 请求
        response = urllib.request.urlopen(seed_url)
        cat_vido = response.read()

    def get_random_ip(self):
        # order是订单号或者序列号
        order = "xxxxxxxxxxxxxxxxxxx"
        APIurl = "http://xxxxxxxxxxxxxxxxxxxx" + order + ".html"
        res = urllib.request.urlopen(APIurl).read().decode("utf-8")
        IPs = res.split("\n")
        proxyip = random.choices(IPs)
        # print(proxyip)
        return 'http://' + proxyip

    def process_request(self, request, spider):
        random.random(IPs)
        ip = self.get_random_ip()
        print("Current IP:Port is %s" % ip)
        request.meta['proxy'] = ip

    def process_response(self, request, response, spider):
        return response


# 请求头插件
class HeaderMiddleware(object):
    def process_request(self, request, spider):
        # 对代理数据进行base64编码
        encoded_user_pass = base64.encodebytes('user_pass')
        # 添加到HTTP代理格式里
        request.headers['Proxy-Authorization'] = 'Basic ' + encoded_user_pass


# user agent 代理插件
class UAMiddleware(object):
    def process_request(self, request, spider):
        ua = random.choice(settings['USER_AGENT_LIST'])
        request.headers['User-Agent'] = ua


# 手动点击辅助插件
class SeleniumMiddleware(object):
    def __init__(self):
        # self.driver = webdriver.Chrome('./chromedriver')
        self.driver = webdriver.Chrome("D:\Chrome下载\chromedriver.exe")

    def process_request(self, request, spider):
        if spider.name == 'seleniumSpider':
            self.driver.get(request.url)
            time.sleep(2)
            body = self.driver.page_source
            return HtmlResponse(self.driver.current_url,
                                body=body,
                                encoding='utf-8',
                                request=request)


class LoginMiddleware(object):
    def __init__(self):
        self.client = redis.StrictRedis()

    def process_request(self, request, spider):
        if spider.name == 'loginSpider':
            cookies = json.loads(self.client.lpop('cookies').decode())
            request.cookies = cookies


class RetryMiddleWare(scrapy.Spider):
    name = "middle_ware_scrapy"
    allowed_domains = []
    start_urls = []

    def start_requests(self):
        yield scrapy.Request(url=self.start_urls[0], method="post", body=json.dumps({}), headers={})

    def parse(self, response):
        print(response.body.decode())


# 失败再测尝试插件。
class RetryOfDateMiddleWare(RetryMiddleWare):
    def __init__(self, settings):
        RetryMiddleWare.__init__(self, settings)

    def process_exception(self, request, exception, spider):
        # 如果失败
        self.remove_broken_proxy()

    def remove_broken_proxy(self):
        pass

    def process_response(self, request, response, spider):
        return_str = response.body.decode()
        request_url = response.url
        # 重定向url
        origin_url = request.meta["redirect_urls"][0]
        # 请求时间设置
        request.meta["request_start_time"] = time.time()
        #
        # ['url', 'method', 'headers', 'body', 'cookies', 'meta', 'flags',
        #  'encoding', 'priority', 'dont_filter', 'callback', 'errback']
        next_request = request.replace(url=origin_url, method="POST", body=json.dumps({}), headers={})
        return next_request


# 爬虫管道异常报错
class ExceptionCheckSpider(object):
    def process_spider_exception(self, response, exception, spider):
        print(f'返回的内容是：{response.body.decode()}\n报错原因：{type(exception)}')
        # 可以返回None，也可以运行yield item语句或者像爬虫的代码一样，使用yield scrapy.Request()，但运行之后就绕过了原来的代码。
        return None
