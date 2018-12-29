# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/6 13:37
# @Author  : abc

from __future__ import unicode_literals
import requests
import json
import datetime
import time


class ShishuoApi:
    def __init__(self):
        self.host = None
        # self.host = 'http://10.5.204.58:5000'
        self.api = None
        # self.apikey = '242aa4b2-0607-11e7-9521-46ccfb50c74d'
        # self.headers = {'apikey': self.apikey}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows XP) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        # self.headers = {}
        self.data = None
        self.method = requests.post

    def sent_request(self, **kwargs):
        url = self.host + self.api
        if self.method == requests.get:
            r = self.method(url=url, headers=self.headers, params=self.data, timeout=60, **kwargs)
        else:
            r = self.method(url=url, headers=self.headers, data=self.data, timeout=60, **kwargs)
        return r

    def get_request_exam(self, data):
        self.api = "/parse"
        self.data = json.dumps(data)
        self.method = requests.post
        return self.sent_request(cookies={'userName': 'aaa'})

    def get_rasa_intents(self, host, api, str, proj, model):
        self.host = host
        self.api = api
        data = {
            "q": str,
            "project": proj,
            "model": model
        }
        self.data = json.dumps(data)
        self.method = requests.post
        return self.sent_request().json()


if __name__ == '__main__':
    sh_api = ShishuoApi()
    a = sh_api.get_rasa_intents('今天熊市还是牛市', "shishuo", "model_root")
    print a
    print len(a)
