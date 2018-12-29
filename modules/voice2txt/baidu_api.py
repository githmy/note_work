#!/usr/bin/python
# -*- coding: UTF-8 -*-
from aip import AipSpeech
import os
import urllib, base64
import urllib.parse, urllib.request
import simplejson


# pip3 install baidu-aip


# 读取文件
def get_file_content(filePath):
    # f = open("../nocode/q2.png", 'rb')
    # return f.read()
    with open(filePath, 'rb') as fp:
        return fp.read()


def baidu_voice():
    """ 你的 APPID AK SK """
    APP_ID = '11563677'
    API_KEY = '9biOFeUr5x4tKUM9EcVZuObS'
    SECRET_KEY = '8DIZ5eNfnFINXzYaji75gVrTMYe6I3Np'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    # 识别本地文件
    # result = client.asr(get_file_content(os.path.join("..", "nocode", "D32_999.wav")), 'wav', 16000, {
    # result = client.asr(get_file_content(os.path.join("..", "nocode", "安利1.wav")), 'wav', 16000, {
    # result = client.asr(get_file_content(os.path.join("..", "nocode", "背景音乐新闻1.wav")), 'wav', 16000, {
    # result = client.asr(get_file_content(os.path.join("..", "nocode", "脱口秀1.wav")), 'wav', 16000, {
    # result = client.asr(get_file_content(os.path.join("..", "nocode", "外景采访1.wav")), 'wav', 16000, {
    # result = client.asr(get_file_content(os.path.join("..", "nocode", "外景新闻1.wav")), 'wav', 16000, {
    result = client.asr(get_file_content(os.path.join("..", "nocode", "gctest.wav")), 'wav', 16000, {
        'dev_pid': 1536,
    })
    print(result)


def baidu_picture():
    """
    #!/bin/bash
    curl -i -k 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=【百度云应用的AK】&client_secret=【百度云应用的SK】'
    curl -i -k 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=Gy4LUvQy6OYZ6iQeV1nqDHjH&client_secret=3rjG9tKm289HepOcmhQfHGUMKLpeu8iD'
    """
    # access_token = '#####调用鉴权接口获取的token#####'
    access_token = '24.fbe93d7bdb780863a16500dabfac8a00.2592000.1534672137.282335-11564058'
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token=' + access_token
    # 二进制方式打开图文件
    f = open(r"../nocode/q1.png", 'rb')
    # f = open(r"../nocode/char_like.png", 'rb')
    # f = open(r"../nocode/news.png", 'rb')
    # 参数image：图像base64编码
    img = base64.b64encode(f.read())
    params = {"image": img}
    # params = urllib.parse.urlencode(params)
    params = urllib.parse.urlencode(params).encode('utf-8')
    # request = urllib.request.Request(url, params)
    request = urllib.request.Request(url=url, data=params, method='POST')
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    content = simplejson.loads(content )
    if (content):
        for i in content["words_result"]:
            print(i["words"])


if __name__ == '__main__':
    baidu_voice()
    # baidu_picture()
