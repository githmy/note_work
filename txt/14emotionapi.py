# -*- coding: utf-8 -*-
import hashlib
import time
import random
import string
import urllib
import sys


def get_params(plus_item):
    '''请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效）'''
    t = time.time()
    time_stamp = int(t)

    '''请求随机字符串，用于保证签名不可预测'''
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))

    '''应用标志，这里修改成自己的id和key'''
    app_id = '1107002492'
    app_key = 'TXU39mL5afomJcU7'

    '''值使用URL编码，URL编码算法用大写字母'''
    text1 = plus_item
    text = urllib.quote(text1.decode(sys.stdin.encoding).encode('utf8')).upper()

    '''拼接应用密钥，得到字符串S'''
    sign_before = 'app_id=' + app_id + '&nonce_str=' + nonce_str + '&text=' + text + '&time_stamp=' + str(
        time_stamp) + '&app_key=' + app_key

    '''计算MD5摘要，得到签名字符串'''
    m = hashlib.md5()
    m.update(sign_before)
    sign = m.hexdigest()
    sign = sign.upper()

    params = 'app_id=' + app_id + '&time_stamp=' + str(
        time_stamp) + '&nonce_str=' + nonce_str + '&sign=' + sign + '&text=' + text

    return params


import requests
# import md5sign
from bs4 import BeautifulSoup
import json
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def get_content(plus_item):
    url = "https://api.ai.qq.com/fcgi-bin/nlp/nlp_textpolar"  # API地址
    # params = md5sign.get_params(plus_item)  # 获取请求参数
    params = get_params(plus_item)  # 获取请求参数
    url = url + '?' + params  # 请求地址拼接

    r = requests.get(url)
    print r.text
    soup = BeautifulSoup(r.text, 'lxml')
    print 3
    allcontents = soup.select('body')[0].text.strip()
    print 4
    allcontents_json = json.loads(allcontents)  # str转成dict
    print 5
    try:
        r = requests.get(url)
        print r
        soup = BeautifulSoup(r.text, 'lxml')
        print 3
        allcontents = soup.select('body')[0].text.strip()
        print 4
        allcontents_json = json.loads(allcontents)  # str转成dict
        print 5

        return allcontents_json["data"]["polar"], allcontents_json["data"]["confd"], allcontents_json["data"]["text"]
    except Exception, e:
        print 'a', str(e)
        return 0, 0, 0


if __name__ == '__main__':
    polar, confd, text = get_content('今天天气真好')
    print polar, confd, text
    print '情感倾向：' + str(polar) + '\n' + '程度：' + str(confd) + '\n' + '文本：' + text

# if __name__ == '__main__':
#     # panda_get_data()
#     plus_item = "今天好开心"
#     aa = get_params(plus_item)
#     print aa
