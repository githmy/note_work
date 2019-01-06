# coding=utf-8
import requests, re, json, time, os
import heapq
from bs4 import BeautifulSoup


class GPINFO(object):
    """docstring for GPINFO"""

    def __init__(self):
        self.Url = 'http://quote.eastmoney.com/stocklist.html'
        self.BaseData = []
        self.Date = time.strftime('%Y%m%d')
        self.Record = 'basedata' + self.Date
        if os.path.exists(self.Record):
            print('record exist...')
            self.BaseData = self.get_base_data_from_record()
        else:
            print('fuck-get data again...')
            self.get_data()

    # 将数据写入到记录文件
    def write_record(self, text):
        with open(self.Record, 'ab') as f:
            f.write((text + '\n').encode('utf-8'))

    # 从记录文件从读取数据
    def get_base_data_from_record(self):
        ll = []
        with open(self.Record, 'rb') as f:
            json_l = f.readlines()
            for j in json_l:
                ll.append(json.loads(j.decode('utf-8')))
        return ll

    # 爬虫获取数据
    def get_data(self):
        # 请求数据
        orihtml = requests.get(self.Url).content
        # 创建 beautifulsoup 对象
        soup = BeautifulSoup(orihtml, 'lxml')
        # 采集每一个股票的信息
        count = 0
        for a in soup.find('div', class_='quotebody').find_all('a', {'target': '_blank'}):
            record_d = {}
            # 代号
            num = a.get_text().split('(')[1].strip(')')  # 获取股票代号
            if not (num.startswith('00') or num.startswith('60')): continue  # 只需要6*/0*    只要以00或60开头的股票代号
            record_d['num'] = num
            # 名称
            name = a.get_text().split('(')[0]  # 获取股票名称
            record_d['name'] = name
            # 详情页
            detail_url = a['href']
            record_d['detail_url'] = detail_url
            cwzburl = detail_url
            # 发送请求
            try:
                cwzbhtml = requests.get(cwzburl, timeout=30).content  # 爬取股票详情页
            except Exception as e:
                print('perhaps timeout:', e)
                continue
            # 创建soup对象
            cwzbsoup = BeautifulSoup(cwzbhtml, 'lxml')
            # 财务指标列表 [浦发银行，总市值    净资产    净利润    市盈率    市净率    毛利率    净利率    ROE] roe:净资产收益率
            try:
                cwzb_list = cwzbsoup.find('div',
                                          class_='cwzb').tbody.tr.get_text().split()  # 获取class为cwzb的div下第一个tbody下第一个tr获取内部文本，并使用空格分割
            except Exception as e:
                print('error:', e)
                continue
            # 去除退市股票
            if '-' not in cwzb_list:
                record_d['data'] = cwzb_list  # 将数据加入到字典中
                self.BaseData.append(record_d)  # 将字典加入到总数据总
                self.write_record(json.dumps(record_d))  # 将字典类型转化为字符串，写入文本
                count = count + 1
                print(len(self.BaseData))


def main():
    test = GPINFO()
    result = test.BaseData
    # [浦发银行，总市值    净资产    净利润    市盈率    市净率    毛利率    净利率    ROE] roe:净资产收益率]
    top_10 = heapq.nlargest(10, result, key=lambda r: float(r['data'][7].strip('%')))  # 获取前10名利率最高者的数据
    for item in top_10:
        for key in item['data']:
            print(key),
        print('\n')


# 打印字符串时，使用print str.encode('utf8');
# 打印中文列表时，使用循环 for key in list：print key

# 打印中文字典时，可以使用循环，也可以使用json：
#  import json
# print json.dumps(dict, encoding='UTF-8', ensure_ascii=False)


if __name__ == '__main__':
    main()
