# -*- coding: utf-8 -*-
import requests
import datetime
import time
from threading import Thread

######
# import os
# import sys
# import django
# pro_dir = os.getcwd()  # 如果放在project目录，就不需要在配置绝对路径了
# sys.path.append(pro_dir)
# os.environ['DJANGO_SETTINGS_MODULE'] = 'cmdb.settings'  # 项目的settings
# django.setup()
######
from server.models import Virtual
import Queue

templateUrl = 'http://10.129.254.18/jitstack/v2/managerinstancelist?zoneid={}&offset={}&limit={}'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows XP) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}


def getZone():
    url = 'http://10.129.254.18/jitstack/v2/managerzonelist'
    result = requests.get(url=url, headers=headers).json()['zones']
    return [item for item in result if item['name'] != 'STDC-01-DEV']


def appendList(zoneList, result):
    for item in result['items']:
        if item['hostId'] != u'null' and item['hostId'] is not None:
            zoneList.append(Virtual(
                physical=item['hostId'],
                virtual=item['name'],
                username=item['username'],
                zone=item['zonename'],
                create=item['createTime'],
                status=item['status']['name'],
                ip=item['ip'],
                elasticIP=item['elasticIp'],
                mem=item['ramCount'],
                cpu=item['cpuCount'],
                osType=item['ostype'],
                dataVolume=pars_volumes(item['volumes'])if item['volumes'] else None,
                sysVolumeCapability=item['sysVolumeCapability'],
                sysVolumeId=item['sysVolumeId ']))


def pars_volumes(volumes):
    try:
        total = []
        for volume in volumes:
            total.append(str(volume['capability']))
        return '+'.join(total)
    except:
        return None


def getVHost(zoneid, offset=0, limit=25):
    endTags = True
    zoneList = []
    while endTags:
        result = requests.get(url=templateUrl.format(zoneid, offset, limit), headers=headers).json()
        if int(offset) + limit < result['totalItems']:
            print("zoneid:{};offset:{};limit:{};resule:{};total:{}".format(zoneid, offset, limit, len(result['items']),
                                                                           result['totalItems']))
            appendList(zoneList, result)
            offset = int(offset) + limit
        else:
            print("zoneid:{};offset:{};limit:{};resule:{};total:{}".format(zoneid, offset, limit, len(result['items']),
                                                                           result['totalItems']))
            appendList(zoneList, result)
            Virtual.objects.bulk_create(zoneList)
            endTags = False


def apiTake():
    for zone in getZone():
        zoneId = zone['objectId']
        t = Thread(target=getVHost, args=(zoneId,))
        # t.setDaemon(True)
        t.start()


# getVHost(zoneid='327a58e5-29d7-4fe9-bea3-ee0c44fc1887')
# getVHost('f801178a-b7c4-4cca-9eb7-1e7e082508dc')
# apiTake()

# ----------------队列urls
ThreadNum = 6


class GetUrlsQ:
    def __init__(self):
        self.urlsQ = Queue.Queue()
        self.templateUrl = 'http://10.129.254.18/jitstack/v2/managerinstancelist?zoneid={}&offset={}&limit={}'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows XP) AppleWebKit/537.36 Chrome/58.0.3029.110 Safari/537.36'
        }

    def getZone(self, url):
        result = requests.get(url=url, headers=self.headers).json()['zones']
        return [item for item in result if item['name'] != 'STDC-01-DEV']

    def getZoneUrls(self, zoneid, offset=0, limit=25):
        endTags = True
        result = requests.get(url=self.templateUrl.format(zoneid, offset, limit), headers=self.headers).json()
        while endTags:
            if int(offset) + limit < result['totalItems']:
                self.urlsQ.put(self.templateUrl.format(zoneid, offset, limit))
                # print(templateUrl.format(zoneid, offset, limit))
                offset = int(offset) + limit
            else:
                self.urlsQ.put(self.templateUrl.format(zoneid, offset, limit))
                # print(templateUrl.format(zoneid, offset, limit))
                endTags = False

    def __call__(self, url):
        ths = []
        for zone in self.getZone(url):
            zoneId = zone['objectId']
            t = Thread(target=self.getZoneUrls, args=(zoneId,))
            t.start()
            ths.append(t)
        for th in ths:
            th.join()
        return self.urlsQ


class GetORMObj:
    def __init__(self):
        self.zoneList = []

    def op_http(self, urlsQ):
        while urlsQ.empty() == False:
            url = urlsQ.get()
            try:
                for item in requests.get(url=url).json()['items']:
                    if item['hostId'] != u'null' and item['hostId'] is not None:
                        self.zoneList.append(Virtual(
                            physical=item['hostId'],
                            virtual=item['name'],
                            username=item['username'],
                            zone=item['zonename'],
                            create=item['createTime'],
                            status=item['status']['name'],
                            ip=item['ip'],
                            elasticIP=item['elasticIp'],
                            mem=item['ramCount'],
                            cpu=item['cpuCount'],
                            osType=item['ostype'],
                            dataVolume=pars_volumes(item['volumes'])if item['volumes'] else None,
                            sysVolumeCapability=item['sysVolumeCapability'],
                            sysVolumeId=item['sysVolumeId ']))
                print("Get: {}".format(url))
            except Exception as e:
                print("Get op_http error:{}, url:{}".format(e, url))
                urlsQ.put(url)
                time.sleep(3)
            finally:
                urlsQ.task_done()

    def __call__(self, urlsQ):
        for th in range(ThreadNum):
            t = Thread(target=self.op_http, args=(urlsQ,))
            t.setDaemon(True)
            t.start()
        urlsQ.join()
        return self.zoneList


def run():
    resQ = GetUrlsQ()('http://10.129.254.18/jitstack/v2/managerzonelist')
    r = GetORMObj()(resQ)
    try:
        from django.db import connection
        connection.cursor().execute('truncate table server_virtual')
        Virtual.objects.bulk_create(r)
        print(r.__len__())
    except Exception as e:
        print(e)


if __name__ == "__main__":
    run()
