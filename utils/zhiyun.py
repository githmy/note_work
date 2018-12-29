# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import os
# import django
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cmdb.settings')
# django.setup()
import pymysql
import threading
import logging
from asset.models import Assets
from aiops.utils.aiops_cmdb_api import ZhiyunApi

logger = logging.getLogger("django")


class MysqlDB:
    def __init__(self, conf):
        self.config = {
            'host': conf['host'],
            'user': conf['user'],
            'password': conf['password'],
            'database': conf['database'],
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.conn = pymysql.connect(**self.config)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()
        # Log.sql_log.(error|info|debug)

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def exec_sql(self, strsql):
        """
        write data, such as insert, delete, update
        :param strsql: string sql
        :return: affected rows number
        return 0 when errors
        """
        try:
            self.lock.acquire()
            self.conn.ping(True)
            res = self.cursor.execute(strsql)
            if strsql.strip().lower().startswith("select"):
                res = self.cursor.fetchall()
            self.conn.commit()
            return res
        except Exception as ex:
            logging.error("exec sql error:")
            logging.error(strsql, exc_info=True)
            return 0
        finally:
            self.lock.release()


def get_zy_cmdb_count():
    data = {}
    try:
        config = {
            'host': "10.128.64.100",
            'user': "zhiyun",
            'password': "zhiyun@tencent",
            'database': "dcos_cmdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        zy_mysql = MysqlDB(config)
        service_count_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info`"""
        service_count_phy_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info` WHERE svr_type ='0';"""
        rack_countrack_count_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_server_basic_info` GROUP BY svr_rack_name;"""
        online_network_count_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_network_basic_info`;"""
        idc_export_line_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_idc_export_line_info` """
        special_idc_line_sql = """SELECT COUNT(*) as cot FROM `t_cmdb_special_idc_line_info` """
        service_count = zy_mysql.exec_sql(service_count_sql)
        service_count_phy = zy_mysql.exec_sql(service_count_phy_sql)
        rack_countrack_count = zy_mysql.exec_sql(rack_countrack_count_sql)
        online_network_count = zy_mysql.exec_sql(online_network_count_sql)
        instock_network_count = Assets.objects.filter(type='network', status='instock').count()
        idc_export_line_count = zy_mysql.exec_sql(idc_export_line_sql)
        special_idc_line_count = zy_mysql.exec_sql(special_idc_line_sql)
        network_count = online_network_count[0].get('cot', 0) + instock_network_count
        data = {
            'service_count': service_count[0].get('cot', 0) if service_count else 3500,
            'network_count': network_count,
            'service_count_phy': service_count_phy[0].get('cot', 0) if service_count_phy else 1000,
            'rack_count': len(rack_countrack_count) if rack_countrack_count else 50,
            'tenant_count': 17,
            'idc_export_line_count': idc_export_line_count[0].get('cot', 0) if idc_export_line_count else 4,
            'special_idc_line_count': special_idc_line_count[0].get('cot', 0) if special_idc_line_count else 12,
        }
        data['servoce_count_vm'] = int(data['service_count']) - int(data['service_count_phy'])
    except:
        pass
    return data


def get_tenant_server_count(tenantname):
    data  = (0,0)
    config = {
        'host': "10.128.64.100",
        'user': "zhiyun",
        'password': "zhiyun@tencent",
        'database': "dcos_cmdb",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
        'cursorclass': pymysql.cursors.DictCursor
    }
    zy_mysql = MysqlDB(config)

    get_module_id1_sql = """SELECT id from chelun_cmdb.modules WHERE `name`='%s' and `level`=1""" % tenantname
    module1_id = zy_mysql.exec_sql(get_module_id1_sql)
    if module1_id:
        zy_api = ZhiyunApi()
        module3_ids = zy_api.get_module3_ids_by_module1_id(module1_id[0]['id'])
        if len(module3_ids) ==1:
            module3_ids.append(0)
    else:
        return data
    pm_service_count_sql = """SELECT count(*) as cot FROM `t_cmdb_server_basic_info` as server
JOIN  chelun_cmdb.modules as yewu on  yewu.id =`server`.svr_bussiness_id
WHERE yewu.id in %s and server.svr_type=0 """ % str(tuple(module3_ids))
    vm_service_count_sql = """SELECT count(*) as cot FROM `t_cmdb_server_basic_info` as server
    JOIN  chelun_cmdb.modules as yewu on  yewu.id =`server`.svr_bussiness_id
    WHERE yewu.id in %s and server.svr_type=1 """ % str(tuple(module3_ids))
    pm_service_count = zy_mysql.exec_sql(pm_service_count_sql)
    vm_service_count = zy_mysql.exec_sql(vm_service_count_sql)
    data =(pm_service_count[0]['cot'],vm_service_count[0]['cot'])
    return data


if __name__ == '__main__':
    print get_tenant_server_count("斑马汽车")
