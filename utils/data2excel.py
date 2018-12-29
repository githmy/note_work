# _*_coding:utf-8_*_

import re
from datetime import datetime

import xlrd
import xlwt
# import MySQLdb
from django.db import connection
from xlutils.copy import copy

from cmdb.settings import BASE_DIR
from dateformat import date2str


class IponeException(Exception):
    def __init__(self, message):
        Exception.__init__(self)
        self.message = message


def get_data(sql):
    # 创建数据库连接.
    # connection = MySQLdb.connect(host='10.135.64.189', user='ppdb', passwd='ppdb', db='ppdb', port=3306, charset='utf8')
    # 创建游标
    cur = connection.cursor()
    # 执行查询，
    cur.execute(sql)
    # 由于查询语句仅会返回受影响的记录条数并不会返回数据库中实际的值，所以此处需要fetchall()来获取所有内容。
    result = cur.fetchall()
    # 关闭游标
    cur.close()
    # 关闭数据库连接
    # connection.close
    # 返给结果给函数调用者。
    return result


def write_data_to_excel(path, name, sql, headarray):
    # 将sql作为参数传递调用get_data并将结果赋值给result,(result为一个嵌套元组)
    result = get_data(sql)
    # 实例化一个Workbook()对象(即excel文件)
    wbk = xlwt.Workbook()
    # 新建一个名为Sheet1的excel sheet。此处的cell_overwrite_ok =True是为了能对同一个单元格重复操作。
    sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)
    # 获取当前日期，得到一个datetime对象如：(2016, 8, 9, 23, 12, 23, 424000)
    today = datetime.today()
    # 将获取到的datetime对象仅取日期如：2016-8-9
    today_date = datetime.date(today)
    # 遍历result中的没个元素。
    for i in xrange(len(headarray)):
        sheet.write(0, i, headarray[i])
    for i in xrange(len(result)):
        # 对result的每个子元素作遍历，
        for j in xrange(len(result[i])):
            # 将每一行的每个元素按行号i,列号j,写入到excel中。
            sheet.write(i + 1, j, date2str(result[i][j]))
    # 以传递的name+当前日期作为excel名称保存。
    wbk.save(path + '/' + name + '.xls')
    # wbk.save(path + '/' + name + '_' + str(today_date) + '.xls')


"""
功能：将Excel数据导入到MySQL数据库
"""


def excel2mysql(fullname, tablename, ignarr, addarr):
    # def excel2mysql(path, name, sql, headarray):
    # 建立一个MySQL连接
    # database = MySQLdb.connect(host="localhost", user="root", passwd="", db="mysqlPython")
    # 获得游标对象, 用于逐行遍历数据库数据
    # cursor = database.cursor()
    cursor = connection.cursor()
    # ***************************** 1.读取字段对象 ***************************
    # 1.1. 查询
    query = 'desc `' + tablename + '`'
    try:
        cursor.execute(query)
        desckey = cursor.fetchall()  # 读取所有
    except Exception as e:
        cursor.close()
        raise "sql error when:" + query
    # 1.2. 属性组赋值
    increasarr = []
    intarr = []
    floatarr = []
    timearr = []
    for (hotelrs) in (desckey):
        if hotelrs[3] == "PRI":
            increasarr.append(hotelrs[0])
        else:
            if 'int' in hotelrs[1]:
                intarr.append(hotelrs[0])
            elif 'float' in hotelrs[1]:
                floatarr.append(hotelrs[0])
            elif 'datetime' in hotelrs[1]:
                timearr.append(hotelrs[0])

    # ***************************** 2.读取表格字段 ***************************
    # Open the workbook and define the worksheet
    book = xlrd.open_workbook(fullname)
    sheet = book.sheet_by_name("Sheet1")
    # 忽略表格的列指标
    ignoindex = []
    # 缓冲sql键数组
    bufkeyar = []
    # 该有没在表里的数组
    shouldhavar = []
    # 链接符
    connecstr = ','
    # 2.1.获取表格全字段,排除表格指定禁止书写的字段 or 排除表格自动禁止书写的字段，即 auto_increment
    # print ignarr
    # print increasarr
    # print ignarr
    for r in range(0, sheet.ncols):
        if sheet.cell(0, r).value in ignarr or sheet.cell(0, r).value in increasarr:
            ignoindex.append(r)
        else:
            bufkeyar.append('`' + sheet.cell(0, r).value + '`')
    for r in range(0, len(addarr[0])):
        if addarr[0][r] not in [str(i) for i in sheet.row_values(0)]:
            bufkeyar.append('`' + addarr[0][r] + '`')
            shouldhavar.append(r)
    keystr = connecstr.join(bufkeyar)
    # ***************************** 3.表格数据格式 ***************************
    # 3.生成values
    try:
        bufvaluesarr = []
        for r in range(1, sheet.nrows):
            # 4.生成value
            bufvaluearr = []
            for s in range(0, sheet.ncols):
                if s not in ignoindex:
                    # 3.1.根据字段类型自动转化格式
                    if sheet.cell(0, s).value in addarr[0]:
                        bufvaluearr.append(addarr[1][addarr[0].index(sheet.cell(0, s).value)])
                    elif str(sheet.cell(r, s).value).rstrip() == "":
                        bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in intarr:
                        try:
                            float(sheet.cell(r, s).value)
                            bufvaluearr.append(str(int(sheet.cell(r, s).value)))
                        except ValueError:
                            bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in floatarr:
                        try:
                            float(sheet.cell(r, s).value)
                            bufvaluearr.append(str(float(sheet.cell(r, s).value)))
                        except ValueError:
                            bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in timearr:
                        try:
                            bufvaluearr.append(
                                "\"" + str(xlrd.xldate.xldate_as_datetime(sheet.cell(r, s).value, 0)) + "\"")
                        except ValueError:
                            bufvaluearr.append("null")
                    else:
                        bufvaluearr.append("\"" + str(sheet.cell(r, s).value) + "\"")
            for s in range(0, len(shouldhavar)):
                bufvaluearr.append(addarr[1][shouldhavar[s]])
            bufvaluesarr.append('(' + connecstr.join(bufvaluearr) + ')')
            # 3.2.根据字段类型手动转化格式
            # ***************************** 4.sql写入 ***************************
            #     query = 'INSERT INTO `' + tablename + '` (' + keystr + ') VALUES (' + connecstr.join(bufvaluearr) + ')'
            #     print query
            # cursor.execute(query)
            # print cursor.lastrowid
        valustrs = connecstr.join(bufvaluesarr)
        # raise "fack"
        # # 3.2.根据字段类型手动转化格式
        # # ***************************** 4.sql写入 ***************************
        query = 'INSERT INTO `' + tablename + '` (' + keystr + ') VALUES ' + valustrs
        # print query
        cursor.execute(query)
        # print int(connection.insert_id())
        # cursor.execute(query, values)
        # 关闭游标
        cursor.close()
        # 提交
        # database.commit()
        connection.commit()
        # 关闭数据库连接
        # database.close()
        connection.close()
        return True
    except Exception as e:
        # 如果出现了错误，那么可以回滚，就是上面的三条语句要么执行，要么都不执行
        connection.rollback()
        connection.close()
        return e


def excel2mysql_audit_old(fullname, tablename, ignarr, addarr, device, eventype, user):
    # 建立一个MySQL连接
    # database = MySQLdb.connect(host="localhost", user="root", passwd="", db="mysqlPython")
    # 获得游标对象, 用于逐行遍历数据库数据
    # cursor = database.cursor()
    cursor = connection.cursor()
    # ***************************** 1.读取字段对象 ***************************
    # 1.1. 查询
    query = 'desc `' + tablename + '`'
    try:
        cursor.execute(query)
        desckey = cursor.fetchall()  # 读取所有
    except Exception as e:
        cursor.close()
        raise "sql error when:" + query
    # 1.2. 属性组赋值
    increasarr = []
    intarr = []
    floatarr = []
    timearr = []
    for (hotelrs) in (desckey):
        if hotelrs[3] == "PRI":
            increasarr.append(hotelrs[0])
        else:
            if 'int' in hotelrs[1]:
                intarr.append(hotelrs[0])
            elif 'float' in hotelrs[1]:
                floatarr.append(hotelrs[0])
            elif 'date' in hotelrs[1]:
                timearr.append(hotelrs[0])

    # ***************************** 2.读取表格字段 ***************************
    # Open the workbook and define the worksheet
    book = xlrd.open_workbook(fullname)
    sheet = book.sheet_by_name("Sheet1")
    # 忽略表格的列指标
    ignoindex = []
    # 缓冲sql键数组
    bufkeyar = []
    # 该有没在表里的数组
    shouldhavar = []
    # 链接符
    connecstr = ','
    # server_summary 一级 二级 三级 ipadd 联系人 位置
    summary_posi = [-1, -1, -1, -1, -1]
    # server_summary 内容自定义
    summary_cont = []
    # 2.1.获取表格全字段,排除表格指定禁止书写的字段 or 排除表格自动禁止书写的字段，即 auto_increment
    for r in range(0, sheet.ncols):
        if sheet.cell(0, r).value == 'company':
            summary_posi[0] = r
        elif sheet.cell(0, r).value == 'role':
            summary_posi[1] = r
        elif sheet.cell(0, r).value == 'service_name':
            summary_posi[2] = r
        elif sheet.cell(0, r).value == 'ipaddress':
            summary_posi[3] = r
        elif sheet.cell(0, r).value == 'contact':
            summary_posi[4] = r
        if sheet.cell(0, r).value in ignarr or sheet.cell(0, r).value in increasarr:
            ignoindex.append(r)
        else:
            bufkeyar.append('`' + sheet.cell(0, r).value + '`')
    for r in range(0, len(addarr[0])):
        if addarr[0][r] not in [str(i) for i in sheet.row_values(0)]:
            bufkeyar.append('`' + addarr[0][r] + '`')
            shouldhavar.append(r)
    keystr = connecstr.join(bufkeyar)
    # ***************************** 3.表格数据格式 ***************************
    # 3.生成values
    iponeindex = -1
    try:
        bufvaluesarr = []
        for s in range(0, sheet.ncols):
            if sheet.cell(0, s).value == "ipone":
                iponeindex = s
        for r in range(1, sheet.nrows):
            # 4.生成value
            bufvaluearr = []
            for s in range(0, sheet.ncols):
                if s not in ignoindex:
                    # 3.1.根据字段类型自动转化格式
                    if sheet.cell(0, s).value in addarr[0]:
                        bufvaluearr.append(addarr[1][addarr[0].index(sheet.cell(0, s).value)])
                    elif str(sheet.cell(r, s).value).rstrip() == "":
                        bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in intarr:
                        try:
                            float(sheet.cell(r, s).value)
                            bufvaluearr.append(str(int(sheet.cell(r, s).value)))
                        except ValueError:
                            bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in floatarr:
                        try:
                            float(sheet.cell(r, s).value)
                            bufvaluearr.append(str(float(sheet.cell(r, s).value)))
                        except ValueError:
                            bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in timearr:
                        try:
                            bufvaluearr.append(
                                "\"" + str(xlrd.xldate.xldate_as_datetime(sheet.cell(r, s).value, 0)) + "\"")
                        except ValueError:
                            bufvaluearr.append("null")
                    else:
                        bufvaluearr.append("\"" + str(sheet.cell(r, s).value) + "\"")
            for s in range(0, len(shouldhavar)):
                bufvaluearr.append(addarr[1][shouldhavar[s]])
            bufvaluesarr.append('(' + connecstr.join(bufvaluearr) + ')')

        # 3.1.5 server_summary 处理
        if summary_posi[0] != -1:
            # ipone的索引
            rex = re.compile(r'^[0-9][0-9.;]*[0-9]$')

            def extract_value(ori_value):
                sp_arry = []
                try:
                    sp_arry = re.split(r'[;/:]', ori_value)
                except:
                    pass
                i10 = [i for i in sp_arry if i.startswith("10.")]
                if len(i10) > 0:
                    return i10[0]
                else:
                    return ""

            for r in range(1, sheet.nrows):
                tmp_summary_cont = ["", "", "", "", ""]
                for s in range(0, sheet.ncols):
                    tmpstr = str(sheet.cell(r, s).value)
                    if s == iponeindex:
                        res = rex.match(tmpstr)
                        if res is None:
                            raise IponeException("业务ip格式问题！")
                    try:
                        indss = summary_posi.index(s)
                        tmp_summary_cont[indss] = tmpstr
                    except:
                        pass
                tmp_summary_cont.append(extract_value(tmp_summary_cont[3]))
                summary_cont.append(tmp_summary_cont)
                # 3.2.根据字段类型手动转化格式
                # ***************************** 4.sql写入 ***************************
                #     query = 'INSERT INTO `' + tablename + '` (' + keystr + ') VALUES (' + connecstr.join(bufvaluearr) + ')'
                #     print query
                # cursor.execute(query)
                # print cursor.lastrowid
        valustrs = connecstr.join(bufvaluesarr)
        # raise "fack"
        # # 3.2.根据字段类型手动转化格式
        # # ***************************** 4.sql写入 ***************************
        query = 'INSERT INTO `' + tablename + '` (' + keystr + ') VALUES ' + valustrs
        cursor.execute(query)
        # print int(connection.insert_id())
        if tablename != "asset_purchase":
            hisval = '(' + keystr + '),' + valustrs
            hisval = hisval.replace("\"", "\\\"")
            hisvalfina = "'" + device + "','" + eventype + "','" + hisval + "',now(),'" + user + "'"
            query = 'INSERT INTO `audit_history` (`deviceType`, `eventType`, `others`, `date`, `user`) VALUES (' + \
                    hisvalfina + ')'
            cursor.execute(query)
        # cursor.execute(query, values)
        # 关闭游标
        cursor.close()
        # 提交
        # database.commit()
        connection.commit()
        # 关闭数据库连接
        # database.close()
        connection.close()
        # ***************************** 5.当为server_summary时，更新表格tenant_app ***************************
        if 'server_summary' == tablename:
            # 5.1 凡是导入的都需要添加
            path = BASE_DIR + r'/server/tools/tenant_app.xls'
            workbook = xlrd.open_workbook(path)
            sheet = workbook.sheet_by_index(0)
            tenant_cols = sheet.col_values(0)  # 获取第1列内容
            new_excel = copy(workbook)
            ws = new_excel.get_sheet(0)
            lines_len = len(tenant_cols)
            for i in range(0, len(summary_cont)):
                ws.write(lines_len + i, 0, summary_cont[i][0])
                ws.write(lines_len + i, 1, summary_cont[i][1])
                ws.write(lines_len + i, 2, summary_cont[i][3])
                ws.write(lines_len + i, 3, summary_cont[i][2])
                ws.write(lines_len + i, 4, summary_cont[i][4])
                ws.write(lines_len + i, 5, summary_cont[i][5])
            new_excel.save(path)
        return True
    except IponeException as e2:
        # 如果出现了错误，那么可以回滚，就是上面的三条语句要么执行，要么都不执行
        # cursor.close()
        connection.rollback()
        connection.close()
        raise Exception("业务ip格式问题！")
        return e2
    except (Exception) as e:
        # 如果出现了错误，那么可以回滚，就是上面的三条语句要么执行，要么都不执行
        # cursor.close()
        connection.rollback()
        connection.close()
        return e


def excel2mysql_audit(fullname, tablename, ignarr, addarr, device, eventype, user):
    # 建立一个MySQL连接
    cursor = connection.cursor()
    # ***************************** 1.读取字段对象 ***************************
    # 1.1. 查询
    query = 'desc `' + tablename + '`'
    try:
        cursor.execute(query)
        desckey = cursor.fetchall()  # 读取所有
    except Exception as e:
        cursor.close()
        raise "sql error when:" + query
    # 1.2. 属性组赋值
    increasarr = []
    intarr = []
    floatarr = []
    timearr = []
    for (hotelrs) in (desckey):
        if hotelrs[3] == "PRI":
            increasarr.append(hotelrs[0])
        else:
            if 'int' in hotelrs[1]:
                intarr.append(hotelrs[0])
            elif 'float' in hotelrs[1]:
                floatarr.append(hotelrs[0])
            elif 'date' in hotelrs[1]:
                timearr.append(hotelrs[0])
    # ***************************** 2.读取表格字段 ***************************
    # Open the workbook and define the worksheet
    book = xlrd.open_workbook(fullname)
    sheet = book.sheet_by_name("Sheet1")
    # 忽略表格的列指标
    ignoindex = []
    # 缓冲sql键数组
    bufkeyar = []
    # 该有没在表里的数组
    shouldhavar = []
    # 链接符
    connecstr = ','
    # server_summary 内容自定义
    summary_cont = []
    # 2.1.获取表格全字段,排除表格指定禁止书写的字段 or 排除表格自动禁止书写的字段，即 auto_increment
    for r in range(0, sheet.ncols):
        if sheet.cell(0, r).value in ignarr or sheet.cell(0, r).value in increasarr:
            ignoindex.append(r)
        else:
            bufkeyar.append('`' + sheet.cell(0, r).value + '`')
    for r in range(0, len(addarr[0])):
        if addarr[0][r] not in [str(i) for i in sheet.row_values(0)]:
            bufkeyar.append('`' + addarr[0][r] + '`')
            shouldhavar.append(r)
    keystr = connecstr.join(bufkeyar)
    # ***************************** 3.表格数据 sql格式拼接 ***************************
    # 3.生成values
    iponeindex = -1
    try:
        bufvaluesarr = []
        # ***************************** 5.当为server_summary时，ipone 格式检查***************************
        if 'server_summary' == tablename:
            # ipone的索引
            for s in range(0, sheet.ncols):
                if sheet.cell(0, s).value == "ipone":
                    iponeindex = s
            rex = re.compile(r'^[0-9][0-9.;]*[0-9]$')
            for r in range(1, sheet.nrows):
                for s in range(0, sheet.ncols):
                    tmpstr = str(sheet.cell(r, s).value)
                    if s == iponeindex:
                        res = rex.match(tmpstr)
                        if res is None:
                            raise IponeException("业务ip格式问题！")
        for r in range(1, sheet.nrows):
            # 4.生成value
            bufvaluearr = []
            for s in range(0, sheet.ncols):
                if s not in ignoindex:
                    # 3.1.根据字段类型自动转化格式
                    if sheet.cell(0, s).value in addarr[0]:
                        bufvaluearr.append(addarr[1][addarr[0].index(sheet.cell(0, s).value)])
                    elif str(sheet.cell(r, s).value).rstrip() == "":
                        bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in intarr:
                        try:
                            float(sheet.cell(r, s).value)
                            bufvaluearr.append(str(int(sheet.cell(r, s).value)))
                        except ValueError:
                            bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in floatarr:
                        try:
                            float(sheet.cell(r, s).value)
                            bufvaluearr.append(str(float(sheet.cell(r, s).value)))
                        except ValueError:
                            bufvaluearr.append("null")
                    elif sheet.cell(0, s).value in timearr:
                        try:
                            bufvaluearr.append(
                                "\"" + str(xlrd.xldate.xldate_as_datetime(sheet.cell(r, s).value, 0)) + "\"")
                        except ValueError:
                            bufvaluearr.append("null")
                    else:
                        bufvaluearr.append("\"" + str(sheet.cell(r, s).value) + "\"")
            for s in range(0, len(shouldhavar)):
                bufvaluearr.append(addarr[1][shouldhavar[s]])
            bufvaluesarr.append('(' + connecstr.join(bufvaluearr) + ')')

        valustrs = connecstr.join(bufvaluesarr)
        # # 3.2.根据字段类型手动转化格式
        # # ***************************** 4.sql写入 ***************************
        query = 'INSERT INTO `' + tablename + '` (' + keystr + ') VALUES ' + valustrs
        cursor.execute(query)
        # print int(connection.insert_id())
        if tablename != "asset_purchase":
            hisval = '(' + keystr + '),' + valustrs
            hisval = hisval.replace("\"", "\\\"")
            hisvalfina = "'" + device + "','" + eventype + "','" + hisval + "',now(),'" + user + "'"
            query = 'INSERT INTO `audit_history` (`deviceType`, `eventType`, `others`, `date`, `user`) VALUES (' + \
                    hisvalfina + ')'
            cursor.execute(query)
        # cursor.execute(query, values)
        # 关闭游标
        cursor.close()
        # 提交
        # database.commit()
        connection.commit()
        # 关闭数据库连接
        # database.close()
        connection.close()
        return True
    except IponeException as e2:
        # 如果出现了错误，那么可以回滚，就是上面的三条语句要么执行，要么都不执行
        # cursor.close()
        connection.rollback()
        connection.close()
        raise Exception("业务ip格式问题！")
        return e2
    except (Exception) as e:
        # 如果出现了错误，那么可以回滚，就是上面的三条语句要么执行，要么都不执行
        # cursor.close()
        connection.rollback()
        connection.close()
        return e


# 如果该文件不是被import,则执行下面代码。
if __name__ == '__main__':
    # 定义一个字典，key为对应的数据类型也用作excel命名，value为查询语句
    db_dict = {'test': 'select * from asset_assets'}
    # 遍历字典每个元素的key和value。
    for k, v in db_dict.items():
        # 用字典的每个key和value调用write_data_to_excel函数。
        write_data_to_excel('/', k, v)
