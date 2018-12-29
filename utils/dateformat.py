# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from datetime import datetime, date
from datetime import timedelta
import calendar


def get_endweek_of_lastmonth(date):
    """
    获取上个月第一天的日期，然后加21天就是22号的日期
    :return: 返回日期
    """
    # today = datetime.today()
    today = date
    year = today.year
    month = today.month
    if month == 1:
        month = 12
        year -= 1
    else:
        month -= 1
    res = datetime(year, month, 1) + timedelta(days=21)
    return res.strftime('%Y-%m-%d %X')


#############################

def get_endweek_of_nextmonth(date):
    """
    获取下个月的22号的日期
    :return: 返回日期
    """
    # today = datetime.today()
    today = date
    year = today.year
    month = today.month
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    res = datetime(year, month, 1) + timedelta(days=21)
    return res.strftime('%Y-%m-%d %X')


def get_1stweek_of_nextmonth(date):
    # today = datetime.today()
    today = date
    year = today.year
    month = today.month
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    res = datetime(year, month, 1) + timedelta(days=6)
    return res.strftime('%Y-%m-%d %X')


def get_1stDay_Of_thismonth(date):
    # d = datetime.now()
    d = date
    # c = calendar.Calendar()

    year = d.year
    month = d.month

    if month == 1:
        month = 12
        year -= 1
    else:
        month -= 1
    days = calendar.monthrange(year, month)[1]
    return (datetime(year, month, 1) + timedelta(days=days)).strftime('%Y-%m-%d %X')


def out_month_week(date):
    return [get_endweek_of_lastmonth(date), get_1stweek_of_nextmonth(date)]


def date2str(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, date):
        return obj.strftime("%Y-%m-%d")
    else:
        return obj
