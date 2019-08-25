# coding:utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2019 yutiansut/QUANTAXIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import sys

import pymongo

from QUANTAXIS.QAFetch.QAfinancial import (download_financialzip, parse_all,
                                           parse_filelist)
from QUANTAXIS.QASetting.QALocalize import (cache_path, download_path, qa_path,
                                            setting_path)
from QUANTAXIS.QAUtil import DATABASE, QA_util_date_int2str
from QUANTAXIS.QAUtil.QASql import ASCENDING, DESCENDING
from QUANTAXIS.QAUtil.QATransform import QA_util_to_json_from_pandas
import datetime


def QA_SU_save_financial_files():
    """本地存储financialdata
    """
    download_financialzip()
    coll = DATABASE.financial
    coll.create_index(
        [("code", ASCENDING), ("report_date", ASCENDING)], unique=True)
    for item in os.listdir(download_path):
        if item[0:4] != 'gpcw':
            print(
                "file ", item, " is not start with gpcw , seems not a financial file , ignore!")
            continue

        date = int(item.split('.')[0][-8:])
        print('QUANTAXIS NOW SAVING {}'.format(date))
        if coll.find({'report_date': date}).count() < 3600:

            print(coll.find({'report_date': date}).count())
            data = QA_util_to_json_from_pandas(parse_filelist([item]).reset_index(
            ).drop_duplicates(subset=['code', 'report_date']).sort_index())
            # data["crawl_date"] = str(datetime.date.today())
            try:
                coll.insert_many(data, ordered=False)

            except Exception as e:
                if isinstance(e, MemoryError):
                    coll.insert_many(data, ordered=True)
                elif isinstance(e, pymongo.bulk.BulkWriteError):
                    pass
        else:
            print('ALL READY IN DATABASE')

    print('SUCCESSFULLY SAVE/UPDATE FINANCIAL DATA')
