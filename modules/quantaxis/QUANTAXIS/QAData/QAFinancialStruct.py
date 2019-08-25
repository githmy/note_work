# coding :utf-8
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
"""
财务指标结构

"""
import pandas as pd

from QUANTAXIS.QAData.financial_mean import financial_dict


class QA_DataStruct_Financial():

    def __init__(self, data):
        self.data = data
        # keys for CN, values for EN
        self.colunms_en = list(financial_dict.values())
        self.colunms_cn = list(financial_dict.keys())

    def __repr__(self):
        return '< QA_DataStruct_Financial >'

    def get_report_by_date(self, code, date):
        return self.data.loc[pd.Timestamp(date), code]

    def get_key(self, code, reportdate, key):
        if isinstance(reportdate, list):
            return self.data.loc[(
                slice(
                    pd.Timestamp(reportdate[0]),
                    pd.Timestamp(reportdate[-1])
                ),
                code
            ),
                                 key]
        else:
            return self.data.loc[(pd.Timestamp(reportdate), code), key]
