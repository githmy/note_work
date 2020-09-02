from __future__ import print_function
# pip install easyquotation
import easyquotation
# 数据获取方式 集思录
# 棣华日利 511880
# http://hq.sinajs.cn/list=sh601006
# http://qt.gtimg.cn/q=sz000858

def main():
    # 选择行情
    quotation = easyquotation.use('sina') # 新浪 ['sina'] 腾讯 ['tencent', 'qq']
    # 获取所有股票行情
    quotation.market_snapshot(prefix=True) # prefix 参数指定返回的行情字典中的股票代码 key 是否带 sz/sh 前缀
    # 单只股票
    quotation.real('162411') # 支持直接指定前缀，如 'sh000001'
    # 多只股票
    quotation.stocks(['000001', '162411'])
    # 同时获取指数和行情
    quotation.stocks(['sh000001', 'sz000001'], prefix=True)
    # 更新股票代码
    easyquotation.update_stock_codes()
    # 选择 jsl（集思路） 行情
    quotation = easyquotation.use('jsl')  # ['jsl']
    # 获取分级基金信息
    quotation.funda()  # 参数可选择利率、折价率、交易量、有无下折、是否永续来过滤
    quotation.fundb()  # 参数如上
    # 分级基金套利接口
    quotation.fundarb(jsl_username, jsl_password, avolume=100, bvolume=100, ptype='price')
    # 指数ETF查询接口 TIP : 尚未包含黄金ETF和货币ETF
    # 集思录ETF源网页 https://www.jisilu.cn/data/etf/#index
    quotation.etfindex(index_id="", min_volume=0, max_discount=None, min_discount=None)
    # 分数图 腾讯分时图地址 http://data.gtimg.cn/flashdata/hushen/minute/sz000001.js
    quotation = easyquotation.use("timekline")
    data = quotation.real(['603828'], prefix=True)
    # 港股日k线图 腾讯日k线图 http://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get?_var=kline_dayqfq&param=hk00700,day,,,350,qfq&r=0.7773272375526847
    quotation  = easyquotation.use("daykline")
    data = quotation.real(['00001','00700'])
    print(data)
    # 腾讯港股时时行情 腾讯控股时时行情 http://sqt.gtimg.cn/utf8/q=r_hk00700
    quotation = easyquotation.use("hkquote")
    data = quotation.real(['00001','00700'])
    print(data)

if __name__ == '__main__':
    main()
