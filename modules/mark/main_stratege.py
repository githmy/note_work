"""
# 参考代码
https://github.com/msincenselee
git clone git://github.com/msincenselee/vnpy.git
git clone https://gitee.com/vnpy2/vnpy.git
vn.trader/NoUIMain.py
 
第三方平台 程序化实盘：
  本地：
    金字塔，MC8s，TB交易开拓者，WH8文华赢智, TS, MQ/QQ, MT4
  云端：
    聚宽ycjq.95358.com，优矿Uqer.io，米筐RiceQuant,Apama,BotVS
  SDK/量化API：
    万得Wind，东财Choice,掘金量化gmsdk
  开源框架：
    PyCTP, vnpy, quicklib, zipline
  自己编写：
    CTP,Webservice,easytrader 
期货交易所：
  上海期货交易所：黄金 白银 铜 铝 锌 橡胶 燃油 螺纹钢 线材 石油 沥青 铅
  郑州商品交易所：小麦 棉花 白糖 PTA 绿豆 红小豆 花生 油菜籽 玻璃
  大连商品交易所：大豆 豆粕 玉米 豆油 啤麦 焦炭 棕榈油 聚氯乙烯
  中国金融期货交易所：沪深300指数 5年期国债期货
  保证金交易
  主力合约，移仓换月
模型：
  原始xgboost
  超额信息+深度学习
  多因子模型：
    因子尽量少，减少复杂度，减少过拟合
    因子尽量独立，不然冗余复杂
    时间尽量长，跨越牛熊的长期指标。
  调参： 
    观察窗口60 30分钟 周线，调仓频率，加择时信号，止盈止损，回复尺度，大盘背景。
    训练测试数据不能过少，参数不能过多，去掉最好的几次，参数敏感性分析，参数相关性分析
1. 数据收集
2. 高性能计算
3. 软件开发
  K线最后时刻的动态收盘价。
4. 特征分析
5. 执行仿真
6. 回测

多因子合成：估值因子，市值因子 F
R1 = B11 * F1 + B12 * F2 + ... + B1k * Fk + e1
R2 = B21 * F1 + B22 * F2 + ... + B2k * Fk + e2
R3 = B31 * F1 + B32 * F2 + ... + B3k * Fk + e3
行业因子：采掘，钢铁，医药，房地产，食品行业因子
风格风险：市场B因子，规模因子，非线性规模因子，动量因子PR1YR，成交量因子HMC，波动率因子CMP，流动性因子，盈利因子，估值因子，成长因子，杠杆因子
基本面风险：规模因子SMB，估值因子HML，成长因子，质量因子，盈利能力因子RMW，投资风格因子CMA
技术面风险：动量因子，成交量因子，波动率因子，效率因子EMI，强度因子SMW, 技术情绪
宏观面风险：真实增长因子GMC，工业增长因子IMC，货币需求因子2M1，货币供应因子2ME
smart beta 6因子: 市场，规模，价值，动量，质量，波动。
因子：
   波动标准差， TR=max(h2-l2,|h2-c1|,|c1-l2|) ATR = 窗口内每天TR的平均值, 均线， 涨幅，MACD
   DIF=12日EMA-26日EMA
   DEA是DIF的9日平均值
   红柱与绿柱：（DIF-DEA）*2即是柱子的数值
   均线型压力支撑：MA,EXPMA,BBI
   趋势型启动延续转折：MACD,SAR,ASI,DMI
   摆动型震荡超买超卖转折突破：KDJ,RSI,CCI,WR,BOLL
   能量型涨跌力度，量在价前量价配合：OBV,VOL,VR
   ROE: 反应企业质量。小市值，PE0-30,市值10-100亿，ROE 10-40%，再平衡10天。
   月度累计换手率/月度日收益波动率，按因子排序分5组，每组是一个股票池。换手率小说明看涨，波动率大说明有风险。是忠诚度指标。
   多因素模型CAPM：rpt=alpha+betam*(rmp-rft)+betas*SMB+betav*HML+betac*CMA+betar*RMW+epsilon。   betam市场风险 betas规模风险 betav价值风险 betac投资风险 betar盈利能力风险
   乖离率：价差变动
股票池：
   品种：股票 ETF 商品期货 股指期货 期权 债券 外汇 
   行业轮动，相似组： (能源，材料，通讯) 此消彼长 (基础消费，可选消费，医药)。 前两组叠加关系：(工业，金融，信息，公用事业)
   行业板块占比均衡，成分股筛选和权重
   收益 top bottom 10%@时段，分别做多做空。
   基本面排劣：剔除最小市值10%的，剔除st，剔除PE<0,PE>100的。
   技术面超跌反弹：25日，跌幅前10%的。
   再平衡周期25天
   平均分配资金

美女策略
屌丝 x 1-x
美女 y 1-y
屌丝期望 E=5xy+(1-x)(1-y)-3x(1-y)-3y(1-x)=12xy-4(x+y)+1<0
y(12x-4)<4x-1; x,y[0,1]
1/4<y<3/8

大小周期
Aα: 指标排序，选股超额收益，组合做空股指，定期调仓。满仓，股指对冲。
  股票多头 sharp>1, 最好>1.5
  α策略最大回撤<5%
  商品期货sharp>1.7
  跨期套利，多多配对，多空配对交，关联性强的品种。如黑色 化工 金属 油粕。
  相同标的的 现货 期货 期权。基本面相关的关联资产。统计方式套利。市场中性beta, 绝对alpha。
  期权平价套利：一条腿是期货，一条腿是期权
  期权箱体套利：
    2条腿都是期权。k1<k2, 牛市买入k1看涨，卖出k2看跌。
    自条腿的组合可任意套利。无风险利润=箱体到期值-权利金支出
  波动套利：
    转债期权 比标的股票低估时。买入期权，同时做空股票。随后调整做空比例达到市场中性，赚取利息和期权上升值。风险小于股票或债券。
Bα: 战法，择时，止损。
   什么情况回调大，周期和品种的选择 天然橡胶RU、螺纹钢RB、热轧卷板HC、石油沥青BU
   趋势跟随 CTA. 入市条件过滤。盈利性退出，亏损性退出。回撤N倍ATR止盈止损。15分钟bar下穿MA30。底背离再次下跌，寻找更及时的出场条件。
      连续6天收盘跌，然后4天同位价比前一天高，买入。
      连续6天收盘涨，然后4天同位价比前一天低，卖出。
      成交量增长没有使价格大幅增长，是趋势停止信号。
      赔率：股票型 >2, 商品期货>5, 心理因子>3
   hans123:   
      上轨:开盘30分钟最高价;
      下轨:开盘30分钟最低价;
      上突破，做多。下突破，做空。
      收盘平仓。
   均线法：
      15分钟bar, MA5上穿MA30，买入
      15分钟bar, MACD底背离，买入
      5分钟bar, MACD连续底背离，买入
   海龟：
       均仓 或 按波动率建仓。
      （波动 ATR * 一手的数量）* 100倍。 即 建仓杠杆 = 1/(波动幅度 * 100)。 即使当日波动幅度为 ATR, 损失不超过1%
       每涨N/2 增加一个单位，同一K线可多次加仓，买入K线内可以不止损。跌超过2*N，清仓。
       胜率不高，赔率大。
   圆形底，圆形顶：反转信号。    
   什么情况大趋势：
       事件驱动：高送转，重大资产重组，产业优惠
       技术分析：上升通道，压力突破
       反应过度：翻转后小海龟
Xα: 风险因素的未来走势，优化承担风险的能力组合。
均值回复：
  https://www.joinquant.com/post/12940
  https://www.joinquant.com/view/community/detail/8fd843f8ad2e5c9432fefbaa8dfe8825
  统计套利，期权波动。
  股票配对，年化收益 top25 之后 指数下跌。回撤0.06,0.1之间。
  均线 分位数 卡尔曼滤波 作为中轨。主动止盈，中线平仓。时间止损，超过一定的时间回复概率降低。
  市场反转：过去1个月表现最差最好的N只股票，多空组合。再平衡1个月。
  高胜率>60%
低风险套利：
  ETF折溢价套利，分级基金申赎套利，期货跨期套利，期货现货套利
  正套开仓 long leg1 short leg2；正套平仓sell leg1 cover leg2；反套开仓 short leg1 long leg2；反套平仓cover leg1 sell leg2
  相关系数越高，价格跟随性越强。协整性越高，价差越稳定。跨品种一般大于跨期的价差。期货的保证金和交割月，是跨期套利价差经常不回归。跨市场收盘时间不一致。
方式：α, 跨市场统计套利，CTA, 海龟, 高频, 算法交易降低成本
加仓：20日波动幅度 N, 突破后建仓 1 单位，顺势0.5N 加仓 1 单位，4 单位为上限。较上次建加仓信号下降2N，平仓。 
评价：预测目标，模型设定，条件变量，胜率，可靠率，盈亏时间比，交易成本，交易频率，单词头寸，资金规模

策略规划：
  基本面，宏观面，投资组合管理
  单边趋势，套利套保，对冲交易
  行情判定，价差交易，强弱及波动率
  收益最大化，高风险厌恶，动态平衡风险收益
  个人能力，人际关系，团队合作
  资金配置，多策略对比，多策略组合，多品种测试，环境测试
策略指标：
  年化收益 总收益 基准收益 alpha beta 胜率 盈亏比(赔率) 最大回撤 心理因子(舒适度) 信息率(投资经理的能力)
  资金：
    单日可用资金上限，单个标的可用资金上限，总可用资金上限。最大持仓上限，风险分配。
  投资比例:
    CPR公式：P头寸=C资金/R风险波动率
    总资金100w,总风险1%，单标的风险5%。标的40元每股，标的数量=100w*1%/(40*5%)=5000股。消耗金额=40*5000=20w
    总资金20w,初始风险2% 4000 持续风险3%，400元建仓4手160000 止损390 暴露风险4000,涨到440，止损410，风险暴露12000。允许风险爆露率（20W+40*400）*30%=6480.平仓两手。
    标的价格差别，风险偏好类型，总风险控制，标的的波动差别
  交易成本：
    手续费 佣金; 融资融券利率; 保证金比率; 做多（融资）交易额X,保证金X; 做空（融券）交易额X,保证金1.5X; 冲击滑点;
  标的：
    投资组合的容量。期货 股票 融资融券
  盈利方面：
    总利润，总收益率，年化收益率，日均盈利，单笔盈利, 收益均线。 参照 巴菲特 21~27%， 西蒙斯 30+% 66%。
  风险方面：
    年化波动率，年下行风险，最大回撤，最大回撤率， 多空头排列
  综合：
    夏普比-收益波动比，卡玛比率-收益回撤比，索提诺比率-收益下行比
  捡钱策略：
    错单，涨跌停策略
  截断亏损，分散化，头寸退出，小亏大赚。
  置信度：
    交易次数，95%置信区间 [X-sigma/sqrt{n}* Za2, X+sigma/sqrt{n}* Za2]
  股指费用：
    合约价值(锚定指数)，保证金30%，手续费低，升水贴水，现金交割，股指+股票+ETF混合套利
  多空统计：
  时间统计：
高频：抢cta的利润。投机者，白噪声
中频：移动平均线消除高频，延迟大。投资者，斜波信号
低频：。套保者，阶越信号

数据：
  获取：
    LTS CTP XTP 门槛高
    新浪L1数据，腾讯，网易
    autoit自动化操作交易软件
    web网上营业厅
    股票：
      新浪L1数据，腾讯，网易
    期货：
      通达信
    分级基金：
      集思录
    新股数据：
      集思录
  类型：
    行情，财务，经济，新闻，网络
  处理过程：
    解析，抓取，清洗对比，标准化
  存储推送：
    数据库架构，存储方式，调用寻址，压缩推送
  金融库：
    基础数据
    公司数据
    股票数据
      A股基本信息库
      B股基本信息库
      港股基本信息库
      证券停复盘信息表
      股票名称变更表
    基金数据
    行业数据
    行情数据
    指数数据
通联
https://m.datayes.com/
from ctaAlgo.datayesCleint import DatayesClient
    
path = "api/market/getMktFutd.json"
params = {}
params['ticker'] = 'rb1705'
params['beginDate'] = '20170101'
params['endDate'] = '20170315'
    
datayes_client = DatayesClient()
result = datayes_client.downloadData(path, params)
     
未来函数：
  提前知道收盘价，收盘撮合，财务公告，基金公告日期，高频成本。

A股特质：
  熊市波动大，震荡波动小，上升波动稳定，牛末波动大。
系统评估：
  疑问：
    系统没问题，运气差。或系统有问题，但运气好。极端情况。隔夜头寸。
    回测不足200次，最少70次。停牌不能成交。涨跌停st不能买卖。日内不能以开盘价成交。收盘价不确定。相同的过程，每次结果不一样。
    买卖信号是否完成了下单指令：
    下单结果能否及时获取：
    策略能否及时维护账户状态：
    异步过程出现问题，如何修复：
      
  准入条件：
    指令是否正确，最小化单笔资金交易量，仿真模拟测试，实盘监控。
  数据：
    收集，校验，比对，清洗，对齐，管理配置，存储，推送
  策略开发：
    函数库
  风控模块：
    下单：
      下单类型，配对交易，组合交易，大单分割，及时高效
    风险：
      下单监控，仓位监控，账户监控，止损设置
    异常：
      数据异常 概率小致命，网络异常 备用网络，行情接受处理异常 概率最大的bug.
      服务器瘫痪 配置合理吗 压力测试，延迟，下单异常，持仓异常，策略异常，人为干预方式。
  分析统计:
    分析维度分类：
      量化型，基本面型，技术型，判断型。量化技术型(西蒙斯)，基本面判断型(巴菲特)，技术判断型(图表，技术分析)
    基础模型，回测平台，统计检验，数据样本检验，协整分析，样本内外测试，过拟合检验，鲁棒性分析
    可视化
    大背景期权操作方法：
      看涨，看跌，强势(cta,套利，高频)，弱势(中长线价值投资)。看涨强势 看涨买方，看跌强势 看跌买方，看涨弱势 看跌卖方，看跌弱势 看涨卖方。
  调试修正:
    计算层面维度选择，tick/min,策略层面逻辑，托管机房，网络架构。一般CTP15步骤大于3ms，高速5步骤小于300us。
  警告：
    回撤1%，一级警告，复盘检查，是否可以继续执行原有策略。
    回撤4%，二级警告，暂停自动交易，风控判断是否可以继续。
    回撤8%，三级警告，降仓至25%以下，风控判断是否停止。
    净值将至0.85，风控判断是否清盘。
    总仓位不超过90%，隔夜不超过80%
    单支股票开仓位 2% 单方向。
    单支股票最大亏损不超过 15%
    单支股票最大不超过10%流通市值。
    单支股票头寸 不超过2% 的流通市值。
    单支股票每日交易额不超过当日成交量的10%
  监控：
    净值，持仓比例，警戒线，清盘线
  人工干预：
     股灾，崩盘，黑天鹅，外盘剧烈波动，重大消息。预期动作不符。网络异常，数据异常，接口异常，服务器异常。
  期货：
    回测移仓换月，指数合约，主力连续， 拼接，映射。杠杆保证金，隔夜跳空，持仓不过节。跨品种套利。
  清盘程序：
    过于复杂，需要人工参与
  模拟误差：
    成交时机 延迟，滑点，撤单。挂单部分成交，冲击成本。涨跌停，交易状态。
  公式：
    头寸数量 = 总风险 / 每股风险  。  单风险如何确定，总风险 = 个人的容忍度， 基金的强制平仓线。
    总风险 = 总资产 * 总风险比率
    单头寸风险 = 标的价格 * 单头寸风险比率
    头寸数量 = (总资产 * 总风险比率) / (标的价格 * 单头寸风险比率)
    头寸数量 = (总风险 / N) / 单头寸风险
    头寸数量 = 账户总风险 / (ATR * 最小交易单位)
    一次交易的资金比率 = 总风险比率 / 单头寸风险比率
    总风险3%， 单股风险6%，一次交易的资金比率为50%
    
大背景：
    经济周期，经济指标，基础利率，cpi,ppi
    财报模块，分析师研报模块，大宗商品板块，产业链，
    市场宏观数据，市场微观数据，流动性分析，机构结构
公司财务指标：
  运营效率：
    总资产周转率，固定资产周转率，存货周转率，应收账款周转率
  前景：
    营业收入增长率，净利润增长率，总资产增长率，净资产增长率
  因子评分：
    风险得分*10%+运营效率得分*25%+盈利能力及质量得分*40%+前景与估值得分*25%
  财务评级：
    10% A, 10~30% B, 30~60% C, 60~80% D, 80~100% E
  资产负债率：
     40~60% 40, 30~40% 30, 0~30% 20, 60~80% 10, 80~100% 5
  流动比率：
    10% 60, 10~30% 50, 30~60% 40, 60~80% 25, 80~100% 10
  例如：
    资产负债率：35%          30
    流动比率：排名35%        40
    得分：(30+40)*0.1=7分
    总资产周转率：排名25%    40
    存货周转率：排名40%      30
    得分：(40+30)*0.25=17.5分
    净资产收益率：排名50%        10
    每股收益：   排名32%        15
    销售收现比：    大于1       20
    净利润现金含量比率：排名46%  15
    得分：(10+15+20+15)*0.4=24分
    营业收入增长率：排名32%      30
    净利润增长率：  排名70%      20
    得分：(30+20)*0.25=12.5分
    总得分：7+17.5+24+12.5=61， 排名35%， C级
    
    
"""

import sys, os
import datetime
import json
import itertools
from modules.portfolio import Portfolio
from modules.event import *
from modules.datahandle import CSVDataHandler, CSVAppendDataHandler, LoadCSVHandler
from modules.strategys import MovingAverageCrossStrategy, MultiCrossStrategy, MlaStrategy
from modules.executions import SimulatedExecutionHandler
from modules.backtests import Backtest, LoadBacktest
from utils.log_tool import *
import pyttsx3
import tushare as ts


def choice_list(plate_list):
    # 行业分类
    tt = ts.get_industry_classified()
    infos = tt['code'].groupby([tt['c_name']]).apply(list)
    industjson = json.loads(infos.to_json(orient='index', force_ascii=False))
    # 中小板分类
    tt = ts.get_sme_classified()
    allsmartlist = []
    for plate1 in plate_list:
        allsmartlist.append(list(set(tt["code"]).intersection(set(industjson[plate1]))))
    # allsmartlist = list(set(itertools.chain(*allsmartlist)))
    allsmartlist = [i1 + "_D" for i1 in set(itertools.chain(*allsmartlist))]
    # smartlist = list(set(tt["code"]).intersection(set(industjson["电子信息"])))
    # # 概念分类
    # tt = ts.get_concept_classified()
    # infos = tt['code'].groupby([tt['c_name']]).apply(to_list)
    # conceptjson = json.loads(infos.to_json(orient='index', force_ascii=False))
    # # 小轻，突破边缘的信息类
    # nearbreaklist = list(set(smartlist).intersection(set(conceptjson["智能机器"])))
    # print(conceptjson["智能机器"])
    # print(nearbreaklist)
    return allsmartlist


class Acount(object):
    def __init__(self, config):
        self.account = config["account"]
        self.func_type = config["data_ori"]["func_type"]
        self.test_type = config["back_test"]["test_type"]
        self.start_train = config["back_test"]["start_train"]
        self.end_train = config["back_test"]["end_train"]
        self.start_predict = config["back_test"]["start_predict"]
        self.end_predict = config["back_test"]["end_predict"]
        self.initial_capital = config["back_test"]["initial_capital"]
        self.heartbeat = config["back_test"]["heartbeat"]
        self.get_startdate = config["data_ori"]["get_startdate"]
        self.date_range = config["data_ori"]["date_range"]
        self.data_type = config["data_ori"]["data_type"]
        self.bband_list = config["data_ori"]["bband_list"]
        self.uband_list = config["data_ori"]["uband_list"]
        self.split = config["data_ori"]["split"]
        self.newdata = config["data_ori"]["newdata"]
        self.csv_dir = config["data_ori"]["csv_dir"]
        self.plate_list = config["data_ori"]["plate_list"]
        self.symbol_list = config["data_ori"]["symbol_list"]
        self.exclude_list = config["data_ori"]["exclude_list"]
        self.ave_list = config["data_ori"]["ave_list"]
        self.bband_list = config["data_ori"]["bband_list"]
        self.strategy_config = config["strategy_config"]
        self.portfolio_name = config["portfolio"]["portfolio_name"]
        self.email_list = config["assist_option"]["email_list"]
        self.policy_config = config["policy_config"]
        self.model_paras = config["model_paras"]
        try:
            self.showconfig = config["showconfig"]
        except Exception as e:
            pass
        # 生成标准参数
        self._gene_stand_paras()

    def _gene_stand_paras(self):
        pass

    def _get_train_list(self):
        flist = []
        for root, dirs, files in os.walk(data_path, topdown=True):
            flist = [i1.replace(".csv", "") for i1 in files if i1.endswith("_D.csv")]
            break
        # flist = flist[0:len(flist) // 2]
        return flist

    def _pattern_generate(self):
        # 1. 判断加载模型
        backtest = None
        if self.test_type == "实盘":
            pass
        elif self.test_type == "模拟":  # 已有数据模式
            if self.data_type == "实盘demo":  # 已有数据，动态模拟, 原始例子
                self.symbol_list = [i1 for i1 in self.symbol_list if i1 not in self.exclude_list]
                backtest = Backtest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list, self.uband_list,
                    CSVDataHandler, SimulatedExecutionHandler, Portfolio, MovingAverageCrossStrategy)
            elif self.func_type == "网络获取数据":  # 已有数据，统计强化学习
                self.symbol_list = [i1 for i1 in self._get_train_list() if i1 not in self.exclude_list]
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    None, self.symbol_list, self.ave_list, self.bband_list, self.uband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy,
                    split=0.8, newdata=self.newdata, date_range=self.date_range, assistant=self.email_list,
                    model_paras=self.model_paras)
            elif self.data_type == "实盘":  # 已有数据，动态模拟, 未完善
                self.symbol_list = [i1 for i1 in self.symbol_list if i1 not in self.exclude_list]
                backtest = Backtest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list, self.uband_list,
                    CSVAppendDataHandler, SimulatedExecutionHandler, Portfolio, MultiCrossStrategy)
            elif self.data_type == "symbol_train_type":  # 已有数据，直观统计
                self.symbol_list = [i1 for i1 in self.symbol_list if i1 not in self.exclude_list]
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list, self.uband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy,
                    split=0.8, newdata=self.newdata, date_range=self.date_range, assistant=self.email_list,
                    model_paras=self.model_paras)
            elif self.data_type == "general_train_type":  # 已有数据，直观统计
                self.symbol_list = [i1 for i1 in self._get_train_list() if i1 not in self.exclude_list]
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list, self.uband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy,
                    split=0.8, newdata=self.newdata, date_range=self.date_range, assistant=self.email_list,
                    model_paras=self.model_paras)
            elif self.data_type == "plate_train_type":  # 已有数据，直观统计
                self.symbol_list = [i1 for i1 in choice_list(self.plate_list) if i1 not in self.exclude_list]
                backtest = LoadBacktest(
                    self.initial_capital, self.heartbeat, self.start_predict,
                    self.csv_dir, self.symbol_list, self.ave_list, self.bband_list, self.uband_list,
                    LoadCSVHandler, SimulatedExecutionHandler, Portfolio, MlaStrategy,
                    split=0.8, newdata=self.newdata, date_range=self.date_range, assistant=self.email_list,
                    model_paras=self.model_paras)
            else:
                raise Exception("error data_type 只允许：实盘demo, 实盘, 模拟, 网络")
        else:
            raise Exception("error test type.")
        return backtest

    def __call__(self, *args, **kwargs):
        # 1. 判断加载模型
        backtest = self._pattern_generate()
        # 2. 判断执行功能
        if self.func_type == "网络获取数据":
            backtest.get_data()
        elif self.func_type == "train":
            backtest.train()
        elif self.func_type == "backtest":
            backtest.simulate_trading(self.policy_config, self.strategy_config, get_startdate=self.get_startdate)
        elif self.func_type == "lastday":
            backtest.simulate_lastday(self.policy_config, self.showconfig, get_startdate=self.get_startdate)
        else:
            raise Exception("func_type 只能是 train, backtest, lastday")


def main(paralist):
    logger.info(paralist)
    account_list = [
        {
            "account": 3,
            "desc": "tushare,离线测试",
            "back_test": {
                "test_type": "模拟",
                "start_train": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "end_train": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "start_predict": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "end_predict": datetime.datetime(1990, 1, 1, 0, 0, 0),
                "heartbeat": 0.0,
                "initial_capital": 10000.0,
            },
            "data_ori": {
                "split": 0.8,
                # 不使用生成的特征数据
                # "newdata": 1,
                "newdata": 0,
                # "func_type": "网络获取数据",
                "func_type": "train",
                # "func_type": "backtest",
                # "func_type": "lastday",
                "data_type": "general_train_type",
                # "data_type": "plate_train_type",
                # "data_type": "symbol_train_type",
                "date_range": [0, None],
                # "date_range": [-4, None],
                # "date_range": [-2, None],
                "get_startdate": "2019-10-15 00:00:00",
                # get_startdate 为 None 不更新数据
                # "get_startdate": None,
                # "data_type": "实盘",
                "csv_dir": data_path,
                "plate_list": ["电子信息"],
                "symbol_list": ["000001_D", "000002_D"],
                # "symbol_list": ["000002_D"],
                "ave_list": [1, 3, 5, 11, 19, 37, 67],
                # "bband_list": [1],
                # "bband_list": [2],
                # "bband_list": [3],
                # "bband_list": [4],
                # "bband_list": [5],
                # "bband_list": [6],
                # "bband_list": [7],
                # "bband_list": [19],
                # "bband_list": [37],
                # "bband_list": [1, 5],
                "bband_list": [1, 2, 3, 4, 5, 6, 7, 19, 37],
                # "bband_list": [5, 19],
                # "bband_list": [1, 5, 19],
                # "bband_list": [1, 5, 19, 37],
                # "bband_list": [5, 19, 37],
                # "exclude_list": ["000002_D"],
                # "uband_list": [1, 2, 3, 4, 5, 6, 7, 19, 37],
                "uband_list": [1, 2, 3, 4, 5, 6, 7, 19, 37],
                # "uband_list": [37],
                "exclude_list": [],
            },
            "stratgey": {
                "stratgey_name": "cross_break",
            },
            "portfolio": {
                "portfolio_name": None
            },
            "assist_option": {
                # "email_list": ["a1593572007@126.com", "619041014@qq.com"],
                # "email_list": ["a1593572007@126.com"],
                "email_list": [],
            },
            "policy_config": {
                "hand_unit": 100,
                "initial_capital": 10000.0,
                "stamp_tax_in": 0.0,
                "stamp_tax_out": 0.001,
                "commission": 5,
                "commission_rate": 0.0003,
            },
            "strategy_config": {
                "oper_num": 3,
                "thresh_low": 1.005,
                "thresh_high": 1.2,
                # "thresh_high": 1.095,
                # "move_out_percent": 0.5,
                # "move_in_percent": 0.5,
                "move_out_percent": 3.1,
                "move_in_percent": 3.1,
            },
            # fake_data显示设置
            "showconfig": {
                "range_low": -10,
                "range_high": 11,
                # "range_low": -1,
                # "range_high": 2,
                "range_eff": 0.01,
                # "mount_low": -4,
                # "mount_high": 6,
                # "mount_eff": 0.2,
                "mount_low": -1,
                "mount_high": 1,
                "mount_eff": 0.2,
            },
            # showconfig = {
            #     "range_low": -3,
            #     "range_high": 4,
            #     "range_eff": 0.01,
            #     "mount_low": -1,
            #     "mount_high": 2,
            #     "mount_eff": 0.5,
            # }
            "model_paras": {
                "env": {
                    "epsilon": 0.5,
                    "min_epsilon": 0.1,
                    "epoch": 100000,
                    "single_num": 1,
                    "max_memory": 5000,
                    "batch_size": 1024,
                    "discount": 0.8,
                    "start_date": "2013-08-26",
                    "end_date": "2025-08-25",
                    "learn_rate": 1e-3,
                    "early_stop": 10000000,
                    "sudden_death": -1.0,
                    "scope": 60,
                    "inputdim": 61,
                    "outspace": 3
                },
                "model": {
                    "retrain": 1,
                    "globalstep": 0,
                    "dropout": 0.8,
                    # "modelname": "cnn_dense_more",
                    # "modelname": "cnn_dense_lossave_more",
                    "modelname": "cnn_pure_model",
                    "normal": 1e-4,
                    "sub_fix": "5",
                    "file": "learn_file"
                }
            }
        }
    ]
    ins = Acount(account_list[0])
    ins()


def test():
    code = "000001"
    startdate = "2019-09-29 00:00:00"
    df2 = ts.get_hist_data(code, ktype="D", start=startdate)
    print(df2)
    df2 = ts.get_realtime_quotes(["000001"])[["date", "open", "high", "low", "price", "volume"]]
    df2 = df2.rename(columns={"price": "close"})
    print(df2)
    exit()


if __name__ == "__main__":
    # test()
    engine = pyttsx3.init()
    logger.info("".center(100, "*"))
    logger.info("welcome to surfing".center(30, " ").center(100, "*"))
    engine.setProperty('rate', int(engine.getProperty('rate') * 0.85))
    engine.setProperty('volume', engine.getProperty('volume') * 1.0)
    engine.say("welcome to surfing!")
    engine.runAndWait()
    logger.info("".center(100, "*"))
    logger.info("")
    main(sys.argv[1:])
    logger.info("")
    engine.say("任务完成。")
    engine.say("bye!")
    engine.runAndWait()
    logger.info("bye!".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
