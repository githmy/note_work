
# QUANTAXIS 量化金融策略框架


点击此处下载 QUANTAXIS2019 介绍性pdf [quantaxis 2019](http://whl.yutiansut.com/QUANTAXIS%202019.pdf)

-------------------------------------------------------------

Quantitative Financial FrameWork

从数据爬取-清洗存储-分析回测-可视化-交易复盘的本地一站式解决方案


![logo](http://pic.yutiansut.com/QUANTAXIS_LOGO_LAST_small.jpg)



![pypidownloads](https://img.shields.io/pypi/dm/quantaxis.svg)
![pypidownloads](https://img.shields.io/pypi/dw/quantaxis.svg)
[![Backers on Open Collective](https://opencollective.com/QUANTAXIS/backers/badge.svg)](#backers) [![Sponsors on Open Collective](https://opencollective.com/QUANTAXIS/sponsors/badge.svg)](#sponsors) 

[![Github workers](https://img.shields.io/github/watchers/quantaxis/quantaxis.svg?style=social&label=Watchers&)](https://github.com/quantaxis/quantaxis/watchers)
[![GitHub stars](https://img.shields.io/github/stars/quantaxis/quantaxis.svg?style=social&label=Star&)](https://github.com/quantaxis/quantaxis/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/quantaxis/quantaxis.svg?style=social&label=Fork&)](https://github.com/quantaxis/quantaxis/fork)

[点击右上角Star和Watch来跟踪项目进展! 点击Fork来创建属于你的QUANTAXIS!]


## 祝贺 QUANTAXIS 入选 [GITEE 最有价值开源项目 GVP](https://gitee.com/gvp)
![gvp](http://pic.yutiansut.com/gvp.jpg)



![main_1](http://pic.yutiansut.com/Main_1.gif)
![presentbyyutiansut](http://pic.yutiansut.com/yutiansut-logo.png)



![version](https://img.shields.io/pypi/v/quantaxis.svg)
![build](https://travis-ci.org/QUANTAXIS/QUANTAXIS.svg?branch=master)
[![Codefresh build status]( https://g.codefresh.io/api/badges/pipeline/quantaxis/QUANTAXIS%2FQUANTAXIS%2FQUANTAXIS?branch=master&key=eyJhbGciOiJIUzI1NiJ9.NWEzMGMwNzE4NTk5YzAwMDAxNWEzOGQ3.INF0oVNGLK79Ql3f10Vs9_CJfwsfq1FYjCbbOKJzBe8&type=cf-1)]( https://g.codefresh.io/pipelines/QUANTAXIS/builds?repoOwner=QUANTAXIS&repoName=QUANTAXIS&serviceName=QUANTAXIS%2FQUANTAXIS&filter=trigger:build~Build;branch:master;pipeline:5c3ade3fc67fe4e3328c9da1~QUANTAXIS)
[![BCH compliance](https://bettercodehub.com/edge/badge/QUANTAXIS/QUANTAXIS?branch=master)](https://bettercodehub.com/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d8504e4af33747bb8117579212425af9)](https://www.codacy.com/app/yutiansut/QUANTAXIS?utm_source=github.com&utm_medium=referral&utm_content=yutiansut/QUANTAXIS&utm_campaign=badger)
[![StackShare](https://img.shields.io/badge/tech-stack-0690fa.svg?style=flat)](https://stackshare.io/yutiansut/quantaxis)
![QAS](https://img.shields.io/badge/QAS-%200.0.8-brown.svg)
![python](https://img.shields.io/badge/python-%203.7/3.6-darkgrey.svg)
![Npm](https://img.shields.io/badge/Npm-%200.4.0-yellow.svg)
![author](https://img.shields.io/badge/Powered%20by-%20%20yutiansut-red.svg)
![license](https://img.shields.io/badge/License-%20MIT-brightgreen.svg)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FQUANTAXIS%2FQUANTAXIS.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FQUANTAXIS%2FQUANTAXIS?ref=badge_shield)
[![Waffle.io - Issues in progress](https://badge.waffle.io/yutiansut/QUANTAXIS.png?label=in%20progress&title=In%20Progress)](http://waffle.io/yutiansut/QUANTAXIS)
[![Reviewed by Hound](https://img.shields.io/badge/Reviewed_by-Hound-8E64B0.svg)](https://houndci.com)
[![](http://rescdn.qqmail.com/zh_CN/htmledition/images/function/qm_open/ico_mailme_22.png)](http://mail.qq.com/cgi-bin/qm_share?t=qm_mailme&email=VSwgITw0OyYgIRUjPCV7JCR7Njo4)

## Contributors

This project exists thanks to all the people who contribute. [[Contribute](https://github.com/QUANTAXIS/QUANTAXIS/graphs/contributors)].
<a href="https://github.com/QUANTAXIS/QUANTAXIS/graphs/contributors"><img src="https://opencollective.com/QUANTAXIS/contributors.svg?width=890&button=false" /></a>


> 欢迎加群讨论: 563280067 [群链接](https://jq.qq.com/?_wv=1027&k=4CEKGzn) 

> QUANTAXIS 开发群: 773602202 (如果想要贡献代码 请加这个群 需要备注你的GITHUB ID)

> 欢迎关注公众号: ![公众号](http://pic.yutiansut.com/qrcode_for_gh_bbb47e0550f7_258%20%281%29.jpg)

> 许多问题 可以在 [GITHUB ISSUE](https://github.com/QUANTAXIS/QUANTAXIS/issues)中找到, 你可以提出新的issue

QUANTAXIS量化金融策略框架,是一个面向中小型策略团队的量化分析解决方案. 我们通过高度解耦的模块化以及标准化协议,可以快速的实现面向场景的定制化解决方案.QUANTAXIS是一个渐进式的开放式框架,你可以根据自己的需要,引入自己的数据,分析方案,可视化过程等,也可以通过RESTful接口,快速实现多人局域网/广域网内的协作.

==========================================


## 关联项目:

技术栈: python/nodejs/vue/mongodb/rabbitmq/c++

### 核心工具链(生产环境在用)

#### 已开源

> 数据存储/数据分析/回测

- [QUANTAXIS](https://github.com/QUANTAXIS/QUANTAXIS) QUANTAXIS的核心部分

> WEB相关, http/websocket/开放数据接口

- [QUANTAXIS_WEBSERVER](https://github.com/QUANTAXIS/QUANTAXIS_WEBSERVER) 基于tornado的web api/ websocket

> 分布式相关, 任务异步执行, 跨进程分布式消息订阅分发

- [QUANTAXIS_RUN](https://github.com/QUANTAXIS/quantaxis_run) 基于rabbitmq/celery的分布式任务部署
- [QUANTAXIS_PUBSUB](https://github.com/QUANTAXIS/QAPUBSUB) 基于RABBITMQ的消息分发订阅

> 接口相关: 交易账户/ 期货接口封装/ Trader实例
- [QUANTAXIS OTGBROKER](https://github.com/QUANTAXIS/QAOTGBROKER) 基于OPEN_TRADE_GATEWAY的接口封装
- [QUANTAXIS CTPBEEBROKER](https://github.com/QUANTAXIS/QACTPBeeBroker) 基于CTPBee的接口封装
- [QUANTAXIS_ATBROKER](https://github.com/QUANTAXIS/QA_AtBroker) 基于海风at的接口封装
- [QUANTAXIS TRADER](https://github.com/yutiansut/QATrader) 一个开源的websocket版本的期货交易实例

> 行情相关: 主推行情实现/ 基于OU过程的模拟行情
- [QUNATAXIS MARKETCOLLECTOR](https://github.com/yutiansut/QUANTAXIS_RealtimeCollector) 全市场订阅分发的行情推送
- [QUANTAXIS_RandomPrice](https://github.com/yutiansut/QUANTAXIS_RandomPrice) 基于OU过程的随机行情模拟


#### 未开源

> 实时交易解决方案/ 无人值守/状态汇报/实时账户评估/多账户/策略账户拆分/事件流风控/PB系统/CEP引擎/多系统终端

- [QUANTAXIS_REALTIME_RESOLUTION](https://github.com/yutiansut/QUANTAXIS_REALTIME_RESOLUTION) 实时交易/部署解决方案(未开源)
- [QUANTAXIS UNICORN](https://github.com/yutiansut/quantaxis_unicorn) QUANTAXIS 策略托管, 交易监控解决方案(未开源)
- [QUANTAXIS_RANK](https://github.com/yutiansut/QARank) QUANTAXIS实时账户评估
- [QUANTAXIS_CEPEngine](https://github.com/yutiansut/QACEPEngine) QUANTAXIS 复杂事件处理引擎
- [QUANTAXIS_PBSystem](https://github.com/yutiansut/QAPBSystem) QUANTAXIS PB系统
- [QUANTAXIS_QARISKPRO](https://github.com/yutiansut/QARISKPRO) QUANTAXIS 多市场多账户集成的实时风控系统
- [QUANTAXIS QADESKPRO](https://github.com/yutiansut/qadeskpro) 新版本客户端网页(部分开源)

> tick回测

- [QUANTAXIS TICKBacktest](https://github.com/yutiansut/QATickBacktest) tick回测 支持真实tick/仿真tick

> jupyterhub 定制(多人编辑)

- [QUANTAXIS JUPYTERHUB](https://github.com/yutiansut/QAJupyter)

> docker cluster

- [QUANTAXIS PROCluster](https://github.com/yutiansut/QAPRO_dockercluster) 一键部署的docker集群, 2地3中心的高可用灾备投研/交易环境


### 社区提供的工具链

- [QUANTAXIS_MONITOR_GUI](https://github.com/QUANTAXIS/QUANTAXIS_Monitor_GUI) 基于QT的python监控
- (目前废弃)[QUANTAXIS_DESKTOP](https://github.com/QUANTAXIS/QADESKTOP) 基于VUE.js/ ELECTRON的 桌面终端
- [portable_QA](https://github.com/QUANTAXIS/portable_QA) 一个独立的python环境,免配置
- [QUANTAXIS_CRAWLY](https://github.com/QUANTAXIS/QUANTAXIS_CRAWLY) 爬虫部分

![qaresolution](http://pic.yutiansut.com/QQ%E6%88%AA%E5%9B%BE20190815112455.png)

==========================================

<!-- TOC -->

- [QUANTAXIS 量化金融策略框架](#quantaxis-量化金融策略框架)
    - [0. 一键部署环境](#0-QASERVICE)
    - [1. 功能](#1-功能)
        - [1.1 行情服务](#11-行情服务)
            - [1.1.1 股票/期货/期权/美股/外汇/宏观的历史/实时行情(日线/分钟线/tick/实时五档)服务](#111-股票期货期权美股外汇宏观的历史实时行情日线分钟线tick实时五档服务)
            - [1.1.2 财务/基本面/宏观数据](#112-财务基本面宏观数据)
            - [1.1.3 自定义数据源的数据](#113-自定义数据源的数据)
        - [1.2 数据运维服务](#12-数据运维服务)
        - [1.3 分析服务](#13-分析服务)
            - [1.3.1 专门为A股股票数据适配的数据结构](#131-专门为a股股票数据适配的数据结构)
            - [1.3.2 精心为A股指标计算适配的指标类](#132-精心为a股指标计算适配的指标类)
        - [1.4 可扩展事件驱动框架](#14-可扩展事件驱动框架)
        - [1.5 回测服务](#15-回测服务)
            - [1.5.1 股票/日内t0/ 的日线/分钟线级别回测](#151-股票日内t0-的日线分钟线级别回测)
        - [1.6 实盘](#16-实盘)
            - [1.6.1 股票/实盘易 QAShiPaneBroker](#161-股票-实盘易--qashipanebroker)
            - [1.6.2 股票/TTS    QATTSBroker](#162-股票-tts----qattsbroker)
            - [1.6.3 期货/Open-trade-gateway QAOTGBROKER](#163-期货--qaotgbroker)
            - [1.6.4 期货/海风AT  QA_ATBroker](#164-期货--海风at封装-qaatbroker)
            - [1.6.5 期货/VNPY   QAVNPYBroker](#165-期货-vnpy-qavnpybroker)
        - [1.7 网站HTTP服务](#17-网站http服务)
            - [1.7.1 网站后台标准化接口](#171-网站后台标准化接口)
    - [2. 文档](#2-文档)
    - [3. 安装和部署](#3-安装和部署)
        - [3.0 安装说明](#30-安装说明)
        - [3.1 小白式上手](#31-小白式上手)
        - [3.2 部署式安装](#32-部署式安装)
        - [3.3  本地代码 开发式安装](#33--本地代码-开发式安装)
        - [3.4 代码提交式安装](#34-代码提交式安装)
    - [4. 更新](#4-更新)
    - [5. Docker](#5-docker)
    - [6. 使用说明](#6-使用说明)
    - [7. Jupyter示例](#7-jupyter示例)
    - [8. 开发计划](#8-开发计划)
    - [9. 常见问题FAQ](#9-常见问题faq)
    - [10. 项目捐赠](#10-项目捐赠)
    - [11. 回测Webkit插件概览](#11-回测webkit插件概览)
    - [12. QUANTAXIS 标准化协议和未来协议](#12-quantaxis-标准化协议和未来协议)
    - [13. 电脑配置推荐](#13-电脑配置推荐)
    - [14. 视频录播](#14-录播)
    - [License](#license)

<!-- /TOC -->




## 0. QASERVICE

docker 一键部署(无需任何环境! 最好是直接拿个新机器)

### 安装docker 

ubuntu 一键脚本

```
wget https://raw.githubusercontent.com/QUANTAXIS/QUANTAXIS/master/config/install_docker.sh
sudo bash install_docker.sh
```

win/mac 安装

非常简单 去docker网站下载win/mac的docker_desktop

文件较大, 我在群文件也共享了

或者你可以使用国内镜像站

- win [https://download.daocloud.io/d/Docker_Mirror/Docker_for_Windows_Mac/17.03.1-ce/docker-for-windows-stable.msi]
- mac [https://download.daocloud.io/d/Docker_Mirror/Docker_for_Windows_Mac/17.03.1-ce/docker-for-mac-stable.dmg]

ps: quantaxis强烈推荐不要使用win10以下的系统...(好吧忽略我)


### 使用QA_SERVICE

第一次使用
```
wget https://raw.githubusercontent.com/QUANTAXIS/QUANTAXIS/master/docker/qaservice_docker.sh
sudo bash qaservice_docker.sh
```

后续使用

```
docker-compose up -d
```

端口:

- 27017 mongodb
- 8888 jupyter
- 8010 quantaxis_webserver
- 81 quantaxis_community 社区版界面
- 61208 系统监控
- 15672 qa-eventmq


然后就可以开始你的量化之路了骚年!



##  1. 功能
======

![](http://pic.yutiansut.com/QUANTAXIS.png)

已经实现：

### 1.1 行情服务

#### 1.1.1 股票/期货/期权/美股/外汇/宏观的历史/实时行情(日线/分钟线/tick/实时五档)服务

参见 [QUANTAXIS的数据获取指南](Documents/DataFetch.md)

#### 1.1.2 财务/基本面/宏观数据

参见 [QUANTAXIS财务指标](Documents/financial_means.md)

#### 1.1.3 自定义数据源的数据

参见 [QUANTAXIS WEB 爬虫](Documents/crawler.md)

### 1.2 数据运维服务

一键更新 参见[WINDOWS数据自动更新](config/windows_autojob_updatedata.md)


### 1.3 分析服务

#### 1.3.1 专门为A股股票数据适配的数据结构

参见 [QUANTAXIS的数据结构](Documents/DataStruct.md)

参见 [QUANTAXIS行情研究](Documents/analysis.md)

#### 1.3.2 精心为A股指标计算适配的指标类

参见 [QUANTAXIS指标系统](Documents/indicators.md)


### 1.4 可扩展事件驱动框架

参见 [QUANTAXIS事件框架](Documents/about_event.md)


### 1.5 回测服务

#### 1.5.1 股票/日内t0/ 的日线/分钟线级别回测

参见 [QUANTAXIS的账户结构](Documents/account.md)

参见 [QUANTAXIS 账户风险分析插件指南](Documents/risk.md)

参见 [QUANTAXIS回测委托成交结算的说明](Documents/orderanddeal.md)

参见 [QUANTAXIS回测分析](Documents/backtestanalysis.md)

参见 [常见策略整理](Documents/strategy.md)

参见 [简单策略回测详解](Documents/QUANTAXIS回测分析全过程讲解.md)

参见 [T0交易的账户详解](Documents/TEST_ORDER_BACKTEST_T0.md)

### 1.6 实盘

#### 1.6.1 (股票) 实盘易  QAShipaneBroker(推荐)


实盘易插件 参见[实盘易](http://www.iguuu.com/e?x=18839)

实盘易安装注意 参见[安装注意](shipane_install_memo.md)

####  1.6.2 (股票) TTS    QATTSBroker

具体参见[QATTSBroker](QUANTAXIX/QAMARKET/QATTSBroker.py)

#### 1.6.3 (期货)  QAOtgBroker

CTP 的websocket交易 :  simnow demo:  ws://www.yutiansut.com:7988

关于websocket交易(可能存在一些不确定的bug 因此暂时不开放):

目前已经开源 参见 [QAOTGBROKER](https://github.com/yutiansut/QAOTGBROKER)

### 1.6.4 (期货) QATRADER(推荐)

CTP 的websocket 封装 包括完整的账户/HTTP控制等

目前已经开源 参见 [QATRADER](https://github.com/yutiansut/QATRADER)

#### 1.6.5 (期货)  海风AT封装 QAAtBroker

CTP 的海风broker魔改封装:  [QA_AtBroker](https://github.com/QUANTAXIS/QA_AtBroker)


#### 1.6.6 (期货) VNPY QAVNPYBroker

目前等VNPY 2.0的横空出世, 兼容py3.6/3.7后将进行对接



### 1.7 网站HTTP服务(目前已经独立为 QUANTAXIS_WEBSERVICE)

#### 1.7.1 网站后台标准化接口

参见 [QUANTAXIS WEB API说明](Documents/backendapi.md)



![](http://pic.yutiansut.com/framework.png)


- [QUANTAXIS 2018开发计划表](job_list.md)


##  2. 文档

文档参见: [book](http://book.yutiansut.com)

下载文档手册(实时更新)

[PDF](https://www.gitbook.com/download/pdf/book/quantaxis/quantaxis) | [MOBI](https://www.gitbook.com/download/mobi/book/quantaxis/quantaxis) | [EPUB](https://www.gitbook.com/download/epub/book/quantaxis/quantaxis)

##  3. 安装和部署

```
quantaxis 1.4.4 增加了cython编译加速过程, 如果需要cython的同学 使用:

python setup_cython.py build_ext --inplace

再继续 pip install -e .

必须要注意的事情: 以上为可选项 并非强制!!


cython 提速:

1.4.3的Account.receive_simpledeal( 包括接受订单/更新账户) 为单次 1.88微秒
1.4.4的Account.receive_simpledeal  为单次 570纳秒
```

### 3.0 安装说明
参见 [安装说明](Documents/install.md)

### 3.1 小白式上手


参见 [小白上手教程WIN](Documents/install_for_new_to_coding.md)

### 3.2 部署式安装


```
pip install quantaxis -U
```
### 3.3  本地代码 开发式安装

本地安装
```
git clone https://github.com/quantaxis/quantaxis --depth 1

cd quantaxis

pip install -e .
```
### 3.4 代码提交式安装

代码提交式安装 代码提交参见  [代码提交](https://github.com/QUANTAXIS/QUANTAXIS/blob/master/Documents/about_pr.md)

- fork QUANTAXIS 到你的github账户

```
git clone https://github.com/你的账户名/quantaxis
```

##  4. 更新
参见 [更新说明](Documents/update.md)

##  5. Docker
参见 [Docker](Documents/docker.md)
##  6. 使用说明
参见 


* [QUANTAXIS的使用示例](https://github.com/quantaxis/QADemo)


##  7. Jupyter示例
参见 [Jupyter示例](jupyterexample)


##  8. 开发计划
参见 [开发计划](job_list.md)

##  9. 常见问题FAQ

参见 [FAQ](Documents/FAQ.md)

##  10. 项目捐赠

写代码不易...请作者喝杯咖啡呗?


![](http://pic.yutiansut.com/alipay.png)

(PS: 支付的时候 请带上你的名字/昵称呀 会维护一个赞助列表~ )

[捐赠列表](CONTRIBUTING.md)



##  11. QUANTAXIS 桌面级产品(全平台 WIN/MAC/LINUX)


参见 [QUANTAXIS/QADESKTOP](https://github.com/quantaxis/qadesktop)

![](http://pic.yutiansut.com/qad1.png)
![](http://pic.yutiansut.com/qad2.png)
![](http://pic.yutiansut.com/qad3.png)
![](http://pic.yutiansut.com/qad4.png)
![](http://pic.yutiansut.com/qad5.png)
![](http://pic.yutiansut.com/qad6.png)
![](http://pic.yutiansut.com/qad8.png)

模拟实盘多账户管理

![](http://pic.yutiansut.com/QQ%E6%88%AA%E5%9B%BE20190311015440.png)
![](http://pic.yutiansut.com/QQ%E6%88%AA%E5%9B%BE20190311015451.png)
![](http://pic.yutiansut.com/QQ%E6%88%AA%E5%9B%BE20190311015550.png)
![](http://pic.yutiansut.com/QQ%E6%88%AA%E5%9B%BE20190311015537.png)

手工定制版本 LAB

![](http://pic.yutiansut.com/QADESK_LAB.png)

##  12. QUANTAXIS 标准化协议和未来协议


QUANTAXIS-Stardand-Protocol 版本号0.0.8

详情参见  [QUANATXISProtocol](Documents/readme.md)


##  13. 电脑配置推荐

推荐配置:
6代以上CPU+ 16/32GB DDR3/DDR4内存+ 256GB以上SSD硬盘

最低配置:
支持X64位的CPU

因为在存储本地数据的时候,需要存储超过2GB的本地数据,而32位的MONGODB最高只支持2GB左右的数据存储,因此最少需要一个X64位的CPU

如果SSD资源够用,尽量将数据存储在SSD中,增加```wiretiger```写盘的速度

如果是阿里云/腾讯云的服务器,请在最初的时候 选择64位的操作系统



## 14. 录播



## 视频录播(2019-03-23)

QUANTAXIS 2019 支持多市场的面向场景解决方案
![](http://pic.yutiansut.com/QAFINANCIAL2019.jpg)
[在哔哩哔哩观看录播](https://www.bilibili.com/video/av47284727/)

==========================================

## Backers

Thank you to all our backers! 🙏 [[Become a backer](https://opencollective.com/QUANTAXIS#backer)]

<a href="https://opencollective.com/QUANTAXIS#backers" target="_blank"><img src="https://opencollective.com/QUANTAXIS/backers.svg?width=890"></a>


## Sponsors

Support this project by becoming a sponsor. Your logo will show up here with a link to your website. [[Become a sponsor](https://opencollective.com/QUANTAXIS#sponsor)]

<a href="https://opencollective.com/QUANTAXIS/sponsor/0/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/0/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/1/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/1/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/2/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/2/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/3/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/3/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/4/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/4/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/5/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/5/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/6/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/6/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/7/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/7/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/8/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/8/avatar.svg"></a>
<a href="https://opencollective.com/QUANTAXIS/sponsor/9/website" target="_blank"><img src="https://opencollective.com/QUANTAXIS/sponsor/9/avatar.svg"></a>



## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FQUANTAXIS%2FQUANTAXIS.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FQUANTAXIS%2FQUANTAXIS?ref=badge_large) [![Join the chat at https://gitter.im/QUANTAXIS_tauruswang/Lobby](https://badges.gitter.im/QUANTAXIS_tauruswang/Lobby.svg)](https://gitter.im/QUANTAXIS_tauruswang/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
