from __future__ import print_function

# sudo apt-get install python3.5-dev -y
# sudo apt-get install python3-venv -y
# sudo apt-get install default-jre -y
"""
mkdir myeasytrader
vim ht.json
{
  "userName": "用户名",
  "servicePwd": "通讯密码",
  "trdpwd": "加密后的密码"
}
"""
# pip install easytrader
import easytrader


def server_client():
    # 交易服务端——启动服务
    from easytrader import server

    server.run(port=1430)  # 默认端口为 1430

    # 量化策略端——调用服务
    from easytrader import remoteclient

    user = remoteclient.use('使用客户端类型，可选 yh_client, ht_client, ths, xq等', host='服务器ip', port='服务器端口，默认为1430')
    user.buy(......)
    user.sell(......)

    # 主动监控远端量化策略的成交记录或仓位变化
    # 1) 初始化跟踪的 trader
    xq_user = easytrader.use('xq')
    xq_user.prepare('xq.json')
    # 初始化跟踪 joinquant / ricequant 的 follower
    target = 'jq'  # joinquant
    target = 'rq'  # ricequant
    follower = easytrader.follower(target)
    follower.login(user='rq/jq用户名', password='rq/jq密码')
    # 连接 follower 和 trader
    # https://www.joinquant.com/algorithm/live/index?backtestId=xxx
    follower.follow(xq_user, 'jq的模拟交易url')
    # trade_cmd_expire_seconds 默认处理多少秒内的信号
    # cmd_cache 是否读取已经执行过的命令缓存，以防止重复执行
    jq_follower.follow(user, '模拟交易url', trade_cmd_expire_seconds=100000000000, cmd_cache=False)
    follower.follow(xq_user, run_id)
    # 注：ricequant的run_id即PT列表中的ID。
    # 2) 初始化跟踪 雪球组合 的 follower
    xq_follower = easytrader.follower('xq')
    xq_follower.login(cookies='雪球 cookies，登陆后获取，获取方式见 https://smalltool.github.io/2016/08/02/cookie/')
    # 连接 follower 和 trader
    xq_follower.follow(xq_user, 'xq组合ID，类似ZH123456', total_assets=100000)
    # 多用户跟踪多策略
    follower.follow(users=[xq_user, yh_user], strategies=['组合1', '组合2'], total_assets=[10000, 10000])
    # 使用市价单跟踪模式，目前仅支持银河
    follower.follow(***, entrust_prop='market')
    # 默认为0s。调大可防止卖出买入时卖出单没有及时成交导致的买入金额不足
    follower.follow(***, send_interval=30)  # 设置下单间隔为 30 s
    # 设置买卖时的滑点
    follower.follow(***, slippage=0.05)  # 设置滑点为 5%
    # 命令行模式
    # # 登录
    #  python cli.py --use yh --prepare gf.json
    # # 注: 此时会生成 account.session 文件保存生成的 user 对象
    # # 获取余额 / 持仓 / 以及其他变量
    #  python cli.py --get balance
    # # 买卖 / 撤单
    #  python cli.py --do buy 162411 0.450 100
    # # 查看帮助
    #  python cli.py --help


def main():
    """
    通用同花顺客户端 不支持自动登录，需要先手动登录。
    user = easytrader.use('ths')
    user.connect(r'客户端xiadan.exe路径') # 类似 r'C:\htzqzyb2\xiadan.exe'
    海通客户端 调用prepare函数自动登录。
    user = easytrader.use('htzq_client')
    user.prepare(user='用户名', password='雪球、银河客户端为明文密码', comm_password='华泰通讯密码，其他券商不用')
    华泰客户端 调用prepare函数自动登录。
    user = easytrader.use('ht_client')
    user.prepare(user='用户名', password='雪球、银河客户端为明文密码', comm_password='华泰通讯密码，其他券商不用')
    国金客户端 调用prepare函数自动登录。
    user = easytrader.use('gj_client') 
    user.prepare(user='用户名', password='雪球、银河客户端为明文密码', comm_password='华泰通讯密码，其他券商不用')
    雪球 调用prepare函数自动登录。
    user = easytrader.use('xq')
    user.prepare('/path/to/your/yh_client.json')  # 配置文件路径

    银河/国金客户端
    {
      "user": "用户名",
      "password": "明文密码"
    }
    华泰客户端
    {
      "user": "华泰用户名",
      "password": "华泰明文密码",
      "comm_password": "华泰通讯密码"
    }
    雪球
    {
      "cookies": "雪球 cookies，登陆后获取，获取方式见 https://smalltool.github.io/2016/08/02/cookie/",
      "portfolio_code": "组合代码(例:ZH818559)",
      "portfolio_market": "交易市场(例:us 或者 cn 或者 hk)"
    }
    """
    user = easytrader.use('ht')
    user.prepare('ht.json')
    # 获取资金状况
    user.balance
    # 获取持仓
    user.position
    # 买入
    user.buy('162411', price=0.55, amount=100)
    # 卖出
    user.sell('162411', price=0.55, amount=100)
    # 一键打新
    user.auto_ipo()
    # 撤单
    user.cancel_entrust('buy/sell 获取的 entrust_no')
    # 查询当日成交
    user.today_trades
    # 查询当日委托
    user.today_entrusts
    # 查询今日可申购新股
    from easytrader.utils.stock import get_today_ipo_data
    ipo_data = get_today_ipo_data()
    print(ipo_data)
    # 刷新数据
    user.refresh()
    # 雪球组合比例调仓
    user.adjust_weight('000001', 10)
    # 是将平安银行在组合中的持仓比例调整到10%
    # 退出客户端软件
    user.exit()
    # user = easytrader.use('yh_client')
    # user.prepare(user='1111111111111', password='111111')


if __name__ == '__main__':
    main()
