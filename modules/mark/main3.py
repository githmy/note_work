import sys
import datetime
from modules.portfolio import Portfolio
from modules.event import *
from modules.datahandle import CSVDataHandler
from modules.strategys import MovingAverageCrossStrategy
from modules.executions import SimulatedExecutionHandler
from modules.backtests import Backtest
from utils.log_tool import *


def main(paralist):
    logger.info(paralist)
    # 1. 起止 学习 回测 的三个时间
    start_date = datetime.datetime(1990, 1, 1, 0, 0, 0)
    heartbeat = 0.0
    csv_dir = data_path
    # csv_list = ["ChinaBank", "DalianRP", "SAPower"]
    csv_list = ["SAPower"]
    initial_capital = 10000.0
    backtest = Backtest(csv_dir, csv_list, initial_capital, heartbeat, start_date,
                        CSVDataHandler, SimulatedExecutionHandler, Portfolio, MovingAverageCrossStrategy)
    backtest.simulate_trading()


if __name__ == "__main__":
    logger.info("".center(100, "*"))
    logger.info("welcome to surfing".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    main(sys.argv[1:])
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
