# git clone --depth 1 git://github.com/vnpy/vnpy.git
# cd vnpy && python setup.py build && bash install.sh
# pip install vnpy
# pip install pyqt5
# https://www.lfd.uci.edu/~gohlke/pythonlibs/
# ta-lib
# wget https://download.lfd.uci.edu/pythonlibs/w3jqiv8s/TA_Lib-0.4.18-cp37-cp37m-win_amd64.whl
# pip install TA_Lib-0.4.18-cp37-cp37m-win_amd64.whl

# encoding: UTF-8
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp
from vnpy.gateway.ctp import CtpGateway
from vnpy.app.cta_strategy import CtaStrategyApp
from vnpy.app.cta_backtester import CtaBacktesterApp


def main():
    """Start VN Trader"""
    qapp = create_qapp()

    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    main_engine.add_gateway(CtpGateway)
    main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(CtaBacktesterApp)

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    qapp.exec()


if __name__ == "__main__":
    main()