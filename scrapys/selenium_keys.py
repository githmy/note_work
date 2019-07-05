#!/usr/bin/env python
# coding: utf-8

# 调用键盘按键操作时需要引入的Keys包
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import Chrome
import os
import time
import pprint
import urllib.request
import pandas as pd

cmd_path = os.getcwd()
basic_path = os.path.join(cmd_path, "..", "data")
project_path = os.path.join(basic_path, "spider")
data_path = os.path.join(project_path, "data")


class Yangcong():
    def __init__(self):
        self.url = "https://yangcong345.com/#/login?type=login"
        self.driver = Chrome("D:\Chrome下载\chromedriver.exe")

    def log_in(self):
        self.driver.get(self.url)
        self.driver.implicitly_wait(10)  # 智能等待10
        time.sleep(3)  # 睡3分钟，等待页面加载
        self.driver.save_screenshot("0.jpg")
        # 输入账号
        time.sleep(3)  # 睡3分钟，等待页面加载
        # 多元素数组发现
        self.driver.find_elements_by_xpath('//button')
        # 单元素发现
        self.driver.find_element_by_xpath('//*[@id="username"]').send_keys("18721986267")
        # 输入密码
        self.driver.find_element_by_xpath('//*[@id="password"]').send_keys("fff111QQQ")
        # 点击登陆
        self.driver.find_element_by_class_name("btn-bg-blue").click()
        time.sleep(2)
        self.driver.save_screenshot("save.jpg")
        # 最大化窗口
        self.driver.maximize_window()
        # 时间后的默认等待时间
        self.driver.implicitly_wait(6)
        # 点击后请求的网络状态
        aa = self.driver.get_network_conditions()
        # 类的名字只含部分
        aab = self.driver.find_elements_by_xpath('//div[contains(@class,"wrape-main")]//ul//button')
        # 元素下的文本
        bbc = aab.find_element_by_xpath('..//h3').text
        print(bbc)
        aa = self.driver.get_log("browser")
        pprint.pprint(aa)
        aa = self.driver.get_log("driver")
        pprint.pprint(aa)
        # aa = self.driver.get_log("client")
        # print(aa)
        aa = self.driver.page_source
        pprint.pprint(aa)
        # 输出登陆之后的cookies
        cookiejson = self.driver.get_cookies()
        print(cookiejson)
        wstr = cookiejson[0]["value"]
        print(wstr)

        # 获取当前url
        print(self.driver.current_url)

        # 页面 请求
        self.driver.get("http...")
        # 页面 刷新
        self.driver.refresh()

        # 鼠标悬停
        from selenium.webdriver.common.action_chains import ActionChains
        # 鼠标悬停在搜索设置按钮上
        mouse = self.driver.find_element_by_link_text("设置")
        ActionChains(self.driver).move_to_element(mouse).perform()
        # context_click()
        # 双击鼠标：double_click()

        # 等待加载
        from selenium.webdriver.support.ui import WebDriverWait
        WebDriverWait(self.driver, 10).until(EC.title_contains("元素"))

        # 执行js
        # 到网页最上端
        self.driver.executeScript("window.scrollTo(0,0)")
        # 网页后退
        self.driver.back()

    def __del__(self):
        self.driver.close()
        # 关闭浏览器
        self.driver.quit()


def test_selenium():
    yangcong = Yangcong()
    yangcong.log_in()  # 之后调用登陆方法


if __name__ == '__main__':
    test_selenium()
