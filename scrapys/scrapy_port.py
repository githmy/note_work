import os

def run_lele_scrapy():
    print("开始执行：run_lele_scrapy")
    a = os.popen("cd lele;scrapy crawl spiderlele").readlines()
    print(a)
    print("执行结束：run_lele_scrapy")

if __name__ == '__main__':
    run_lele_scrapy()
