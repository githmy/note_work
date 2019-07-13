import pytesseract
import urllib
import urllib.request
from bs4 import BeautifulSoup
from lxml import etree
from PIL import Image


# xpath 语法参考网址
# https://msdn.microsoft.com/zh-cn/library/ms256039(v=vs.80).aspx

def test():
    seed_url = "https://www.xicidaili.com/nn"
    test_url = "https://www.whatismyip.com/my-ip-information/?iref=home"
    test_url = "https://www.whatismyip.com/ip-address-lookup/"
    # 创建ProxyHandler
    proxy_support = urllib.request.ProxyHandler(None)
    # 创建Opener
    opener = urllib.request.build_opener(proxy_support)
    # 添加User Angent
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2934.87 Safari/537.36')]
    # 安装OPener
    urllib.request.install_opener(opener)
    # 使用自己安装好的Opener
    with urllib.request.urlopen(seed_url) as response:
        # 2. 规范返回对象
        soup = BeautifulSoup(response.read().decode('utf-8'), "html.parser")
        html = etree.HTML(soup.prettify())
        a_list = html.xpath('//table[@id="ip_list"]//tr')
        iplist = []
        for i1 in a_list:
            iplist.append("{}://{}:{}".format(i1[5].text.strip().lower(), i1[1].text.strip(), i1[2].text.strip()))
        print(iplist[1:])
        # return iplist[1:]
    # 测试
    with urllib.request.urlopen(test_url) as response:
        # ori_page = response.read().decode('utf-8')
        # 2. 规范返回对象
        soup = BeautifulSoup(response.read().decode('utf-8'), "html.parser")
        html = etree.HTML(soup.prettify())
        a_list = html.xpath('//div[@class="container-fluid"]//div[@class="row"]/div[1]/table//text()')
        # 其他语法
        # /bookstore/book[1]
        # /bookstore/book[last()]
        # /bookstore/book[last()-1]
        # /bookstore/book[position()<3]
        # //title[@lang="eng"]
        # /bookstore/book[price>35.00]
        # /bookstore/book[price>35.00]/title
        print(a_list)


if __name__ == '__main__':
    test()
