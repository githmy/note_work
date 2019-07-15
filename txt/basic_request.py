import os
import urllib.request
import requests


def session_way():
    session = requests.Session()
    post_url = "https://yangcong345.com/#/login?type=login"
    post_data = {"email": "mr_mao_hacker@163.com", "password": "alarmchime"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36"
    }
    # 1. 登录
    # 使用session发送post请求，cookie保存在其中
    session.post(post_url, data=post_data, headers=headers)
    # 2. 请求
    # 在使用session进行请求登陆之后才能访问的地址
    r = session.get("https://yangcong345.com", headers=headers)
    print(r.content.decode())
    # 3. 保存页面
    with open("renren1.html", "w", encoding="utf-8") as f:
        f.write(r.content.decode())


def request_skill():
    # 1. 超时
    response = requests.get("https://www.baidu.com", timeout=10)
    # 2. 免 ssl 认证
    response = requests.get("https://www.baidu.com", verify=False)
    # 3. COOKIE转字典
    requests.utils.dict_from_cookiejar(response.cookies)
    # 4. 字典转COOKIE
    requests.utils.cookiejar_from_dict({"aa": "bb"})
    # 5. url解密
    requests.utils.unquote("http%3a%2f%2fbaidu.com%2f?kw=....")
    # 6. url加密
    requests.utils.quote("http://baidu.com/?kw=张三")
    # 7. 判断是否成功
    assert response.status_code == 200
    # 8. 内容返回
    # response.text 根据http头推测文本解码
    response.text
    # 默认编码改方式
    response.encoding = "gbk"
    # response.content 未指定编码方式
    response.content.decode("utf-8")
    # 9. url 拼接
    response.urljoin("/items/book")
    aa = urllib.parse.urljoin(response.url, "/items/book")


def request_way():
    link_demo = "https://hls.media.yangcong345.com/pcM/pcM_58c26cbb36eaf35866aae1161.ts"
    # 1. 请求
    response = urllib.request.urlopen(link_demo)
    cat_vido = response.read()
    seed_name = "aaa.m3u8"
    # 2. 保存
    with open(seed_name, 'wb') as f:
        f.write(cat_vido)


def request_proxy_way():
    # 访问网址
    url = 'https://www.whatismyip.com/my-ip-information/?iref=home'
    url = 'http://www.whatismyip.com.tw/'
    # 这是代理IP
    proxy = {'http': '106.46.136.112:808'}
    # 创建ProxyHandler
    proxy_support = urllib.request.ProxyHandler(proxy)
    # 创建Opener
    opener = urllib.request.build_opener(proxy_support)
    # 添加User Angent
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36')]
    # 安装OPener
    urllib.request.install_opener(opener)
    # 使用自己安装好的Opener
    response = urllib.request.urlopen(url)
    # 读取相应信息并解码
    html = response.read().decode("utf-8")
    # 打印信息
    print(html)


if __name__ == '__main__':
    session_way()
    request_way()
