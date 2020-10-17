# coding=utf-8
import sys
import os
import codecs


#  本地读写
def local_read():
    # 文件读写
    with open(os.path.join(wav_path, i1), "r", encoding="utf-8") as f:
        # content = f.read()
        content = f.readline()

    # 从文件读取数据
    # 'r'：只读（缺省。如果文件不存在，则抛出错误）
    # 'w'：只写（如果文件不存在，则自动创建文件）
    # 'a'：附加到文件末尾
    # 'r+'：读写
    data = codecs.open("2.txt", encoding="UTF-8")
    # 一行一行读取数据
    data1 = data.readline()
    print(data1)
    # 度去完数据要把数据对象进行关闭，从内存里面释放出来
    data.close()

    f = codecs.open('c:/intimate.txt','a','utf-8')
    f.write(u'中文')
    s = '中文'
    f.write(s.decode('gbk'))
    f.close()

# 远程读写
def remote_read():
    url = "https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/data/titanic.csv"
    response = urllib.request.urlopen(url)
    html = response.read()
    with open('titanic.csv', 'wb') as f:
        f.write(html)


# 交互读写
def mutal_read():
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ans = 0
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        for v in values:
            ans += v
    print(ans)


if __name__ == "__main__":
    local_read()
    mutal_read()
    remote_read()
