# coding=utf-8
import sys
import os


#  本地读写
def local_read():
    # 文件读写
    with open(os.path.join(wav_path, i1), "r", encoding="utf-8") as f:
        # content = f.read()
        content = f.readline()


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
