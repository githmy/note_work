import os
import json
import operator
import pandas as pd
from utils.connect_mongo import MongoDB
import re

outpath = os.path.join("..", "data", "mongofiles")


def iterjson(source):
    reslist = []
    if type(source) is dict:
        for v in source.values():
            reslist += iterjson(v)
    elif type(source) is list:
        for v in source:
            reslist += iterjson(v)
    elif type(source) is str:
        reslist = [item.strip("$") for item in re.findall(r"\$.*?\$", source, re.U | re.M)]
    else:
        pass
    return reslist


def main():
    # 0. 先执行倒库
    quire_list = ["examples", "questions"]
    config_old = {
        'host': "127.0.0.1",
        'port': 27017,
        'database': "thinking2-test",
        'col': "examples",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
    }
    ins_old = MongoDB(config_old)
    res_map_old = ins_old.exec_require(quire_list)
    dataout = []
    for typecol in quire_list:
        for i1 in res_map_old[typecol]:
            tmpjson = {"title": i1["description"], "mainReviewPoints": i1["mainReviewPoints"],
                       "reviewPoints": i1["reviewPoints"]}
            dataout.append(tmpjson)
    ttpu = pd.DataFrame(dataout)
    ttpu.to_csv(os.path.join(outpath, 'text.txt'), index=False, encoding="utf-8")
    exit()


if __name__ == "__main__":
    main()
