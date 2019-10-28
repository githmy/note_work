import os
import json
import operator
import pandas as pd
from utils.connect_mongo import MongoDB

outpath = os.path.join("..", "data", "mongofiles")


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
    config_new = {
        # 'host': "192.168.1.52",
        'host': "127.0.0.1",
        'port': 27017,
        'database': "thinking2ht",
        'col': "examples",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
    }
    ins_new = MongoDB(config_new)
    res_map_new = ins_new.exec_require(quire_list)
    # 1. change list
    changemap_idlist = {}
    changemap_pointlist = {}
    for typecol in quire_list:
        changemap_idlist[typecol] = []
        changemap_pointlist[typecol] = []
        tmpoldlist = list(res_map_old[typecol])
        tmpnewlist = list(res_map_new[typecol])
        for old_id in tmpoldlist:
            for new_id in tmpnewlist:
                if old_id["_id"] == new_id["_id"]:
                    if operator.ne(set(old_id["mainReviewPoints"]), set(new_id["mainReviewPoints"])):
                        changemap_idlist[typecol].append(new_id["_id"])
                        changemap_pointlist[typecol].append(json.dumps(new_id["mainReviewPoints"]))
                    break
    for typecol in quire_list:
        ttpu = pd.DataFrame({"_id": changemap_idlist[typecol], "mainReviewPoints": changemap_pointlist[typecol]})
        ttpu.to_csv(os.path.join(outpath, '{}.txt'.format(typecol)), index=False, encoding="utf-8")
    print("end")


if __name__ == "__main__":
    main()
