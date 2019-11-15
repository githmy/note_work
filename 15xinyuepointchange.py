import os
import re
import json
import operator
import pandas as pd
from utils.connect_mongo import MongoDB
from utils.path_tool import makesurepath

outpath = os.path.join("..", "data", "mongofiles")


def main():
    # 1. 获取库信息
    change_file = os.path.join("~", "Desktop", "100道例题.xlsx")
    ori_fold = os.path.join("e:\\project", "ht_data", "prod")
    out_fold = os.path.join("e:\\project", "data", "prod")
    makesurepath(out_fold)
    quire_list = ["examples", "questions", "assignmenttemplates"]
    config_new = {
        'host': "192.168.1.52",
        'port': 27017,
        'database': "thinking2ht",
        'col': "examples",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
    }
    ins = MongoDB(config_new)
    res_map = ins.exec_require(quire_list)
    mongolist = []
    for i1 in res_map:
        mongolist += list(res_map[i1])
    print(mongolist)
    # 2. 获取更改信息
    change_data = pd.read_excel(change_file, sheet_name='Sheet1', header=0).values
    print(change_data)
    # 3. 获取老文件信息
    file_names = os.listdir(ori_fold)

    # 4. 生成新mongo
    def deliterid(obj):
        if isinstance(obj, dict):
            try:
                del obj["__v"]
                # print("__v")
            except Exception as e:
                pass
            try:
                del obj["_id"]
                # print("_id")
            except Exception as e:
                pass
            for i1 in obj:
                # print(i1)
                deliterid(obj[i1])
        elif isinstance(obj, list):
            for i1 in obj:
                deliterid(i1)
        else:
            pass

    for title in mongolist:
        try:
            del title["__v"]
        except Exception as e:
            pass
        for i1 in title:
            if "_id" != i1:
                # print("starting", i1)
                deliterid(title[i1])
                # print(title)
                # exit()
    for id1, token in enumerate(change_data[:, 0]):
        for title in mongolist:
            if title["_id"] == token:
                title["mainReviewPoints"] = [change_data[id1, 1]]
                title["reviewPoints"] = [change_data[id1, 1]]
    # 5. 生成老文件
    new_contents = {}
    for token in file_names:
        print(token)
        with open(os.path.join(ori_fold, token), 'rt', encoding="utf8") as f:
            old_contents = "".join(f.readlines())
        new_contents[token] = []
        for title in mongolist:
            # normalstr = str(title["_id"])
            # print(r"{}".format(title["_id"]))
            searchsig = re.search(str(title["_id"]), str(old_contents), re.M)
            if searchsig is not None:
                new_contents[token].append(title)
        break

    # 6. 格式化输出
    def dequota(restrs):
        reslist = restrs.split("\n")
        newlist = []
        for i1 in reslist:
            tmparry = i1.split(":")
            llenth = len(tmparry)
            if len(tmparry) > 1:
                tailstr = ":".join(tmparry[1:])
                i1 = ":".join([tmparry[0].replace('"', ''), tailstr])
            newlist.append(i1)
        return "\n".join(newlist)

    finalnum = 0
    for key, value in new_contents.items():
        print(key)
        print(type(key))
        print(value)
        print(type(value))
        dumpstrs = json.dumps(value, indent=4, ensure_ascii=False)
        restrs = "module.exports = " + dumpstrs
        with open(os.path.join(out_fold, key), 'w', encoding="utf-8") as fp:
            fp.write(dequota(restrs))
        finalnum += len(new_contents[key])
        break
    print("total length is: {}".format(finalnum))
    exit()
    print("end")


def test():
    import demjson
    import json
    ff = open(os.path.join("e:\\", "project", "ht_data", "prod", "question_podupojiao.js"), encoding="utf-8")
    contents = ff.readlines()
    getstrs = "".join(contents).replace("module.exports =", "")
    objres = demjson.decode(getstrs)
    strsres = json.dumps(objres, ensure_ascii=False)  # From Python to JSON


if __name__ == "__main__":
    test()
    exit()
    main()
