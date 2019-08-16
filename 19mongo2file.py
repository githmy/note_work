import os
import json
import pandas as pd
from utils.connect_mongo import MongoDB


def main():
    # baspath = os.path.join("..", "ht_data", "prod")
    baspath = os.path.join("..", "data")
    config = {
        'host': "192.168.1.52",
        'port': 27017,
        # 'user': "root",
        # 'password': "root",
        'database': "thinking2ht",
        'col': "examples",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
    }
    ins = MongoDB(config)
    res = ins.exec_require()
    EZKB = []
    SHZK = []
    RJ090A = []
    RJ090B = []
    RJ080A = []
    RJ080B = []
    RJ070A = []
    RJ070B = []
    for i1 in res:
        del i1["__v"]
        if "solutions" in i1:
            for i2 in i1["solutions"]:
                del i2["_id"]
                if "steps" in i2:
                    for i3 in i2["steps"]:
                        del i3["_id"]
        if "steps" in i1:
            for i2 in i1["steps"]:
                del i2["_id"]
        # print(i1)
        # json.dumps(i1)
        if i1["_id"].startswith("EZKB"):
            EZKB.append(i1)
        elif i1["_id"].startswith("SHZK"):
            SHZK.append(i1)
        elif i1["_id"].startswith("RJ070A"):
            RJ070A.append(i1)
        elif i1["_id"].startswith("RJ070B"):
            RJ070B.append(i1)
        elif i1["_id"].startswith("RJ080A"):
            RJ080A.append(i1)
        elif i1["_id"].startswith("RJ080B"):
            RJ080B.append(i1)
        elif i1["_id"].startswith("RJ090A"):
            RJ090A.append(i1)
        elif i1["_id"].startswith("RJ090B"):
            RJ090B.append(i1)
        else:
            print("ERROR:", i1)
    dumpstrs = json.dumps(SHZK, indent=4, ensure_ascii=False)
    restrs = "module.exports = " + dumpstrs
    with open(os.path.join(baspath, 'example_zhongkao.js'), 'w', encoding="utf-8") as fp:
        fp.write(restrs)
    dumpstrs = json.dumps(EZKB, indent=4, ensure_ascii=False)
    restrs = "module.exports = " + dumpstrs
    with open(os.path.join(baspath, 'example_zkb.js'), 'w', encoding="utf-8") as fp:
        fp.write(restrs)
    dumpstrs = json.dumps(RJ070A, indent=4, ensure_ascii=False)
    restrs = "module.exports = " + dumpstrs
    with open(os.path.join(baspath, 'example_class7A.js'), 'w', encoding="utf-8") as fp:
        fp.write(restrs)
    dumpstrs = json.dumps(RJ070B, indent=4, ensure_ascii=False)
    restrs = "module.exports = " + dumpstrs
    with open(os.path.join(baspath, 'example_class7B.js'), 'w', encoding="utf-8") as fp:
        fp.write(restrs)
    dumpstrs = json.dumps(RJ080A, indent=4, ensure_ascii=False)
    restrs = "module.exports = " + dumpstrs
    with open(os.path.join(baspath, 'example_class8A.js'), 'w', encoding="utf-8") as fp:
        fp.write(restrs)
    dumpstrs = json.dumps(RJ080B, indent=4, ensure_ascii=False)
    restrs = "module.exports = " + dumpstrs
    with open(os.path.join(baspath, 'example_class8B.js'), 'w', encoding="utf-8") as fp:
        fp.write(restrs)
    dumpstrs = json.dumps(RJ090A, indent=4, ensure_ascii=False)
    restrs = "module.exports = " + dumpstrs
    with open(os.path.join(baspath, 'example_class9A.js'), 'w', encoding="utf-8") as fp:
        fp.write(restrs)
    dumpstrs = json.dumps(RJ090B, indent=4, ensure_ascii=False)
    restrs = "module.exports = " + dumpstrs
    with open(os.path.join(baspath, 'example_class9B.js'), 'w', encoding="utf-8") as fp:
        fp.write(restrs)
    print("end")


if __name__ == "__main__":
    # 视频批量转化
    main()
