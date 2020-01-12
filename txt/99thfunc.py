import pandas as pd
import numpy as np
import json
import os
import itertools

propertyfile = os.path.join(".", "99propertydic.json")
objfile = os.path.join(".", "99objlistf.json")


def iter4property(obj, propertylist, desclist):
    if "tuple" in obj:
        for one in obj["tuple"]:
            propertylist.append([one[0]["name"], one[1]["name"]])
            desclist.append([one[0]["desc"], one[1]["desc"]])
            if "tuple" in one[0]:
                iter4property(one[0], propertylist, desclist)
            if "tuple" in one[1]:
                iter4property(one[1], propertylist, desclist)
    else:
        propertylist.append([obj[0]["name"], obj[1]["name"]])
        desclist.append([obj[0]["desc"], obj[1]["desc"]])


def property2obj():
    # 根据属性字典生成对象列表,如果id已存在不变化和描述相同，
    # 1. 读取属性字典
    with open(propertyfile, "rt", encoding="utf-8") as f:
        jsonobj = json.load(f)
    print(jsonobj)
    # 2. 遍历属性，生成列表
    propertylist = []
    desclist = []
    iter4property(jsonobj, propertylist, desclist)
    print(propertylist)
    print(desclist)
    objlist = []
    propertylenth = len(propertylist)
    # if os.path.exists(objfile):
    if not os.path.exists(objfile):
        for i1 in itertools.product('01', repeat=propertylenth):
            tmproperlist = []
            tmpdesclist = []
            for id2, i2 in enumerate(i1):
                if i2 == "0":
                    tmproperlist.append(propertylist[id2][1])
                    tmpdesclist.append(desclist[id2][1])
                elif i2 == "1":
                    tmproperlist.append(propertylist[id2][0])
                    tmpdesclist.append(desclist[id2][0])
                elif i2 == "2":
                    tmproperlist.append(" ".join(propertylist[id2]))
                    tmpdesclist.append("$".join(desclist[id2]))
                else:
                    pass
            tmpobj = {
                "id": "".join(i1),
                "name": "".join(i1),
                "properties": ",".join(tmproperlist),
                "desc": "#".join(tmpdesclist)
            }
            objlist.append(tmpobj)
        # 3. 写入对象列表
        with open(objfile, "wt", encoding="utf-8") as f:
            json.dump(objlist, f, ensure_ascii=False, indent=2)


def main():
    property2obj()
    exit()
    # jsonstr = json.dumps(tmpjson, ensure_ascii=False, indent=2)
    # # 1. 读取属性字典
    # jsonobj = json.load(open(propertyfile), encoding="utf-8")
    # 2. 读取对象列表
    # 3. 读取对象关系
    # 4. 读取方法列表
    # 根据属性字典生成对象列表
    pass


if __name__ == '__main__':
    main()
