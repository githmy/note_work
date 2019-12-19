import os
import re
import json
import operator
import pandas as pd
from utils.connect_mongo import MongoDB
from utils.path_tool import makesurepath
import demjson
import json

outpath = os.path.join("..", "data", "mongofiles")


def get_file_js():
    ori_fold = os.path.join("..", "ht_data", "prod")
    # 3. 获取老文件信息
    file_names = os.listdir(ori_fold)
    # file_names.remove('example_class7A.js')
    # file_names.remove('example_class7B.js')
    # file_names.remove('example_class8A.js')
    # file_names.remove('example_class8B.js')
    # file_names.remove('example_class9A.js')
    # file_names.remove('example_class9B.js')
    # file_names.remove('example_cofm.js')
    # file_names.remove('example_zhongkao.js')
    # file_names.remove('question_0820.js')
    # file_names.remove('question_class5B.js')
    # file_names.remove('question_podupojiao.js')
    # file_names.remove('question_fcbds.js')
    # file_names.remove('question_hanshu.js')
    # file_names.remove('question_sdzc.js')
    # file_names.remove('question_sjfx.js')
    # file_names.remove('question_thinking.js')
    # file_names.remove('question_tjgl.js')
    # #
    # file_names.remove('assignment_htexam.js')
    # file_names.remove('example_04A.js')
    # file_names.remove('example_07A.js')
    # file_names.remove('example_08A.js')
    # file_names.remove('example_09A.js')
    # file_names.remove('example_09B.js')
    # file_names.remove('example_class6.js')
    # file_names.remove('example_class6789.js')
    # file_names.remove('example_demo.js')
    # file_names.remove('example_zkb.js')
    # file_names.remove('question_0903.js')
    # file_names.remove('question_4A.js')
    # file_names.remove('question_class3A.js')
    # file_names.remove('question_class3B.js')
    # file_names.remove('question_class4A.js')
    # file_names.remove('question_class4B.js')
    # file_names.remove('question_class5A.js')
    # file_names.remove('question_class6.js')
    # file_names.remove('question_cofm.js')
    # file_names.remove('question_demo.js')
    # file_names.remove('question_enteranceExam.js')
    # file_names.remove('question_exam_ht.js')
    # file_names.remove('question_shuyshi.js')
    # file_names.remove('example_rttat.js')
    # file_names.remove('question_enteranceExamB.js')
    # file_names.remove('question_rttat.js')
    # file_names.remove('question_txdbh.js')
    # file_names.remove('question_txdxz.js')
    file_names = ["question_thinking.js", "question_lesson_class9A.js", "question_lesson_class9B.js"]
    print(file_names)
    # 5. 生成老文件
    new_contents = []
    for token in file_names:
        print(token)
        with open(os.path.join(ori_fold, token), 'rt', encoding="utf8") as f:
            getstrs = "".join(f.readlines()).replace("module.exports =", "")
            # print(type(demjson.decode(getstrs)))
            new_contents.append(demjson.decode(getstrs))
            # new_contents += json.loads(getstrs, encoding="utf-8")
    return new_contents


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


def find_keys_injs(keys):
    new_contents = get_file_js()
    id3s = []
    for onefile in new_contents:
        id3s.append([oneline['_id'] for oneline in onefile])
    save_ids = []
    save_ids.append([i1 for i1 in id3s[1] if i1 not in id3s[0]])
    save_ids.append([i1 for i1 in id3s[2] if i1 not in id3s[0]])
    new_contents[1] = [i1 for i1 in new_contents[1] if i1["_id"] in save_ids[0]]
    new_contents[2] = [i1 for i1 in new_contents[2] if i1["_id"] in save_ids[1]]
    parafile = os.path.join("..", "ht_data", "prod", "z_question_lesson_class9A.js")
    json.dump(new_contents[1], open(parafile, mode='w', encoding="utf-8"), ensure_ascii=False, indent=2)
    parafile = os.path.join("..", "ht_data", "prod", "z_question_lesson_class9B.js")
    json.dump(new_contents[2], open(parafile, mode='w', encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    keys = ["比例的性质", "百分比", "射影定理"]
    find_keys_injs(keys)
    exit()
    main()
