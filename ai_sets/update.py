import os
import logging
from gene_conf_pack.gene_config_nlp import process_tain_nlp
from preprocess_pack.get_basic import get_basic
from preprocess_pack.preprocess_nlp import preprocess_clean_nlp
from utils.cmd_paras_check import create_argparser
from utils.get_files import read_csv_data
from ai_sets.tool_db import Datadb
from ai_sets.config import AisetsConfig
import json
import simplejson
import re


def get_jsons():
    # 1. 读各个文件的信息
    # rootpath = os.getcwd()
    rootpath = "."
    alllist = os.listdir(os.path.join(rootpath, "pub_modules"))
    res = [re.match(r'.*py$', i) for i in alllist]
    mapobj = {}
    target_content = ""
    for i in res:
        if i is not None:
            filename = os.path.join(rootpath, "pub_modules", i.group())
            print(filename)
            target_content += "from pub_modules." + i.group().split(".py")[0] + " import *\n"
            with open(filename, 'r', encoding='UTF-8') as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
                startsig = 0
                for i2, line in enumerate(lines):
                    # 2. 找到尾
                    if re.match(r'\s*\"\"\".*', line):
                        startsig = 0
                    # 1. 内容处理
                    if startsig == 1:
                        mapobj[result] += line.strip()
                        # print(mapobj[result])
                    # 1. 找到头
                    if re.match(r'\s*\"\"\"func_add.*', line):
                        startsig = 1
                        result = re.findall("^def (.*)\(.*", lines[i2 - 1])[0]
                        mapobj[result] = ""
    # 2. 写入性能信息，配置文件
    filename = os.path.join(rootpath, "config", "model_config.py")
    with open(filename, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        startsig = 0
        for line in lines:
            # 1. 头文件，忽略容易覆盖目录
            if re.match(r'^from ', line):
                if re.match(r'^from ai_sets.openapi ', line):
                    pass
                else:
                    continue
            # 2. 函数名部分
            # 2.1 找到尾
            if re.match(r'^}.*', line):
                startsig = 0
                continue
            # 2.2 内容处理
            if startsig == 1:
                continue
            # 2.3 找到头
            if re.match(r'^func_name.*', line):
                startsig = 1
                continue
            target_content += line
    target_content += "func_name = {\n"
    jsonlist = []
    for i2 in mapobj:
        tmpjson = simplejson.loads("{" + mapobj[i2] + "}")
        jsonlist.append(tmpjson)
        namekey = [i3 for i3 in tmpjson]
        target_content += '    "' + namekey[0] + '": ' + tmpjson[namekey[0]]["func_name"] + ',\n'
    target_content += "}\n"
    with open(filename, 'w', encoding='UTF-8') as f:
        lines = f.write(target_content)
    # 3. 根据信息，数据库操作
    aisets_config = AisetsConfig(os.environ, None)
    dbt = Datadb(aisets_config)
    for i2 in jsonlist:
        namekey = [i3 for i3 in i2][0]
        # 3.1 判断是否存在，不存在创建
        havesize = len(dbt.check_func_info(namekey))
        if havesize == 0:
            dbt.insert_func_info(namekey, i2[namekey]["env_id"], simplejson.dumps(i2[namekey], ensure_ascii=False))
        else:
            dbt.update_func_info(namekey, i2[namekey]["env_id"], simplejson.dumps(i2[namekey], ensure_ascii=False))


if __name__ == '__main__':
    # Running as standalone python application
    # 1. 清洗数据
    get_jsons()
