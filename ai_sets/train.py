# -*- coding: utf-8 -*-
import logging
from gene_conf_pack.gene_config_nlp import process_tain_nlp
from preprocess_pack.get_basic import get_basic
from preprocess_pack.preprocess_nlp import preprocess_clean_nlp
from utils.cmd_paras_check import create_argparser
from utils.get_files import read_csv_data
from config import rootpath
import json

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Running as standalone python application
    # 1. 清洗数据
    arg_parser = create_argparser()
    # 1.1 命令行检验
    args = arg_parser.parse_args()
    # 1.2 路径 文件 json 获取
    allpaths, allfiles, standjson = get_basic(args, rootpath)
    # 1.3 获取数据
    train_fkey = "train_file"
    ori_data = read_csv_data(allfiles[train_fkey])
    # ori_data = pd.read_csv(allfiles[train_fkey], header=0, encoding="GBK")  # , dtype=str
    # 1.4 清理函数
    pddata = preprocess_clean_nlp(allpaths, allfiles, ori_data)

    # 2. 训练数据
    # 2.1 得到训练结果
    resjson = process_tain_nlp(allpaths, standjson, pddata)

    # 3. 覆盖json
    json_fkey = "json_file"
    with open(allfiles[json_fkey], "w", encoding="utf-8") as f:
        json.dump(resjson, f, ensure_ascii=False)
    print("end")
