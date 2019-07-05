# -*- coding: utf-8 -*-
from preprocess_pack.preprocess_nlp import preprocess_clean_nlp
from preprocess_pack.get_basic import get_basic
from utils.cmd_paras_check import create_argparser
from utils.get_files import read_csv_data
from predict_pack.predict_nlp import predict_nlp
from config import rootpath
import pandas as pd

if __name__ == '__main__':
    # Running as standalone python application

    # 1. 清洗数据
    arg_parser = create_argparser()
    # 1.1 命令行检验
    args = arg_parser.parse_args()
    # 1.2 路径 文件 json 获取
    allpaths, allfiles, standjson = get_basic(args, rootpath)
    # 1.3 获取数据
    predict_fkey = "predict_file"
    ori_data = read_csv_data(allfiles[predict_fkey])
    # ori_data = pd.read_csv(allfiles[predict_fkey], header=0, encoding="GBK")  # , dtype=str
    # 1.4 清理函数
    pddata = preprocess_clean_nlp(allpaths, allfiles, ori_data)

    # 2. 训练数据
    # 2.1 得到训练结果
    labeldata = predict_nlp(standjson, pddata)

    # 3. 文件输出
    predict_res_fkey = "predict_res_file"
    datares = pd.DataFrame(labeldata)
    datares.to_csv(allfiles[predict_res_fkey], encoding='utf-8', index=False)
    print("end")
