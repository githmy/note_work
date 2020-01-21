import pandas as pd
import numpy as np
import re



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
    from utils.connect_mongo import MongoDB
    quire_list = ["examples", "questions"]
    config_new = {
        'host': "127.0.0.1",
        'port': 27017,
        'database': "thinking2-test",
        'col': "examples",
        'charset': 'utf8mb4',  # 支持1-4个字节字符
    }
    ins_new = MongoDB(config_new)
    res_map_new = ins_new.exec_require(quire_list)
    contente = iterjson(list(res_map_new["examples"]))
    print(len(contente))
    contentq = iterjson(list(res_map_new["questions"]))
    print(len(contentq))
    print(contente + contentq)
    print(len(contente + contentq))
    print(len(set(contente + contentq)))
    pass


if __name__ == "__main__":
    main()
