import argparse
import simplejson
import os


def parseArgs(args):
    parser = argparse.ArgumentParser()
    globalArgs = parser.add_argument_group('Global options')
    globalArgs.add_argument('--modelname', default=None, nargs='?',
                            choices=["full", "one", "one_y", "one_space", "one_attent", "one_attent60"])
    globalArgs.add_argument('--learnrate', type=float, nargs='?', default=None)
    globalArgs.add_argument('--globalstep', type=int, nargs='?', default=None)
    globalArgs.add_argument('--dropout', type=float, nargs='?', default=None)
    globalArgs.add_argument('--normal', type=float, nargs='?', default=None)
    globalArgs.add_argument('--sub_fix', type=str, nargs='?', default=None)
    return parser.parse_args(args)


class bcolors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_paras(args):
    # 1. 命令行
    argspar = parseArgs(args)
    # 2. 模型参数赋值
    parafile = os.path.join(os.getcwd(), "config", "paraf.json")
    config = {}
    if argspar.modelname is None:
        hpara = simplejson.load(open(parafile))
        config["modelname"] = hpara["model"]["modelname"]
        config["sub_fix"] = hpara["model"]["sub_fix"]
        config["tailname"] = "%s-%s" % (config["modelname"], config["sub_fix"])
    else:
        config["modelname"] = argspar.modelname
        if argspar.sub_fix is None:
            config["tailname"] = "%s-" % (argspar.modelname)
        else:
            config["sub_fix"] = argspar.sub_fix
            config["tailname"] = "%s-%s" % (argspar.modelname, argspar.sub_fix)
        parafile = "para_%s.json" % (config["tailname"])
        hpara = simplejson.load(open(parafile))
        # if argspar.sub_fix is None:
        #     config["sub_fix"] = hpara["model"]["sub_fix"]

    if argspar.learnrate is None:
        config["learn_rate"] = hpara["env"]["learn_rate"]
    else:
        config["learn_rate"] = argspar.learnrate

    if argspar.globalstep is None:
        globalstep = hpara["model"]["globalstep"]
    else:
        globalstep = argspar.globalstep
    if argspar.dropout is None:
        config["dropout"] = hpara["model"]["dropout"]
    else:
        config["dropout"] = argspar.dropout
    if argspar.normal is None:
        config["normal"] = hpara["model"]["normal"]
    else:
        config["normal"] = argspar.normal

    config["scope"] = hpara["env"]["scope"]
    config["inputdim"] = hpara["env"]["inputdim"]
    config["outspace"] = hpara["env"]["outspace"]
    config["single_num"] = hpara["env"]["single_num"]
    config["modelfile"] = hpara["model"]["file"]

    # news paras
    # paras print
    print()
    print("**********************************************************")
    print("parafile:", parafile)
    config["process"] = hpara["process"]
    keyslevel1 = ["scrap_data", "data_filter", "get_chara", "get_learn", "back_test", "trade_fun"]
    for i1 in keyslevel1:
        if i1 not in list(hpara):
            raise ValueError("do not have key {} in parajson.".format(i1))
        if "usesig" not in hpara[i1]:
            raise ValueError("do not have usesig {} in parajson.".format(i1))
        config[i1] = hpara[i1]
        if hpara[i1]["usesig"] == 1:
            print("module %s used.", i1)
        elif hpara[i1]["usesig"] == 0:
            print("module %s not used.", i1)
        else:
            raise ValueError("usesig value is {}. illegal.".format(hpara[i1]["usesig"]))
    print()
    print("modelname:", config["modelname"])
    print("tailname:", config["tailname"])
    print("learn_rate:", config["learn_rate"])
    print("dropout:", config["dropout"])
    print("**********************************************************")
    print()

    # 概率小于此值时，取随机值不使用模型
    epsilon = hpara["env"]["epsilon"]
    epoch = hpara["env"]["epoch"]
    start_date = hpara["env"]["start_date"]
    end_date = hpara["env"]["end_date"]
    sudden_death = hpara["env"]["sudden_death"]
    scope = hpara["env"]["scope"]
    max_memory = hpara["env"]["max_memory"]
    batch_size = hpara["env"]["batch_size"]
    discount = hpara["env"]["discount"]
    return config
