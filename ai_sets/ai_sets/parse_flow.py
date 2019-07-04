import os
import json
import six
import logging
import utils
from utils import json_to_string
from utils.cmd_paras_check import ConfigException
from utils.get_files import read_csv_data, write_csv_data
from utils.path_tool import makesurepath
import copy
import requests
import numpy as np
from config.model_config import func_name
from ai_sets.tool_db import Datadb
from ai_sets.tool_db import RDdb

logger = logging.getLogger(__name__)


def do_pipline_in_worker(envjson, parajson, require_values, config):
    print("do_pipline_in_worker")
    pf = ParseFlow(envjson, parajson, require_values, config, Datadb(config), RDdb(config))
    pf.parseflow()
    print("end_pipline_in_worker")


def do_single_in_worker(config, outjson):
    print("do_single_in_worker")
    connnect = Datadb(config)
    envinfo = connnect.func_env()
    envinfo = {i["func_name"]: i["env_id"] for i in envinfo}
    funcclass = outjson["func_instance"].split("__")[0]
    if envinfo[funcclass] == config["env_id"]:
        parain = (outjson["paras"], outjson["env"])
        return func_name[funcclass](outjson["data"], parain)
    else:
        print("check env error, in %s" % funcclass)
        return None
    print("end_single_in_worker")


class ParseFlow(object):
    """
    # 10. 管线拓扑关系，有向。
    # 10.1 基本元素
    # 输入2维，输出1维，添加实例时按全局数值添加序号。-1为自定义，实例化时应指定大小，默认跟输入或输出匹配。
    # 10.2 子结构展开
    结果保存模型后，该link_list 和instance_list 也保存，具体模型不能有-1编号。并重新编号作为子模块的输入输出序号。
    # 10.3 驱动结构
    遍历输入-->link到被动节点-->添加到已存在输入列表-->遍历被连节点的实例-->对比输入要素是否完备-->选出合并实例分到运行列表，剩下的分到待运行列表
                   ^                     v
    -->遍历运行列表,成功后移除该项   如果没有下级link-->结束。
    
    碰到新实例自动输出编号保存数据到列表，供调用
    """

    def __init__(self, envjson=None, instance_config=None, require_values=None, config=None, connect=None,
                 rediscon=None):
        """
        遍历输入-->link到被动节点-->添加到已存在输入列表-->遍历被连节点的实例-->对比输入要素是否完备-->选出合并实例分到运行列表，剩下的分到待运行列表
                       ^                     v
        -->遍历运行列表,成功后移除该项   如果没有下级link-->结束。   
        """
        # 1. 初始数组变量
        self._instance_config = instance_config
        self._envjson = envjson
        self._connect = connect
        self._rediscon = rediscon
        self._require_values = require_values
        print(self._instance_config)
        self._outnode_tuple = tuple(self._instance_config["outnode_file"].keys())
        self._link_tuple = tuple(self._instance_config["link_list"])
        # 2.1 初始化wait_list
        basictree = ['illustration', 'innode_file', 'outnode_file', 'link_list']
        self._instance_tuple = [i for i in self._instance_config]
        [self._instance_tuple.remove(i) for i in basictree]
        self._instance_tuple = tuple(self._instance_tuple)
        print("models wait to run is : ", self._instance_tuple)
        # ok,running,fail,wait
        self._status_list = {i: "wait" for i in self._instance_tuple}
        self._perform_list = {i: None for i in self._instance_tuple}
        self._wait_list = [i for i in self._status_list]
        self._ready_list = []
        self._alldatas = {}

    def _readinnode(self):
        # 1. 路径生成
        print("reading files ...")
        rootpath = "."
        # rootpath = os.getcwd()
        for i in self._envjson["paths"]["rootpath"]:
            rootpath = os.path.join(rootpath, i)
        response_in_dir = rootpath
        if self._require_values["usertype"] == "general":
            # 普通用户需要+id
            for i in self._envjson["paths"]["sysknowunitin"]:
                response_in_dir = os.path.join(response_in_dir, i)
            response_in_dir = os.path.join(response_in_dir, str(self._require_values["userid"]))
        elif self._require_values["usertype"] == "admin":
            for i in self._envjson["paths"]["sysknowunit"]:
                response_in_dir = os.path.join(response_in_dir, i)
        else:
            raise "usertype unknow."
        response_in_dir = os.path.join(response_in_dir, str(self._require_values["project"]),
                                       str(self._require_values["branch"]))
        makesurepath(response_in_dir)
        # 2. 数据加载
        for i1 in self._instance_config["innode_file"]:
            try:
                self._alldatas[i1] = tuple(
                    np.array(read_csv_data(os.path.join(response_in_dir, self._instance_config["innode_file"][i1][0]))[
                                 self._instance_config["innode_file"][i1][1]]))
            except Exception as e:
                print("error when reading ", os.path.join(response_in_dir, self._instance_config["innode_file"][i1][0]))

    def _writeoutnode(self):
        # 1. 路径生成
        print("writing files ...")
        rootpath = "."
        # rootpath = os.getcwd()
        for i in self._envjson["paths"]["rootpath"]:
            rootpath = os.path.join(rootpath, i)
        response_out_dir = rootpath
        for i in self._envjson["paths"]["sysknowunitout"]:
            response_out_dir = os.path.join(response_out_dir, i)
        response_in_dir = os.path.join(response_out_dir, str(self._require_values["userid"]))
        response_out_dir = os.path.join(response_out_dir, str(self._require_values["project"]),
                                        str(self._require_values["branch"]))
        makesurepath(response_out_dir)
        # 2. 数据写入
        for i1 in self._outnode_tuple:
            write_csv_data(self._alldatas[i1],
                           os.path.join(response_out_dir, self._instance_config["outnode_file"][i1][0]),
                           self._instance_config["outnode_file"][i1][1])

    def _pipe_flow(self):
        # 1. 环境变量
        envinfo = self._connect.func_env()
        envinfo = {i["func_name"]: i["env_id"] for i in envinfo}
        # 2. 待运行分到准备好的列表
        print("beginning pipline ...")
        # 2.0 link到被动节点
        tmplist = [i for i in self._alldatas]
        for i1 in tmplist:
            for i2 in self._link_tuple:
                if str(i2[0]) == str(i1):
                    self._alldatas[str(i2[1])] = self._alldatas[i1]

        self._connect.write_status(json.dumps(self._status_list, ensure_ascii=False), self._require_values)
        print(self._alldatas.keys())
        while True:
            # 2.1 便利查找待运行的条件
            self._ready_list = []
            for i1 in self._wait_list:
                havsig = 1
                # 对比输入要素是否完备
                for i2 in self._instance_config[i1]["inout"][0]:
                    if str(i2) not in self._alldatas:
                        havsig = 0
                        break
                # 选出合并实例分到运行列表
                if 1 == havsig:
                    self._ready_list.append(i1)
            if 0 == len(self._ready_list):
                # 没有可运行的示例，结束
                return "已经没有可运行的实例。"
            # 2.2 执行ready_list模块。
            print("batch ready list:", self._ready_list)
            for i1 in self._ready_list:
                print("running :", i1)
                # 2.2.1 输入参数，出入位置信息 和 data_list，返回 新加data_list 和 筛选的参数。
                tmpindata = [self._alldatas[str(i2)] for i2 in self._instance_config[i1]["inout"][0]]
                self._status_list[i1] = "running"
                self._connect.write_status(json.dumps(self._status_list, ensure_ascii=False), self._require_values)
                # 2.2.2 环境判断。相符则继续运行，不符查找redis里符合的ip_port
                if envinfo[i1.split("__")[0]] == self._envjson["env_id"]:
                    parain = (self._instance_config[i1]["paras"], self._envjson.as_dict())
                    [datas, perform_paras] = func_name[i1.split("__")[0]](tmpindata, parain)
                else:
                    # 2.2.2.1 查看可用资源
                    maxtry = 10
                    usableenv = ""
                    for i2 in range(maxtry):
                        keysss = self._rediscon.show_env()
                        if maxtry - 1 == i2:
                            raise Exception("fitted enviroment busy. please try again later .. ")
                        for i3 in keysss:
                            if self._rediscon.get_env(i3).decode('utf-8') == str(envinfo[i1.split("__")[0]]):
                                try:
                                    self._rediscon.del_env(i3)
                                    usableenv = i3.decode('utf-8')
                                    break
                                except Exception as e:
                                    pass
                        if usableenv != "":
                            break
                    # 2.2.2.2 访问资源
                    url = "http://" + usableenv.split("__")[1] + "/func_single"
                    tmpjson = json.dumps({'func_instance': i1, 'data': tmpindata,
                                          'paras': self._instance_config[i1]["paras"],
                                          'env': self._envjson.as_dict(), 'sess_json': self._require_values},
                                         ensure_ascii=False)
                    print("redirect to " + url)
                    res = requests.post(url=url, data=tmpjson.encode("utf-8"),
                                        headers={'Content-Type': 'application/x-www-form-urlencoded'})
                    # timeout=(None, None))
                    res = json.loads(res.text)
                    datas, perform_paras = res[0], res[1]
                    # 2.2.2.3 数据接收
                # 2.2.3 成功标记
                # print(perform_paras)
                self._status_list[i1] = "ok"
                self._connect.write_status(json.dumps(self._status_list, ensure_ascii=False), self._require_values)
                self._perform_list[i1] = perform_paras
                self._connect.write_perform(json.dumps(self._perform_list, ensure_ascii=False), self._require_values)
                self._wait_list.remove(i1)
                # 2.2.4 link到被动节点
                for i2 in self._link_tuple:
                    # 遍历被连节点的实例，添加到已存在输入列表。
                    for key, val in enumerate(self._instance_config[i1]["inout"][1]):
                        if str(i2[0]) == str(val):
                            self._alldatas[str(i2[1])] = tuple(datas[key])
            # 2.3 输出是否结束，忽略不用的计算。
            print(self._alldatas.keys())
            endsig = 1
            for i1 in self._outnode_tuple:
                if i1 not in self._alldatas:
                    endsig = 0
                    break
            if endsig == 1:
                return "结束：已经没有可运行的实例。"

    def parseflow(self):
        # 1. 输入node
        self._readinnode()

        self._pipe_flow()
        # 3. 数据输出
        self._writeoutnode()

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        self.override(state)

    def items(self):
        return list(self.__dict__.items())

    def as_dict(self):
        return dict(list(self.items()))

    def view(self):
        return json_to_string(self.__dict__, indent=4)

    def split_arg(self, config, arg_name):
        if arg_name in config and isinstance(config[arg_name], six.string_types):
            config[arg_name] = config[arg_name].split(",")
        return config

    def split_pipeline(self, config):
        if "pipeline" in config and isinstance(config["pipeline"], six.string_types):
            config = self.split_arg(config, "pipeline")
            if "pipeline" in config and len(config["pipeline"]) == 1:
                config["pipeline"] = config["pipeline"][0]
        return config

    def create_cmdline_config(self, cmdline_args):
        cmdline_config = {k: v
                          for k, v in list(cmdline_args.items())
                          if v is not None}
        cmdline_config = self.split_pipeline(cmdline_config)
        cmdline_config = self.split_arg(cmdline_config, "duckling_dimensions")
        return cmdline_config

    def create_env_config(self, env_vars):
        keys = [key for key in env_vars.keys() if "RASA_" in key]
        env_config = {key.split('RASA_')[1].lower(): env_vars[key] for key in keys}
        env_config = self.split_pipeline(env_config)
        env_config = self.split_arg(env_config, "duckling_dimensions")
        return env_config

    def make_paths_absolute(self, config, keys):
        abs_path_config = dict(config)
        for key in keys:
            if key in abs_path_config and abs_path_config[key] is not None and not os.path.isabs(abs_path_config[key]):
                abs_path_config[key] = os.path.join("./", abs_path_config[key])
                # abs_path_config[key] = os.path.join(os.getcwd(), abs_path_config[key])
        return abs_path_config

    # noinspection PyCompatibility
    def make_unicode(self, config):
        if six.PY2:
            # Sometimes (depending on the source of the config value) an argument will be str instead of unicode
            # to unify that and ease further usage of the config, we convert everything to unicode
            for k, v in config.items():
                if type(v) is str:
                    config[k] = unicode(v, "utf-8")
        return config

    def override(self, config):
        abs_path_config = self.make_unicode(self.make_paths_absolute(config, ["path", "response_log"]))
        self.__dict__.update(abs_path_config)
