import os
import json
import six
import utils
from utils import json_to_string
from utils.cmd_paras_check import ConfigException
import copy
from config.model_config import func_name


class ParseFlow(object):
    """
    # 10. 管线拓扑关系，有向。
    # 10.1 基本元素
    # 输入2维，输出1维，添加实例时按全局数值添加序号。-1为自定义，实例化时应指定大小，默认跟输入或输出匹配。
    class_instance = {
      "分词":{
        "shape":[2,1],
        "basecode":[[1,2],[3]],
        "descripe":[["前句"，"后句"],["标签"]]
      },
    }
    instance_list = {
      "inform":[5],
      "outform":[42,43,15],
      "分词__i1":{
        "inout":[[22,23],[5]]
        "paras":{
          "chara":300,
          "kvalue":5
        }
      },
      "分词__i2":{
        "inout":[[32,33],[15]]
        "paras":{
          "chara":300,
          "kvalue":5
        }
      },
      "分类__i1":{
        "inout":[[42,43],[25]]
        "paras":{
          "chara":300,
          "kvalue":5
        }
      },
      "分类__i2":{
        "inout":[[52,53],[35]]
        "paras":{
          "chara":300,
          "kvalue":5
        }
      },
      "没有输出":{
        "默认为最后实例的输出"
      },
    }
    link_list = [
      [35,24],
      [1,2]
    ]
    # 10.2 子结构展开
    结果保存模型后，该link_list 和instance_list 也保存，具体模型不能有-1编号。并重新编号作为子模块的输入输出序号。
    # 10.3 驱动结构
    遍历输入-->link到被动节点-->添加到已存在输入列表-->遍历被连节点的实例-->对比输入要素是否完备-->选出合并实例分到运行列表，剩下的分到待运行列表
                   ^                     v
    -->遍历运行列表,成功后移除该项   如果没有下级link-->结束。
    
    碰到新实例自动输出编号保存数据到列表，供调用
    """

    def __init__(self, env_vars=None, cmdline_args=None):
        """
        遍历输入-->link到被动节点-->添加到已存在输入列表-->遍历被连节点的实例-->对比输入要素是否完备-->选出合并实例分到运行列表，剩下的分到待运行列表
                       ^                     v
        -->遍历运行列表,成功后移除该项   如果没有下级link-->结束。   
        """
        # 1. 读取3个文件
        # 1. 子结构展开
        class_instance = {
            "分词": {
                "shape": [2, 1],
                "basecode": [[1, 2], [3]],
                "descripe": [["前句输入", "后句输入"], ["标签输出"]]
            }
        }
        instance_list = {
            "inform": [5, 42, 43, 15],
            "分词__i1": {
                "inout": [[22, 23], [5]],
                "paras": {
                    "chara": 300,
                    "kvalue": 5
                }
            },
            "分词__i2": {
                "inout": [[32, 33], [15]],
                "paras": {
                    "chara": 300,
                    "kvalue": 5
                }
            },
            "分类__i1": {
                "inout": [[42, 43], [25]],
                "paras": {
                    "chara": 300,
                    "kvalue": 5
                }
            },
            "分类__i2": {
                "inout": [[52, 53], [35]],
                "paras": {
                    "chara": 300,
                    "kvalue": 5
                }
            },
            "没有输出": {
                "默认为最后实例的输出"
            },
        }
        link_list = [
            [35, 24],
            [1, 2]
        ]
        # 2.1 初始化wait_list
        wait_list = {}
        # 分到待运行列表
        for i1 in instance_list:
            if i1.split("__")[0] in class_instance:
                wait_list[i1] = instance_list[i1]
        # 2.2 判断data_list,ready_list
        ready_list = []
        data_list = {}
        for i1 in wait_list["inform"]:
            data_list[str(i1)] = []
        while True:
            for i1 in wait_list:
                havsig = 1
                # 对比输入要素是否完备
                for i2 in wait_list[i1]["inout"][0]:
                    if wait_list[i1]["inout"][0][i2] not in data_list:
                        havsig = 0
                        break
                # 选出合并实例分到运行列表
                if 1 == havsig:
                    ready_list.append(i1)
            if 0 == len(ready_list):
                return ""
            # 2.3 执行ready_list模块。
            for i1 in ready_list:
                # 2.3.1 输入参数，出入位置信息 和 data_list，返回 新加data_list 和 筛选的参数。
                [datas, perform_paras] = func_name[i1](wait_list[i1], data_list)
                wait_list.__delattr__(i1)
                # 2.3.2 link到被动节点
                for i2 in link_list:
                    # 遍历被连节点的实例，添加到已存在输入列表。
                    if i2[0] in wait_list[i1]["inout"][1]:
                        data_list[str(i2[1])] = datas
        try:
            file_config = utils.read_json_file(fileenv)
        except ValueError as e:
            pass
        for key, value in self.items():
            setattr(self, key, value)

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
                abs_path_config[key] = os.path.join(os.getcwd(), abs_path_config[key])
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
