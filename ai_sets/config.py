import os
import json
import six
import utils
from utils import json_to_string
from utils.cmd_paras_check import ConfigException
import copy

__version__ = '0.12'
rootpath = os.path.join("..", "nocode")
# serverlogpath = os.path.join(rootpath, "logs")
DEFAULT_PROJECT_NAME = "default"

DEFAULT_CONFIG_LOCATION = "config.json"

DEFAULT_CONFIG = {
    "project": None,
    "fixed_model_name": None,
    "config": DEFAULT_CONFIG_LOCATION,
    "data": None,
    "emulate": None,
    "language": "en",
    "log_file": None
}


def parse_paras(configtree, singleconfig, config_func, config_parse_twe):
    customkey1 = ""
    # 2.1判断方式路由[train predict hard]
    for key in singleconfig:
        customkey1 = key
        break
    if customkey1 not in configtree["process"]:
        raise ConfigException("value: error {0} is not in process choice.".format(customkey1))
    # 2.2 判断模块参数[inform outform pipline]
    keyt = copy.copy(customkey1)
    wayt = "key"
    typet = "neces"
    classt = "class"
    config_parse_twe(singleconfig[customkey1], keyt, wayt, typet, classt)
    # 2.3 判断pipline参数[模型分类 分词 PMI]
    # 2.3.1 必要条件
    keyt = "func_name"
    wayt = "key"
    typet = "lest1"
    classt = "class"
    config_func(singleconfig[customkey1]["pipline"], keyt, wayt, typet, classt)
    ff = "train.csv"
    # 2.4 判断pipline参数[模型分类 分词 PMI]
    # 2.4.1 充分条件
    keyt = "func_name"
    wayt = "key"
    typet = "lest1"
    classt = "class"
    config_func(configtree["process"][customkey1]["pipline"], keyt, wayt, typet, classt)
    config_func(singleconfig[customkey1]["pipline"], keyt, wayt, typet, classt)
    # 2.5 遍历每一个参数
    tmpojson = configtree["process"][customkey1]["pipline"]
    tmpcjson = singleconfig[customkey1]["pipline"]

    # 参数集处理，配置列表必须全部列出，给出默认值，找不到报错。
    # 如果不是list dict,不存在强行赋值；如果是list,不存在忽略该功能；如果是dict,不存在报错。
    def para_check(ojson, cjson):
        for i1 in ojson:
            # 默认处理普通类型
            try:
                if isinstance(ojson[i1], list):
                    if cjson[i1] not in ojson[i1]:
                        print(ojson[i1])
                        print(cjson[i1])
                        raise ConfigException("value: error {0} is not in process choice.".format(i1))
                elif isinstance(ojson[i1], dict):
                    # 参数集处理
                    para_check(ojson[i1], cjson[i1])
                else:
                    # 默认处理普通类型
                    try:
                        cjson[i1]
                    except Exception as e:
                        cjson[i1] = copy.deepcopy(ojson[i1])
            except Exception as e:
                pass

    para_check(tmpojson, tmpcjson)
    return customkey1, tmpcjson


class InvalidConfigError(ValueError):
    """Raised if an invalid configuration is encountered."""

    def __init__(self, message):
        # type: (Text) -> None
        super(InvalidConfigError, self).__init__(message)


class AisetsConfig(object):
    DEFAULT_PROJECT_NAME = "default"

    def __init__(self, env_vars=None, cmdline_args=None):
        fileenv = os.path.join("config", "env.json")
        # if filename is None and os.path.isfile(DEFAULT_CONFIG_LOCATION):
        #     filename = DEFAULT_CONFIG_LOCATION
        #
        # self.override(DEFAULT_CONFIG)
        if fileenv is not None:
            try:
                file_config = utils.read_json_file(fileenv)
            except ValueError as e:
                raise InvalidConfigError("Failed to read configuration file "
                                         "'{}'. Error: {}".format(fileenv, e))
            self.override(file_config)
        # fileprocess = os.path.join("config", "process.json")
        # if fileprocess is not None:
        #     try:
        #         file_config = utils.read_json_file(fileprocess)
        #     except ValueError as e:
        #         raise InvalidConfigError("Failed to read configuration file "
        #                                  "'{}'. Error: {}".format(fileprocess, e))
        #     self.override(file_config)

        if env_vars is not None:
            env_config = self.create_env_config(env_vars)
            self.override(env_config)

        if cmdline_args is not None:
            cmdline_config = self.create_cmdline_config(cmdline_args)
            self.override(cmdline_config)

        # if isinstance(self.__dict__['pipeline'], six.string_types):
        #     from aisetts import registry
        #     if self.__dict__['pipeline'] in registry.registered_pipeline_templates:
        #         self.__dict__['pipeline'] = registry.registered_pipeline_templates[self.__dict__['pipeline']]
        #     else:
        #         raise InvalidConfigError("No pipeline specified and unknown pipeline template " +
        #                                  "'{}' passed. Known pipeline templates: {}".format(
        #                                      self.__dict__['pipeline'],
        #                                      ", ".join(registry.registered_pipeline_templates.keys())))

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
