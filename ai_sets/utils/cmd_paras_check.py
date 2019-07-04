# -*- coding: utf-8 -*-
import argparse
import re


class ConfigException(Exception):
    """Exception wrapping lower level exceptions that may happen while training

      Attributes:
          failed_target_project -- name of the failed project
          message -- explanation of why the request is invalid
      """

    def __init__(self, exception=None):
        if exception:
            self.message = exception

    def __str__(self):
        return self.message


def create_argparser():
    parser = argparse.ArgumentParser(description='parse incoming text')
    parser.add_argument('-c', '--config',
                        help="config file, all the command line options can "
                             "also be passed via a (json-formatted) config "
                             "file. NB command line args take precedence")
    parser.add_argument('-e', '--emulate',
                        choices=['wit', 'luis', 'dialogflow'],
                        help='which service to emulate (default: None i.e. use '
                             'simple built in format)')
    parser.add_argument('-l', '--language',
                        choices=['de', 'en'],
                        help="model and data language")
    parser.add_argument('-m', '--mitie_file',
                        help='file with mitie total_word_feature_extractor')
    parser.add_argument('-p', '--path',
                        help="path where project files will be saved")
    parser.add_argument('--project',
                        type=str,
                        help="project name")
    parser.add_argument('--branch',
                        type=str,
                        help='port on which to run server')
    parser.add_argument('-P', '--port',
                        type=int,
                        help='port on which to run server')
    parser.add_argument('--model',
                        type=str,
                        help="config file")
    parser.add_argument('--jsonfile',
                        type=str,
                        help="config file")
    parser.add_argument('--pipeline',
                        help="The pipeline to use. Either a pipeline template "
                             "name or a list of components separated by comma")
    parser.add_argument('-t', '--token',
                        help="auth token. If set, reject requests which don't "
                             "provide this token as a query parameter")
    parser.add_argument('-w', '--write',
                        help='file where logs will be saved')

    return parser


def json_parser(jsonobj):
    """
    :param jsonobj:输入json
    :return: 清理好的json
    """
    keys = [
        "project",
        "json_file",
        "col_use",
        "col_new",
        "label_map",
        "purpose",
    ]
    # 1. 判断是否包含一级关键词
    for key in keys:
        if key not in jsonobj:
            print("keyword: %s is need in config json file" % key)
            raise "keyword: %s is need in config json file" % key

    # 2. 判断主功能
    purkey = ["name"]  # , "chara", "method"
    purallkey = ["name", "chara", "method"]  #
    for key in purkey:
        if key not in jsonobj["purpose"]:
            print("keyword: %s is need in 'purpose' file" % key)
            raise "keyword: %s is need in 'purpose' file" % key

    # 2.2 内容判断
    for key in purallkey:
        if key in jsonobj["purpose"]:
            if key == "chara":
                # 格式转化
                jsonobj["purpose"]["chara"] = int(jsonobj["purpose"]["chara"])
            if key == "method":
                # 后续处理
                pass
            if key == "dict":
                # 格式转化
                if key not in ["self", "default"]:
                    print("keyword: %s is not in 'purpose' file" % key, ". should be self or default!")
                    raise "keyword: %s is need in 'purpose' file" % key

    for i in jsonobj["col_new"]:
        i = str(i)

    # 3. 判断col_use的关键词
    if not isinstance(jsonobj["col_use"], dict):
        raise "keyword col_use's obj show be type of dict."
    for i in jsonobj["col_use"]:
        pass
        # i = str(i)

    # 4. 判断col_new的关键词
    if not isinstance(jsonobj["col_new"], dict):
        raise "keyword col_new's obj show be type of dict."
    for i in jsonobj["col_new"]:
        pass

    # 5. 判断label_map的格式
    if not isinstance(jsonobj["label_map"], dict):
        raise "keyword label_map's obj show be type of dict."
    allusecol = []
    for i in jsonobj["label_map"]:
        if not isinstance(jsonobj["label_map"][i], list):
            raise "keyword col_use's obj show be type of list."
        allusecol.append(i)
        allusecol.extend(jsonobj["label_map"][i])
    red = re.compile(r'^_.*|.*_$|.*__.*')
    findres = map(red.search, allusecol)
    for i in findres:
        if i is not None:
            raise "column key not in proper format."
    return jsonobj


def config_key_parser(jsonobj, neces=[], lest1=[]):
    """
    :param jsonobj:输入json
    :param neces:必要键
    :param lest1:至少存在键
    :return: 清理好的json
    """
    # 1. 判断是否包含一级关键词
    if len(neces) != 0:
        for key in neces:
            if key not in jsonobj and key is not None:
                print(neces)
                print(jsonobj.keys())
                print(key)
                raise ConfigException("keyword: error {0} is necessary.".format(key))
        return jsonobj

    # 1. 判断是否包含一级关键词
    if len(lest1) != 0:
        for key in jsonobj:
            if key not in lest1 and key is not None:
                print(lest1)
                print(jsonobj.keys())
                print(key)
                raise ConfigException("keyword: error {0} is at least 1.".format(key))
        return jsonobj


def config_val_parser(jsonobj, inkey, neces=[], lest1=[]):
    """
    :param jsonobj:输入jsonobj
    :param inkey:inkey
    :param neces:必要值
    :param lest1:至少存在值
    :return: 清理好的json
    """
    # 1. 判断是否包含一级关键词
    if len(neces) != 0:
        for val in neces:
            if val not in jsonobj[inkey] and val is not None:
                print(neces)
                print(val)
                raise ConfigException("value: error {0} is necessary".format(val))
        return jsonobj

    # 1. 判断是否包含一级关键词
    if len(lest1) != 0:
        if jsonobj[inkey] in lest1:
            return jsonobj
        print(jsonobj)
        print(inkey)
        print(lest1)
        raise ConfigException("value: error {0} is at least 1".format(jsonobj[inkey]))
