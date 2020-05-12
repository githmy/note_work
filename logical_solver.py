"""
逻辑内核:
感知 行动 认知 语言
领域 场景 条件 步骤

场景: 某条件环境
领域: 方法流程

存储空间: 都要带 场景 领域
  实体关系:
  实体属性:
解析空间--实例空间
步骤堆栈:
  堆栈操作:
步骤时间: 都要带 场景 领域
  语言解析:
    根据场景 分解为三元组
  类比: 取出属性
  归纳: 取出共性
  抽象: 步骤概念集合
    联想:
  想象: 可以无逻辑的假设
  演绎: 逻辑推理 稳定规则
    对象加载:
    对象映射:
    数学计算:
"""
import re
import os
# !pip install jsonpatch
import jieba
import jieba.posseg as pseg
import json
import logging
import logging.handlers
from utils.path_tool import makesurepath
import jsonpatch
import itertools
from pyhanlp import *
from latex_solver import latex2list_P, postfix_convert_P, latex2space, latex2unit
from latex_solver import latex_json, baspath, step_alist, step_blist

# pd.set_option('display.max_columns', None)
cmd_path = os.getcwd()
datalogfile = os.path.join(cmd_path, '..', 'data', 'log')
makesurepath(datalogfile)
datalogfile = os.path.join(datalogfile, 'anal.log')

logger1 = logging.getLogger('log')
logger1.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler(datalogfile, maxBytes=104857600, backupCount=10)
# fh = logging.FileHandler(datalogfile)
ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger1.addHandler(fh)
logger1.addHandler(ch)


def latex2split(instr):
    # 字符串 以 {} 分成单元。只分一层，不递归
    strlist = instr.split()
    slenth = len(strlist)
    keyindexlist = []
    ein = -1
    for i1 in range(slenth):
        singl_cout = 0
        numcout = 0
        if strlist[i1] == "{" and i1 >= ein:
            singl_cout += 1
            for i2 in range(i1 + 1, slenth):
                if strlist[i2] == "{":
                    singl_cout += 1
                elif strlist[i2] == "}":
                    singl_cout -= 1
                if singl_cout == 0:
                    numcout += 1
                    if numcout == 1:
                        ein = i2 + 1
                        keyindexlist.append([i1, ein])
                        break
            if numcout != 1:
                raise Exception("match num error.")
    sin = 0
    stroutlist = []
    for i1 in keyindexlist:
        stroutlist += strlist[sin:i1[0]]
        sin = i1[1]
        stroutlist.append(" ".join(strlist[i1[0]:i1[1]]))
    stroutlist += strlist[keyindexlist[len(keyindexlist) - 1][1]:-1]
    return stroutlist


class Field(object):
    def __init__(self, name):
        # 0. 加载原始obj
        self.field_name = name


class Scene(object):
    def __init__(self, name):
        # 0. 加载原始obj
        self.scene_name = name


class GStack(object):
    """一组语言一个Gstack"""

    def __init__(self):
        # 步骤列表
        self.step_list = []
        # 相关的空间列表
        self.space_list = []
        # 语言记录
        self.lang_list = []

    def is_inspace_list(self, space_name, scene_name, field_name):
        for space in self.space_list:
            if space["space_name"] == space_name and space["field_name"] == field_name and space[
                "scene_name"] == scene_name:
                return True

    def loadspace(self, space_name, scene_name, field_name, space_ins):
        self.space_list.append(
            {"space_name": space_name, "field_name": field_name, "scene_name": scene_name, "space_ins": space_ins})

    def readspace(self, space_name, scene_name, field_name):
        for space in self.space_list:
            if space["space_name"] == space_name and space["field_name"] == field_name and space[
                "scene_name"] == scene_name:
                return space["space_ins"]


class BasicalSpace(object):
    """存储内核: 固定存储形式和操作形式 """

    def __init__(self, space_name="basic", field_name=None, scene_name=None):
        """按需加载"""
        self.space_name = space_name
        self.field_name = field_name
        self.scene_name = scene_name
        # 1. 加载属性, 加载关系 只有作为逻辑的基类才会加载。
        if space_name == "basic":
            self._proper_keys, self._proper_trip, self._relation_trip = self.storage_oper("r")
        else:
            self._proper_keys, self._proper_trip = {}, {}
            self._relation_trip = {}
            self._questproperbj = {}
            self._questtriobj = {}

    def storage_oper(self, operstr):
        """硬件：存储交互操作"""
        if operstr == "r":
            # 属性 是 xx 特制 实例类 不代表 等价
            properobj = {
                "角": {
                },
                "n边形": {
                    "边数": "x",
                    "点数": "x",
                    "内角和": "x * 1 8 0 - 3 6 0",
                    "面积": None,
                    "周长": None,
                },
                "三角形": {
                    "底": "x",
                    "高": "y",
                    "面积": "x * y / 2",
                },
                "圆": {
                    "半径": "x",
                    "直径": "2 x",
                    "面积": "\\pi * r * r",
                    "周长": "\\pi * r * 2",
                },
                "矩形": {
                    "长": "x",
                    "宽": "y",
                    "面积": "x * y",
                    "周长": "2 x + 2 y",
                    "邻边": "_垂直",
                    "对角线": "_相等",
                },
                "正方形": {
                    "长": "x",
                    "宽": "x",
                    "面积": "x * x",
                    "周长": "4 x",
                    "对角线": "_垂直",
                }
            }
            triobj = [
                ["三角形", "属于", "n边形"],
                ["四边形", "属于", "n边形"],
                ["平行四边形", "属于", "四边形"],
                ["等腰三角形", "属于", "三角形"],
                ["等边三角形", "属于", "等腰三角形"],
                ["矩形", "属于", "平行四边形"],
                ["正方形", "属于", "矩形"],
                ["等角三角形", "等价", "等边三角形"],
                ["直径", "属于", "弦"],
                ["弦", "附属", "圆"],
                ["切线", "附属", "圆"],
                ["半径", "附属", "圆"],
                ["半径", "附属", "圆"],
                ["圆心", "附属", "半径"],
            ]
            proper_trip = {}
            couindex = 0
            for i1 in properobj:
                for i2 in properobj[i1]:
                    couindex += 1
                    proper_trip[str(couindex)] = [i1, "有属性", i2, properobj[i1][i2]]
            # print(proper_trip)
            # print(len(proper_trip))
            proper_keys = list(properobj.keys())
            relation_trip = {id1: i1 for id1, i1 in enumerate(triobj)}
            return proper_keys, proper_trip, relation_trip
        elif operstr == "w":
            return True

    def triple_oper_bak(self, add={"properobj": {}, "triobj": []}, dele={"properobj": [], "triobj": []}):
        """内存：triple交互操作"""
        for oneproper in add["properobj"]:
            self._proper_trip[oneproper] = add["properobj"][oneproper]
        for onetri in add["triobj"]:
            havesig = 0
            for oritri in self._relation_trip:
                patch = jsonpatch.JsonPatch.from_diff(onetri, oritri)
                if list(patch) == []:
                    havesig = 1
                    break
            if havesig == 0:
                self._relation_trip.append(onetri)
        for oneproper in dele["properobj"]:
            try:
                del self._proper_trip[oneproper]
            except Exception as e:
                logger1.info("delete %s error %s" % (oneproper, e))
        for onetri in dele["triobj"]:
            lenth = len(self._relation_trip)
            for id1 in range(lenth - 1, -1, -1):
                patch = jsonpatch.JsonPatch.from_diff(onetri, self._relation_trip[id1])
                if list(patch) == []:
                    del self._relation_trip[id1]
                    break

    def property_oper(self, properobj, addc={}, delec=[]):
        """内存：triple交互操作"""
        for oneproper in addc:
            # print(oneproper)
            properobj[oneproper] = addc[oneproper]
        for oneproper in delec:
            try:
                del properobj[oneproper]
            except Exception as e:
                logger1.info("delete %s error %s" % (oneproper, e))
        return properobj

    def triple_oper(self, oritriple, addc=[], delec=[]):
        """内存：triple交互操作"""
        for onetri in addc:
            havesig = 0
            for orikey in oritriple:
                patch = jsonpatch.JsonPatch.from_diff(onetri, oritriple[orikey])
                if list(patch) == []:
                    havesig = 1
                    break
            if havesig == 0:
                oritriple[str(len(oritriple) + 1)] = onetri
        for onetri in delec:
            for orikey in oritriple:
                patch = jsonpatch.JsonPatch.from_diff(onetri, oritriple[orikey])
                if list(patch) == []:
                    del oritriple[orikey]
                    break
        return oritriple

    def get_child(self, obj):
        waitelist = [obj]
        outlist = []
        # 1. 找到所有的 子级
        lastlenth = -1
        nowlenth = 0
        while lastlenth != nowlenth:
            lastlenth = nowlenth
            for id in range(len(waitelist)):
                tmpobj = waitelist[id]
                for trip in self._relation_trip:
                    if tmpobj == trip[2] and "属于" == trip[1] and trip[0] not in outlist + waitelist:
                        waitelist.append(trip[0])
                outlist.append(tmpobj)
                waitelist.remove(tmpobj)
            nowlenth = len(outlist)
        return outlist

    def get_father(self, obj):
        waitelist = [obj]
        outlist = []
        # 1. 找到所有的 父级
        lastlenth = -1
        nowlenth = 0
        while lastlenth != nowlenth:
            lastlenth = nowlenth
            for id in range(len(waitelist)):
                tmpobj = waitelist[id]
                for trip in self._relation_trip:
                    if tmpobj == trip[0] and "属于" == trip[1] and trip[2] not in outlist + waitelist:
                        waitelist.append(trip[2])
                outlist.append(tmpobj)
                waitelist.remove(tmpobj)
            nowlenth = len(outlist)
        return outlist

    def find_obj_property_value(self, obj="矩形", property="面积"):
        """内存：给定 主体 和 属性，查找 值list"""
        # 1. 找到所有的父级
        fatherlist = self.get_father(obj)
        # 2. 查找继承级的 属性值
        properlist = []
        for item in fatherlist:
            if item in self._proper_keys and property in self._proper_trip[item] and self._proper_trip[item][
                property] is not None:
                properlist.append(self._proper_trip[item][property])
        return properlist

    def find_property_value_child(self, obj="四边形", property="对角线", value="相等"):
        """内存：给定子级属性值 返回 子级list"""
        # 1. 找到所有的 子级
        # print("get_child")
        childlist = self.get_child(obj)
        # 2. 查找 符合 属性值的主体
        objlist = []
        for item in childlist:
            if item in self._proper_keys and property in self._proper_trip[item] and value == self._proper_trip[item][
                property]:
                objlist.append(item)
        return objlist


class VirtualSpace(BasicalSpace):
    def __init__(self, space_name="basic", field_name=None, scene_name=None):
        self._proper_trip = super(VirtualSpace, self)._proper_trip
        self._relation_trip = super(VirtualSpace, self)._relation_trip
        self._proper_keys = super(VirtualSpace, self)._proper_keys


class NLPtool(object):
    def __init__(self):
        # token list
        self.latex_token = list(latex_json.keys())
        # token 中文名
        self.latex_map = {}
        for item in latex_json:
            if latex_json[item] is not None:
                self.latex_map[item] = latex_json[item]
        # 同义词典
        with open(os.path.join(baspath, 'synonymous.txt'), 'rt', encoding="utf-8") as f:
            result = f.readlines()
            self.synonymous = [i1.strip("\n").split() for i1 in result]
        # 词性
        with open(os.path.join(baspath, 'nominal.txt'), 'rt', encoding="utf-8") as f:
            result = f.readlines()
            self.nominal = {i1.strip("\n").split()[0]: i1.strip("\n").split()[1] for i1 in result}
        self.latex_punc = [":", ",", ".", ";"]
        # print(self.latex_token)
        # print(self.latex_map)
        # print(self.synonymous)
        # print(self.nominal)
        # print(self.latex_punc)

    def replace_synonymous(self, word):
        for i1 in self.synonymous:
            if word in i1:
                return i1[0]
        else:
            return word

    def get_extract(self, strlist, key):
        """获取 抽象概念属性 词"""
        slenth = len(strlist)
        matchlist = []
        keyindexlist = []
        if slenth < 2:
            return " ".join(strlist), matchlist
        for i1 in range(slenth - 2, -1, -1):
            if strlist[i1] == key:
                if not strlist[i1 + 1].startswith("{"):
                    matchlist.append(["{ " + strlist[i1 + 1] + " }", "是", self.latex_map[key]])
                else:
                    matchlist.append([strlist[i1 + 1], "是", self.latex_map[key]])
                keyindexlist.append(i1)
        strlist = [strlist[i1] for i1 in range(slenth) if i1 not in keyindexlist]
        return " ".join(strlist), matchlist

    def get_step(self, strlist):
        """获取 步骤 词 三类标签"""
        questiontype = ["求", "什么", "多少"]
        quest_notype = ["要求"]
        slenth = len(strlist)
        matchlist = {}
        key_type = []
        key_index = []
        if strlist[0] in step_blist:
            key_type.append("导出")
            key_index.append(0)
        elif strlist[0] in questiontype and strlist[0] not in quest_notype:
            key_type.append("求")
            key_index.append(0)
        else:
            key_type.append("已知")
            key_index.append(0)
        for i1 in range(1, slenth):
            if strlist[i1] in step_alist:
                key_type.append("已知")
                key_index.append(i1)
            elif strlist[i1] in step_blist:
                key_type.append("导出")
                key_index.append(i1)
            elif strlist[i1] in questiontype and strlist[i1] not in quest_notype:
                key_type.append("求")
                key_index.append(i1)
        lenth = len(key_index)
        for i1 in range(lenth - 1, 0, -1):
            if key_type[i1] == key_type[i1 - 1]:
                del key_type[i1]
                del key_index[i1]

        # index_question = 9999
        # for idwt, word_tri in enumerate(fenci_list):
        #     if not isinstance(word_tri[0], list):
        #         for key in questiontype:
        #             if key in word_tri[0]:
        #                 for nkey in quest_notype:
        #                     if nkey not in word_tri[0]:
        #                         index_question = idwt
        #                         break
        #             if index_question != 9999:
        #                 break
        #         if index_question != 9999:
        #             break
        # if index_question != 9999:
        #     for indwt in range(index_question, -1, -1):
        #         if not isinstance(fenci_list[indwt][0], list) and "标点符号" == fenci_list[indwt][1]:
        #             index_question = indwt
        #             break

        return " ".join(strlist), matchlist

    def latex_extract_property(self, instr):
        """单句： 分组后 返还 去掉抽象概念的实体latex"""
        # 1. token 预处理
        listtk = {}
        for i1 in self.latex_token:
            tt = i1.replace(" ", "")
            tlsit = []
            for i2 in tt:
                tlsit.append(i2)
            listtk[" ".join(tlsit)] = len(" ".join(tlsit))
        contenta = sorted(listtk.items(), key=lambda x: -x[1])
        contenta = [i1[0] for i1 in contenta]
        contento = [i1.replace(" ", "") for i1 in contenta]
        # 2. 原始字符处理
        tt = instr.replace(" ", "")
        tinlist = [i1 for i1 in tt]
        tinstr = " ".join(tinlist)
        for s, n in zip(contenta, contento):
            tinstr = tinstr.replace(s, n)
        tinstr = tinstr.replace("  ", " ").replace("  ", " ")
        # 3. latex 分词。先按标点符号分，再按关键算符分
        tinlist = re.split(',|，|\n,\t', tinstr)

        # 每个断句
        latexlist = []
        outjson = []
        for i1 in tinlist:
            # 3.1 分割 关键词 索引。句意级。
            tstr = i1
            for word in self.latex_map:
                # 每个 属性词
                if "n" == self.nominal[self.latex_map[word]]:
                    tstrli = latex2split(tstr)
                    tstr, tjson = self.get_extract(tstrli, word)
                    outjson += tjson
            latexlist.append(tstr)
        return latexlist, outjson

    def latex_split(self, instr):
        """
        如果识别的好，这部没必要，如果识别的不好这步可以作为修正，或 对非标准格式的字符串标准化
        
        """
        # 1. token 预处理
        listtk = {}
        for i1 in self.latex_token:
            tt = i1.replace(" ", "")
            tlsit = []
            for i2 in tt:
                tlsit.append(i2)
            listtk[" ".join(tlsit)] = len(" ".join(tlsit))
        contenta = sorted(listtk.items(), key=lambda x: -x[1])
        contenta = [i1[0] for i1 in contenta]
        contento = [i1.replace(" ", "") for i1 in contenta]
        # 2. 原始字符处理
        tt = instr.replace(" ", "")
        tinlist = [i1 for i1 in tt]
        tinstr = " ".join(tinlist)
        for s, n in zip(contenta, contento):
            tinstr = tinstr.replace(s, n)
        tinstr = tinstr.replace("  ", " ").replace("  ", " ")
        # 3. latex 分词。先按标点符号分，再按关键算符分
        tinlist = re.split(',|，|\n,\t', tinstr)
        outlist = []
        # 每个断句
        for i1 in tinlist:
            # 3.1 分割 关键词 索引。句意级，
            tindexlist = []
            for word in self.latex_map:
                # 每个词
                rword = word.replace("\\", "\\\\")
                for tm in re.finditer(rword, i1):
                    tindexlist.append([tm.start(), tm.end()])
            tindexlist = sorted(tindexlist, key=lambda x: x[0])
            # 3.2 内容组合
            toutlist = latex2split(instr)
            # toutlist = []
            # tstart = 0
            # for ind in tindexlist:
            #     tstris = i1[tstart:ind[0]].strip()
            #     if tstris.startswith("{") and tstris.endswith("}"):
            #         tstris = tstris[1:-1].strip()
            #     toutlist.append([tstris, "n"])
            #     # toutlist.append([i1[tstart:ind[0]].strip(), "n"])
            #     latexstr = self.latex_map[i1[ind[0]:ind[1]]]
            #     toutlist.append([latexstr, self.nominal[latexstr]])
            #     tstart = ind[1]
            # tstris = i1[tstart:].strip()
            # if tstris.startswith("{") and tstris.endswith("}"):
            #     tstris = tstris[1:-1].strip()
            # # toutlist.append([i1[tstart:].strip(), "n"])
            # toutlist.append([tstris, "n"])
            # print(toutlist)
            # exit()
            # 去空
            outlist.append([onetup for onetup in toutlist if onetup[0] != ""])
        return outlist

    def latex_split_bak(self, instr):
        """如果识别的好，这部没必要，如果识别的不好这步可以作为修正，或 对非标准格式的字符串标准化"""
        # 1. token 预处理
        listtk = {}
        for i1 in self.latex_token:
            tt = i1.replace(" ", "")
            tlsit = []
            for i2 in tt:
                tlsit.append(i2)
            listtk[" ".join(tlsit)] = len(" ".join(tlsit))
        contenta = sorted(listtk.items(), key=lambda x: -x[1])
        contenta = [i1[0] for i1 in contenta]
        contento = [i1.replace(" ", "") for i1 in contenta]
        # 2. 原始字符处理
        tt = instr.replace(" ", "")
        tinlist = [i1 for i1 in tt]
        tinstr = " ".join(tinlist)
        for s, n in zip(contenta, contento):
            tinstr = tinstr.replace(s, n)
        tinstr = tinstr.replace("  ", " ").replace("  ", " ")
        # 3. latex 分词。先按标点符号分，再按关键算符分
        tinlist = re.split(',|，|\n,\t', tinstr)
        outlist = []
        # 每个断句
        for i1 in tinlist:
            # 3.1 分割 关键词 索引。句意级，
            tindexlist = []
            for word in self.latex_map:
                # 每个词
                rword = word.replace("\\", "\\\\")
                for tm in re.finditer(rword, i1):
                    tindexlist.append([tm.start(), tm.end()])
            tindexlist = sorted(tindexlist, key=lambda x: x[0])
            # 3.2 内容组合
            toutlist = []
            tstart = 0
            for ind in tindexlist:
                tstris = i1[tstart:ind[0]].strip()
                if tstris.startswith("{") and tstris.endswith("}"):
                    tstris = tstris[1:-1].strip()
                toutlist.append([tstris, "n"])
                # toutlist.append([i1[tstart:ind[0]].strip(), "n"])
                latexstr = self.latex_map[i1[ind[0]:ind[1]]]
                toutlist.append([latexstr, self.nominal[latexstr]])
                tstart = ind[1]
            tstris = i1[tstart:].strip()
            if tstris.startswith("{") and tstris.endswith("}"):
                tstris = tstris[1:-1].strip()
            # toutlist.append([i1[tstart:].strip(), "n"])
            toutlist.append([tstris, "n"])
            # 去空
            outlist.append([onetup for onetup in toutlist if onetup[0] != ""])
        return outlist

    def stand_trans(self, fenci_list):
        """text latex 统一，然后分因果"""

        def fixword(word):
            if "text" in word:
                return [word["text"]]
            else:
                return self.latex_split(word["latex"])

        stand_fenci_list = [fixword(word) for word in fenci_list]
        stand_fenci_list = list(itertools.chain(*stand_fenci_list))
        stand_fenci_list = [word if len(word) > 1 else word[0] for word in stand_fenci_list]
        return stand_fenci_list

    def json2space(self, write_json, space_ins):
        # 1. 先写属性
        space_ins.property_oper(space_ins._proper_trip, addc=write_json["add"]["properobj"],
                                delec=write_json["dele"]["properobj"])
        logger1.info("write property: %s" % write_json["add"]["properobj"])
        # 2. 修正元组实体
        varlist = space_ins._proper_keys
        write_json["add"]["triobj"] = [
            [latex2unit(online[0], varlist=varlist), online[1], latex2unit(online[2], varlist=varlist)] for online in
            write_json["add"]["triobj"]]
        # 3. 再写元组
        space_ins.triple_oper(space_ins._proper_trip, addc=write_json["add"]["triobj"],
                              delec=write_json["dele"]["triobj"])
        logger1.info("write triple: %s" % write_json["add"]["triobj"])
        # space_ins.property_oper(space_ins._properobj, addc=write_json["add"]["properobj"], delec=write_json["dele"]["properobj"])
        # space_ins.triple_oper(space_ins._triobj, addc=write_json["add"]["triobj"], delec=write_json["dele"]["triobj"])

    def fenci2triple(self, fenci_list, space_ins):
        """ 解析：词性逻辑 到 内存格式"""
        print(88)
        print(fenci_list)
        # print(fenci_list)
        # 1. 问句分割索引
        newfenci = []
        for i1 in fenci_list:
            newfenci.append(self.get_step(i1))
        # 2. 三元组提取
        triobj = []
        quest_triobj = []
        properobj = {}
        quest_properobj = {}
        for idwt, word_tri in enumerate(newfenci):
            lenthwt = len(word_tri)
            if isinstance(word_tri[0], list):
                # 处理latex元组 名词 实例类
                print(23211)
                print(word_tri)
                for idw, wordtup in enumerate(word_tri):
                    print(wordtup)
                    if idw < lenthwt - 1 and "n" == word_tri[idw + 1][1] and "n" == word_tri[idw][1] and wordtup[
                        0] in space_ins._properobj:
                        newins = word_tri[idw + 1][0].replace(" ", "")
                        if newins not in properobj:
                            properobj[newins] = {}
                        properobj[newins]["是"] = wordtup[0]
                # triobj += [[word_tri[idw + 1][0], "是", wordtup[0]] for idw, wordtup in enumerate(word_tri) if
                #            idw < lenthwt - 1 and "n" == word_tri[idw + 1][1] and "n" == word_tri[idw][1] and wordtup[
                #                0] in space_ins._properobj]
                # print(triobj)
                # 处理latex元组 动词 前后必须删除 抽象类，保留实体
                tmp_word_tri = [newci for newci in word_tri if newci[0] not in space_ins._proper_trip]
                # print(tmp_word_tri)
                triobj += [[tmp_word_tri[idw - 1][0], wordtup[0], tmp_word_tri[idw + 1][0]] for idw, wordtup in
                           enumerate(tmp_word_tri) if idw > 0 and idw < lenthwt - 1 and "v" == wordtup[1]]
                # print(triobj)
            else:
                # 普通 字符 暂时略过
                if index_question != -1:
                    # 问句类型处理
                    pass
                else:
                    # 陈述句处理
                    pass
        # 暂时没想到
        # print(properobj)
        # write_json = {"add": {"properobj": {"test": {"red": 1, "blue": 2, "green": 3}},
        #                       "triobj": [['三角形n', '属于', 'n边形'], ['三角形', '属于', 'n边形']]},
        #               "dele": {"properobj": ["n边形"], "triobj": [['三角形', '属于', 'n边形']]}}
        write_json = {
            "add": {
                "properobj": properobj, "triobj": triobj,
                "quest_properobj": quest_properobj, "quest_triobj": quest_triobj,
            },
            "dele": {"properobj": {}, "triobj": [], "quest_properobj": {}, "quest_triobj": []},
        }
        # print(565656)
        # print(write_json)
        return write_json

    def nature2space(self, instr_list, gstack):
        """ 解析：自然语言 到 空间三元组。按因果 分步骤"""
        # 1. 语言录入初始化, 添加语句
        fenci_list = []
        for strs in instr_list:
            # 2. 分词实体
            if "text" in strs:
                word_array = HanLP.parseDependency(strs["text"]).getWordArray()
                for word in word_array:
                    fenci_list.append({"text": [word.LEMMA, word.DEPREL, word.HEAD.LEMMA]})
            else:
                # latex 分句 空格标准化
                latex_splist = re.split(',|，|;|\n|\\\qquad|\\\quad|\t', strs["latex"])
                fenci_list += [{"latex": latex2space(latstr)} for latstr in latex_splist]
        # 3. text latex 分词标记 统一化
        stand_fenci_list = self.stand_trans(fenci_list)
        gstack.lang_list.append(stand_fenci_list)
        logger1.info("language clean: %s" % stand_fenci_list)
        # 4. 分词实体 2 3元组空间
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = gstack.readspace(space_name, scene_name, field_name)
        # 按空间规则写入3元组，各种实体属性去空格 尚未展开
        write_json = self.fenci2triple(stand_fenci_list, basic_space_ins)
        logger1.info("json write: %s" % write_json)
        # 5. 写入空间, 先写 属性再根据属性 合并 三元组
        self.json2space(write_json, space_ins)
        print(basic_space_ins._proper_trip)
        print(basic_space_ins._relation_trip)
        print(space_ins._proper_trip)
        print(space_ins._relation_trip)
        print("check ok")


class Steps(object):
    """ 步骤: """

    def __init__(self):
        self.step_name = None
        # 临时分两种 easy detail
        self.inference_type = "easy"
        self.out_type = "easy"

    def loadspace(self, bs_ins):
        pass

    def inference(self, operstr):
        pass


class LogicalInference(object):
    """ 推理内核: """

    def __init__(self):
        self.language = NLPtool()
        self.gstack = GStack()
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        if not self.gstack.is_inspace_list(space_name, scene_name, field_name):
            space_ins = BasicalSpace(space_name=space_name, field_name=field_name, scene_name=scene_name)
            self.gstack.loadspace(space_name, scene_name, field_name, space_ins)
        space_name = "customer"
        if not self.gstack.is_inspace_list(space_name, scene_name, field_name):
            space_ins = BasicalSpace(space_name=space_name, field_name=field_name, scene_name=scene_name)
            self.gstack.loadspace(space_name, scene_name, field_name, space_ins)

    def __call__(self, *args, **kwargs):
        """输入为语言的 list dic 数组: text latex"""
        analist = args[0]
        logger1.info("initial analyzing: %s" % analist)
        # 0. 处理句间 关系，写入部分实体。 基于符号类型的区分标签
        analist = self.sentence2normal(analist)
        # 1. 循环处理每句话 生成空间解析 和 步骤list
        for sentence in analist:
            logger1.info("sentence: %s" % sentence)
            # 2. 基于因果的符号标签
            self.analyize_strs(sentence)
        # 2. 内存推理，基于之前的步骤条件
        print(self.gstack.space_list)
        self.inference(triplelist)
        print("end")
        exit()

    def get_allkeyproperty(self, analist):
        """text latex 句子间合并, 写入概念属性json，返回取出主题概念的列表"""
        # 1. 展成 同级 list
        analist = list(itertools.chain(*analist))
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [sentence[list(sentence.keys())[0]].strip() for sentence in analist]
        olenth = len(contlist)
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        outlatex = []
        for idn in range(olenth):
            if keylist[idn] == "latex":
                # print("latex")
                # print(contlist[idn])
                latexlist, propertyjson = self.language.latex_extract_property(contlist[idn])
                # print(latexlist)
                # print(propertyjson)
                outlatex.append([{"latex": " ; ".join(latexlist)}])
                write_json = {
                    "add": {
                        "properobj": {}, "triobj": propertyjson,
                        "quest_properobj": {}, "quest_triobj": {},
                    },
                    "dele": {"properobj": {}, "triobj": [], "quest_properobj": {}, "quest_triobj": []},
                }
                self.language.json2space(write_json, space_ins)
            else:
                outlatex.append([{keylist[idn]: contlist[idn]}])
        # print(outlatex)
        # exit()
        return outlatex

    def sentence2normal(self, analist):
        """text latex 句子间合并 按句意合并"""
        # 1. 展成 同级 list
        analist = list(itertools.chain(*analist))
        # 2. 去掉空的
        analist = [{list(sentence.keys())[0]: sentence[list(sentence.keys())[0]].strip()} for sentence in analist if
                   sentence[list(sentence.keys())[0]].strip() != ""]
        # 3. 合并 临近相同的
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [sentence[list(sentence.keys())[0]].strip() for sentence in analist]
        olenth = len(contlist)
        if olenth < 2:
            analist = [[{keylist[i1]: contlist[i1]} for i1 in range(olenth)]]
            analist = self.get_allkeyproperty(analist)
            return analist
        for i1 in range(olenth - 1, 0, -1):
            contlist[i1] = contlist[i1].strip(",，。 \t")
            if contlist[i1] == "":
                del analist[i1]
                continue
            if keylist[i1] == keylist[i1 - 1]:
                analist[i1 - 1] = {keylist[i1 - 1]: contlist[i1 - 1] + " ; " + contlist[i1]}
                del analist[i1]
        # 4. latex text 转化
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        sortkey = list(basic_space_ins._properobj.keys()) + [trio[0] for trio in basic_space_ins._triobj] \
                  + [trio[2] for trio in basic_space_ins._triobj]
        sortkey = set(sortkey)
        sortkey = [[onk, len(onk)] for onk in sortkey]
        sortkey = [onk[0] for onk in sorted(sortkey, key=lambda x: -x[1])]
        # 前一个为 text, 已关键字结尾，且后一个为latex, 以字母开始。则拆分合并。
        ins_json = {}
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [sentence[list(sentence.keys())[0]].strip() for sentence in analist]
        olenth = len(analist)
        if olenth < 2:
            # print(92)
            analist = [[{keylist[i1]: contlist[i1]} for i1 in range(olenth)]]
            analist = self.get_allkeyproperty(analist)
            return analist
        for i1 in range(olenth - 1, 0, -1):
            if keylist[i1] == "latex" and keylist[i1 - 1] == "text":
                for jsonkey in sortkey:
                    mt = re.sub(u"{}$".format(jsonkey), "", contlist[i1 - 1])
                    if mt != contlist[i1 - 1]:
                        # 前一个 以属性名结尾
                        se = re.match(r"^(\w|\s)+", contlist[i1])
                        if se.group() is not None:
                            # 后一个 以字母空格开头的 单元字符串 去空
                            contlist[i1 - 1] = mt.strip()
                            ins_json[contlist[i1]] = {}
                            ins_json[contlist[i1]]["是"] = jsonkey
                            break
        analist = [[{keylist[i1]: contlist[i1]}] for i1 in range(olenth)]
        # 5. 提取所有 抽象类。对应实例，改变字符。属性
        analist = self.get_allkeyproperty(analist)
        # analist = [[i1] for i1 in analist]
        logger1.info("initial clean sentence: %s" % analist)
        # 6. 写入句间的实例
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        write_json = {
            "add": {
                "properobj": ins_json, "triobj": [],
                "quest_properobj": {}, "quest_triobj": {},
            },
            "dele": {"properobj": {}, "triobj": [], "quest_properobj": {}, "quest_triobj": []},
        }
        self.language.json2space(write_json, space_ins)
        return analist

    def analyize_strs(self, instr_list):
        """解析字符串到空间: 考虑之前的话语"""
        self.language.nature2space(instr_list, self.gstack)

    def loadspace(self, bs_ins):
        """加载实体空间: """
        pass

    def inference(self, triplelist):
        """推理流程: 三元组 到 三元组"""
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # 查找 具体 属性值
        res = space_ins.find_obj_property_value(obj="矩形", property="面积")
        print(res)
        # 猜谜查找具体 实体
        res = space_ins.find_property_value_child(obj="四边形", property="对角线", value="相等")
        print(res)
        newtrilist = triplelist
        return newtrilist

    def outputinfo(self, operstr):
        """结果输出: """
        pass

    def compare(self, operstr):
        """类比: """
        pass

    def induce(self, operstr):
        """归纳: """
        pass

    def imagine(self, operstr):
        """想象: """
        pass


# 3. latex 证明
def solve_latex_prove(printstr3, handestr3):
    # 输入为 text 、latex
    # 1. 输入字符标准化
    if isinstance(printstr3, str):
        # 1.1 行级 处理单元
        printstr3 = printstr3.replace("\\\n", "\n")
        # print(printstr3)
        # 1.2 句组级 处理单元
        sentenc_list = re.split('。|\?|？|！|；|。|;|\n', printstr3)
        dic_inlist = []
        for sentence in sentenc_list:
            sp_list = sentence.strip(" ,，.。\t").split("$")
            dic_inlist.append([{"text": sp} if id % 2 == 0 else {"latex": sp} for id, sp in enumerate(sp_list)])
    else:
        dic_inlist = printstr3
    # 2. 分析问句
    li_ins = LogicalInference()
    li_ins(dic_inlist)
    # # 2. 分析解答
    # dicinlist = [handestr3]
    # li_ins = LogicalInference(language=NLPtool)
    # li_ins(dicinlist)
    # # # 2. 分析输入到 虚拟空间 三元组
    # # bs_ins = BasicalSpace()
    # # bs_ins.inference(printstr3)
    # # bs_ins = BasicalSpace()
    # # bs_ins.inference(handestr3)


if __name__ == '__main__':
    """
    印刷体的提问规范：
      1. 写出四则运算表达式（公式题目类型指定, 指定精度） 
      2. 列出方程 并求解 x 代表小明 y 代表时间（方程题 指明变量, 指定精度）
      3. pi 取 3.1415 （常数需赋值）
      4. varlist 至简从单字符变量（a） 到多字符变量 （a_{2}, a_{2}b_{3}），不包含前后缀如 角度
    手写输入为: 印刷体的输出
    [
        { "题号":"3","类型":"公式","已知": ["\\sin { 4 5 ^ { \\circ } } * 2"],"varlist":["ab","ABCD"],"求解":[], "参考步骤":[{"表达式":"\\sqrt 2","分值":"0.5"}]},
        { "题号":"4","类型":"方程","已知": ["a \\sin { 4 5 ^ { \\circ } } * 2 = 6"],"varlist":["ab","ABCD"],"求解":["a"], "参考步骤":[{},{}]},
        { "题号":"5",
          "类型":"证明",
          "已知": [
            {"type":"text","txt":"求证"},{"type":"latex","txt":"\\sin { 4 }"},
          ],
          "varlist":["ab","ABCD"],  (实体提取)
          "求解":["ab"], 
          "steps":[
            {},
            {"title":[{"ab":5.2,"cd":3},{"ab":-5.2,"cd":-3}]} (最后的步骤为答案, 多解存在用数组)
          ]
        },
    ]
    """
    # ss = " 。asdfb.,".strip(",，。 ")
    # print(ss)
    # 3. latex 证明
    # ss = "@aa12 d ~ f"
    # se = re.match(r"^(\w|\s)+", ss)
    # print(se.group())
    # se = re.sub(r"^(\w|\s)+","", ss)
    # print(se.string)
    # print(se)
    # exit()
    # printstr3 = "已知：四边形 $ABCD$ 中 ， $AD\\parallel BC , Ac=bf $，$AC=BD$ ，\n 是不是容易求证 ：$AB=DC$"
    # printstr3 = "某村计划建造如图所示的矩形蔬菜温室，要求长与宽的比为$2:1$．在温室内，沿前侧内墙保留$3m$宽的空地，其他三侧内墙各保留$1m$宽的通道．当矩形温室的长与宽各为多少米时，蔬菜种植区域的面积是$288m^{2}$？"
    # printstr3 = "证明：\\\n 联结 $CE$ \\\n $\\because \\angle{ACB}=90^{\\circ}\\qquad AE=BE$ \\\n $\\therefore CE=AE=\\frac{1}{2}AB$ \\\n 又 $\\because CD=\\frac{1}{2}AB$ \\\n $\\therefore CD=CE$ \\\n $\\therefore \\angle{CED}=\\angle{CDE}$ \\\n 又 $\\because A$ 、$C$ 、$D$ 成一直线 \\\n $\\therefore \\angle{ECA}=\\angle{CED}+\\angle{CDE}$ \\\n $=2\\angle{CDE}$ \\\n $\\angle{CDE}=\\frac{1}{2}\\angle{ECA}$ \\\n 又 $\\because EC=EA$ \\\n $\\therefore \\angle{ECA}=\\angle{EAC}$ \\\n $\\therefore \\angle{ADG}=\\frac{1}{2}\\angle{EAC}$ \\\n 又 $\\because AG$ 是 $\\angle{BAC}$ 的角平分线 \\\n $\\therefore \\angle{GAD}=\\frac{1}{2}\\angle{EAC}$ \\\n $\\therefore \\angle{GAD}=\\angle{GDA}$ \\\n $\\therefore GA=GD$"
    printstr3 = "$\\therefore \\angle{ECA}=\\angle{CED}+\\angle{CDE}$"
    # printstr3 = "$\\therefore CE=AE=\\frac{1}{2}AB$"
    handestr3 = "已知：四边形 $ABCD$ 中 ， $AD\\parallel BC$，$AC=BD$ ，\n 求证 ：$AB=DC$"
    # print(printstr3)
    # print(handestr3)
    solve_latex_prove(printstr3, handestr3)
    print("end")
    exit()

    # 1. 单行测验
    # printstr1 = "$\\therefore \\angle{ECA}=\\angle{CED}+\\angle{CDE}$"
    printstr1 = "$\\therefore CE=AE=\\frac{1}{2}AB$"
    print(printstr1)
    # 标准空格化
    printstr1 = latex2space(printstr1)
    # 先按概念实例合并表达式，再按句意分割，合并最小单元
    varlist = ["CE", "AE", "AB"]
    tmplist = latex2list_P(printstr1, varlist=varlist)
    print(tmplist)
    # # 3. 解析堆栈形式
    postfix = postfix_convert_P(tmplist)
    print(postfix)
    exit()
