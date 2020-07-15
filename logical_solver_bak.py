# coding:utf-8
"""
输入 需要完全描述集合元素，如判断同条直线必须至少两点相同，且。平行线相交，需指定锐角

"""
import copy
import itertools
import logging.handlers
import os
import re
import sys
from itertools import combinations
import jieba.posseg as pseg
# !pip install jsonpatch
import time
import jsonpatch
import operator
from latex_solver import latex2list_P, postfix_convert_P, latex2space, latex2unit
from latex_solver import solve_latex_formula2, solve_latex_equation
from latex_solver import latex_json, baspath, step_alist, step_blist, symblist, pmlist, addtypelist, funclist, operlist
from meta_property import triobj, properobj, setobj
from utils.path_tool import makesurepath
import json
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

mpl.rcParams[u'font.sans-serif'] = u'SimHei'
mpl.rcParams[u'axes.unicode_minus'] = False

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


def latex_fenci(instr):
    # 字符串 先以_为单位，再以 {} 分成单元。只以概念词，分一层，不递归
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
    if len(keyindexlist) > 0:
        stroutlist += strlist[keyindexlist[len(keyindexlist) - 1][1]:]
    else:
        stroutlist = strlist
    # 下标
    tlenth = len(stroutlist)
    if tlenth > 2:
        # 即 至少为下标3元组
        for i1 in range(tlenth - 2, 0, -1):
            if stroutlist[i1] == "_":
                stroutlist[i1 - 1] = " ".join(stroutlist[i1 - 1:i1 + 2])
                del stroutlist[i1 + 1]
                del stroutlist[i1]
    return stroutlist


def list_set_shrink(inlistset):
    "一级列表二级集合，集合传递缩并。"
    inlistset = [setins for setins in inlistset if setins != set()]
    lenth_paralist = len(inlistset)
    for indmain in range(lenth_paralist - 1, -1, -1):
        for indcli in range(indmain - 1, -1, -1):
            if operator.eq(inlistset[indcli], inlistset[indmain]):
                del inlistset[indmain]
                break
    return inlistset


def list_list_deliver(inlistset):
    "一级列表二级集合，集合传递缩并。如平行 等值"
    # print("list_list_deliver")
    inlistset = [setins for setins in inlistset if setins != set()]
    lenth_paralist = len(inlistset)
    for indmain in range(lenth_paralist - 1, 0, -1):
        for indcli in range(indmain - 1, -1, -1):
            if len(set(inlistset[indcli]).intersection(set(inlistset[indmain]))) > 0:
                inlistset[indcli] = list(set(inlistset[indcli]) | set(inlistset[indmain]))
                del inlistset[indmain]
                break
    return inlistset


def lines_deliver(inlistset):
    "一级列表二级列表，有序列表传递缩并。如 直线。只能缩并 子集，不能处理超出"

    def aisfirstb(incli, inmain):
        incli, inmain = copy.deepcopy(incli), copy.deepcopy(inmain)
        outlist = []
        for i1 in range(len(incli) + len(inmain)):
            if len(incli) > 0:
                if incli[0] not in inmain:
                    outlist.append(incli.pop(0))
                elif len(inmain) > 0 and incli[0] == inmain[0]:
                    outlist.append(incli.pop(0))
                    inmain.pop(0)
                else:
                    # 走main的流程
                    pass
            if len(inmain) > 0:
                if inmain[0] not in incli:
                    outlist.append(inmain.pop(0))
                elif len(incli) > 0 and incli[0] == inmain[0]:
                    outlist.append(inmain.pop(0))
                    incli.pop(0)
                else:
                    # 走下个流程
                    pass
        return outlist

    lenth_paralist = len(inlistset)
    for indmain in range(lenth_paralist - 1, 0, -1):
        for indcli in range(indmain - 1, -1, -1):
            inseclist = set(inlistset[indcli]).intersection(set(inlistset[indmain]))
            if len(inseclist) > 1:
                # 有两个共同点，排序
                indexlist = []
                for point in inseclist:
                    indexlist.append(inlistset[indcli].index(point))
                if indexlist[0] > indexlist[1]:
                    inlistset[indcli] = [inlistset[indcli][id1] for id1 in range(len(inlistset[indcli]) - 1, -1, -1)]
                indexlist = []
                for point in inseclist:
                    indexlist.append(inlistset[indmain].index(point))
                if indexlist[0] > indexlist[1]:
                    inlistset[indmain] = [inlistset[indmain][id1] for id1 in range(len(inlistset[indmain]) - 1, -1, -1)]
                # 合并
                inlistset[indcli] = aisfirstb(inlistset[indcli], inlistset[indmain])
                # 删除
                del inlistset[indmain]
                break
    return inlistset


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
            self._proper_keys, self._proper_trip, self._relation_trip, self._setobj = self.storage_oper("r")
        else:
            self._proper_keys, self._proper_trip = {}, {}
            self._relation_trip = {}
            self._questproperbj = {}
            self._questtriobj = {}
            self._setobj = {"等价集合": [], "全等集合": [], "全等三角形集合": [], "垂直集合": [], "平行集合": [], "角集合": set(),
                            "锐角集合": set(), "钝角集合": set(), "直角集合": set(), "平角集合": set(), "直角三角形集合": set(),
                            "等腰三角形集合": set(), "等边三角形集合": set(), "圆集合": set(), "余角集合": [], "补角集合": [],
                            "表达式集合": set(), "弧集合": set(), "直径集合": set(), "弦切角集合": [], "等值集合": []}
            self._stopobj = {"等价集合": [], "全等集合": [], "全等三角形集合": [], "垂直集合": [], "平行集合": [], "角集合": set(),
                             "锐角集合": set(), "钝角集合": set(), "直角集合": set(), "平角集合": set(), "直角三角形集合": set(),
                             "等腰三角形集合": set(), "等边三角形集合": set(), "圆集合": set(), "余角集合": [], "补角集合": [],
                             "表达式集合": set(), "弧集合": set(), "直径集合": set(), "弦切角集合": [], "等值集合": []}
            self._initobj = {}
            self._step_node = []

    def storage_oper(self, operstr):
        """硬件：存储交互操作"""
        if operstr == "r":
            # 属性 是 xx 特制 实例类 不代表 等价
            proper_trip = {}
            couindex = 0
            for i1 in properobj:
                for i2 in properobj[i1]:
                    couindex += 1
                    proper_trip[str(couindex)] = [i1, "有属性", i2, properobj[i1][i2]]
            proper_keys = list(properobj.keys())
            relation_trip = {str(id1): i1 for id1, i1 in enumerate(triobj)}
            return proper_keys, proper_trip, relation_trip, setobj
        elif operstr == "w":
            return True

    def property_oper(self, properobj, addc={}, delec=[]):
        """内存：triple交互操作"""
        for oneproper in addc:
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

    def tri2set_oper(self, basic_set, oldsetobj, stopobj, addc=[], delec=[]):
        """内存：triple交互操作"""
        so_obj = {}
        newout = [oldsetobj, so_obj, stopobj]
        keydic = {i1.rstrip("集合"): i1 for i1 in basic_set}
        for oneitems in addc:
            if "因为" in oneitems:
                onetri = oneitems["因为"]
                newsetobj = newout[0]
            elif "所以" in oneitems:
                onetri = oneitems["所以"]
                newsetobj = newout[1]
            elif "求证" in oneitems:
                onetri = oneitems["求证"]
                newsetobj = newout[2]
            else:
                print(oneitems)
                raise Exception("没有考虑的情况")
            if onetri[2] in keydic:
                if keydic[onetri[2]] not in newsetobj:
                    if basic_set[keydic[onetri[2]]]["结构形式"] == "一级集合":
                        newsetobj[keydic[onetri[2]]] = set()
                    elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表":
                        newsetobj[keydic[onetri[2]]] = []
                    elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表二级集合":
                        newsetobj[keydic[onetri[2]]] = []
                    elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表二级列表":
                        newsetobj[keydic[onetri[2]]] = []
                    else:
                        print(onetri)
                        raise Exception("没有考虑的情况")
                if basic_set[keydic[onetri[2]]]["结构形式"] == "一级集合":
                    newsetobj[keydic[onetri[2]]].add(onetri[0])
                elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表":
                    newsetobj[keydic[onetri[2]]].append(onetri[0])
                elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表二级集合":
                    newsetobj[keydic[onetri[2]]].append(set())
                    for i1 in onetri[0]:
                        newsetobj[keydic[onetri[2]]][-1].add(i1)
                elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表二级列表":
                    newsetobj[keydic[onetri[2]]].append([])
                    for i1 in onetri[0]:
                        newsetobj[keydic[onetri[2]]][-1].append(i1)
                else:
                    print(onetri)
                    raise Exception("没有考虑的情况")
            else:
                print(onetri)
                raise Exception("没有考虑的情况")
        return newout

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

    def gene_instance_info(self, ins_key, ):
        infolist = []
        return infolist


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
        self._func_map = {
            "名字对称": self.name_symmetric,
            "多边形生成元素": self.polyname2elements,
        }

    def name_symmetric(self, strs):
        instr = strs.strip()
        # 传入时已做了预防
        in_list = latex_fenci(instr)
        lenth = len(in_list)
        newstr = " ".join([in_list[lenth - 1 - i1] for i1 in range(lenth)])
        return sorted([instr, newstr])[0]

    def name_cyc_one(self, strs):
        "多边形唯一命名"
        instr = strs.strip()
        # 传入时已做了预防
        in_list = latex_fenci(instr)
        lenth = len(in_list)
        in_list = in_list + in_list[:-1]
        outlist = []
        for i1 in range(lenth):
            outlist.append(" ".join(in_list[i1:i1 + lenth]))
            outlist.append(" ".join(in_list[i1:i1 + lenth][::-1]))
        return sorted(outlist)[0]

    def name_normal(self, strs):
        # 标准形式 { 角@A B C }, 返回 {角@ABC}。 包裹是为了防止外部对内的拆分 如 _^
        if len(latex_fenci(strs)) == 1 and strs[0] == "{":
            strs = strs.strip("{ }")
            tlist = strs.split("@")
            key = tlist[0]
            tail = "@".join(tlist[1:])
        return "{" + key + "@" + self.name_symmetric(tail).replace(" ", "") + "}"

    def polyname2elements(self, *args):
        # 表示字符串
        strs = args[0]
        # 预判{}，后面处理不带，直至返回才加上
        in_list = strs.strip().split()
        lenth = len(in_list)
        # 点
        outlist = []
        for i1 in in_list:
            outlist.append(["{ " + i1 + " }", "是", "点"])
        # 边
        in_list.append(in_list[0])
        for i1 in range(lenth):
            side = "@".join(in_list[i1:i1 + 2])
            side = self.name_symmetric(side)
            outlist.append(["{ " + side + " }", "是", "边"])
        # 角
        in_list.append(in_list[1])
        for i1 in range(lenth):
            angle = "@".join(in_list[i1:i1 + 3])
            angle = self.name_symmetric(angle)
            outlist.append(["{ " + angle + " }", "是", "角"])
        return outlist

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
                if key == "\\triangle":
                    strlist[i1 + 1] = strlist[i1 + 1].strip("{ }")
                    strlist[i1 + 1] = self.name_cyc_one(" ".join(strlist[i1 + 1])).replace(" ", "")
                elif key == "\\angle":
                    strlist[i1 + 1] = strlist[i1 + 1].strip("{ }")
                    strlist[i1 + 1] = self.name_symmetric(" ".join(strlist[i1 + 1])).replace(" ", "")
                elif key == "\\bigodot":
                    # 圆的第一个元素为 圆心，其余的为有序点
                    strlist[i1 + 1] = strlist[i1 + 1].strip("{ }")
                    # print(strlist[i1 + 1])
                    # print(type(strlist[i1 + 1]))
                    ttlist = strlist[i1 + 1].split(" ")
                    strlist[i1 + 1] = ttlist[0] + self.name_cyc_one(" ".join(ttlist[1:])).replace(" ", "")
                    # print(strlist[i1 + 1])
                    # raise 6666
                else:
                    strlist[i1 + 1] = strlist[i1 + 1]
                if not strlist[i1 + 1].startswith("{"):
                    strlist[i1 + 1] = "{ " + self.latex_map[key] + "@" + strlist[i1 + 1] + " }"
                else:
                    strlist[i1 + 1] = "{ " + self.latex_map[key] + "@" + strlist[i1 + 1].strip("{ }") + " }"
                strlist[i1 + 1] = self.name_normal(strlist[i1 + 1])
                matchlist.append([strlist[i1 + 1], "是", self.latex_map[key]])
                keyindexlist.append(i1)
        strlist = [strlist[i1] for i1 in range(slenth) if i1 not in keyindexlist]
        return " ".join(strlist), matchlist

    def get_step(self, strlist):
        """获取 步骤 词 三类标签"""
        questiontype = ["求", "什么", "多少"]
        quest_notype = ["要求"]
        slenth = len(strlist)
        key_type = []
        key_index = []
        for i1 in range(slenth):
            if strlist[i1] in step_alist:
                key_type.append("已知")
                key_index.append(i1)
            elif strlist[i1] in step_blist:
                key_type.append("导出")
                key_index.append(i1)
            elif strlist[i1] in questiontype and strlist[i1] not in quest_notype:
                key_type.append("求")
                key_index.append(i1)
            else:
                pass
        lenth = len(key_index)
        # 高于等于2，正常去连续索引
        for i1 in range(lenth - 1, 0, -1):
            if key_type[i1] == key_type[i1 - 1]:
                del key_type[i1]
                del key_index[i1]
        lenth = len(key_index)
        # 大于0 的连接
        outlist = []
        if lenth > 0:
            for i1 in range(lenth - 1):
                outlist.append({key_type[i1]: " ".join(strlist[key_index[i1] + 1:key_index[i1 + 1]])})
            outlist.append({key_type[lenth - 1]: " ".join(strlist[key_index[lenth - 1] + 1:])})
        # 连接完后 index只考虑第一个或空
        if lenth == 0:
            outlist.append({"已知": " ".join(strlist)})
        elif key_index[0] != 0:
            # 长度大于零，索引不等于0
            if "已知" == key_type[0]:
                if len(outlist) > 0:
                    outlist[0][key_type[0]] = strlist[0:key_index[0]] + " " + outlist[0][key_type[0]]
                else:
                    outlist.append({key_type[0]: " ".join(strlist[0:key_index[0]])})
            else:
                # 插入第一条
                outlist.insert(0, {"已知": " ".join(strlist[0:key_index[0]])})
        else:
            pass
        return outlist

    def latex_default_property(self, inlist):
        '[tnewstr, "是", "点"]'
        # 1. 字符串 得出索引
        # print("latex_default_property")
        # print(inlist)
        elemindexlist = []
        alllist = set(step_alist + step_blist + symblist + pmlist + addtypelist + funclist + operlist)
        legth = len(inlist)
        for i1 in range(legth):
            if inlist[i1] not in alllist and re.match("^({\s[a-zA-Z]|[a-zA-Z])", inlist[i1]):
                elemindexlist.append(i1)
        # 2. 根据连续性得出默认的 线段 点 属性,三点以上必有说明，在前一步已经赋值。
        lengindex = len(elemindexlist)
        write_json = []
        # 先遍历非最后一个，最后单独考虑
        #  大于一的结尾处理
        if lengindex > 1:
            if elemindexlist[-2] + 1 == elemindexlist[-1]:
                # 线段后续会已处理，略过
                pass
            else:
                tnewstr = "{ 点@" + inlist[elemindexlist[-1]] + " }"
                tnewstr = self.name_normal(tnewstr)
                write_json.append([tnewstr, "是", "点"])
                inlist[elemindexlist[-1]] = tnewstr
        if lengindex > 2:
            # 至少3个元素
            for i1 in range(lengindex - 2, 0, -1):
                if elemindexlist[i1] + 1 != elemindexlist[i1 + 1] and elemindexlist[i1 - 1] + 1 != elemindexlist[i1]:
                    tnewstr = "{ 点@" + inlist[elemindexlist[i1]] + " }"
                    tnewstr = self.name_normal(tnewstr)
                    inlist[elemindexlist[i1]] = tnewstr
                    write_json.append([tnewstr, "是", "点"])
                elif elemindexlist[i1] + 1 == elemindexlist[i1 + 1]:
                    # { 角@A B C }
                    tnewstr = "{ 线段@" + " ".join(inlist[elemindexlist[i1]:elemindexlist[i1] + 2]) + " }"
                    tnewstr = self.name_normal(tnewstr)
                    inlist[elemindexlist[i1]] = tnewstr
                    del inlist[elemindexlist[i1] + 1]
                    write_json.append([tnewstr, "是", "线段"])
                else:
                    pass
        # 大于一的起始处理
        if lengindex > 1:
            if elemindexlist[0] + 1 == elemindexlist[1]:
                tnewstr = "{ 线段@" + " ".join(inlist[elemindexlist[0]:elemindexlist[0] + 2]) + " }"
                tnewstr = self.name_normal(tnewstr)
                inlist[elemindexlist[0]] = tnewstr
                del inlist[elemindexlist[0] + 1]
                write_json.append([tnewstr, "是", "线段"])
            else:
                tnewstr = "{ 点@" + inlist[elemindexlist[0]] + " }"
                tnewstr = self.name_normal(tnewstr)
                write_json.append([tnewstr, "是", "点"])
                inlist[elemindexlist[0]] = tnewstr
        if lengindex == 1:
            # 只能是点
            tnewstr = "{ 点@" + inlist[elemindexlist[0]] + " }"
            tnewstr = self.name_normal(tnewstr)
            write_json.append([tnewstr, "是", "点"])
            inlist[elemindexlist[0]] = tnewstr
        else:
            # 等于0 或已处理的情况
            pass
        return inlist, write_json

    def latex_extract_property(self, instr):
        """单句： 空格标准分组后 返还 去掉抽象概念的实体latex"""
        # 1. token 预处理
        tinlist = re.split(',|，|\n|\t', latex2space(instr))
        # tinlist = re.split(',|，|\n|\t', instr)
        # 2. 每个断句
        latexlist = []
        outjson = []
        for i1 in tinlist:
            # 3.1 分割 关键词 索引。句意级。
            tstr = i1
            # print(15)
            # print(tstr)
            for word in self.latex_map:
                # 每个 属性词
                if "n" == self.nominal[self.latex_map[word]]:
                    # print(16)
                    tstrli = latex_fenci(tstr)
                    # print(tstrli, word)
                    tstr, tjson = self.get_extract(tstrli, word)
                    # print(tstr, tjson)
                    outjson += tjson
            latexlist.append(tstr)
        return latexlist, outjson

    def latex_extract_word(self, instr):
        """单句： 空格标准分组后 返还 去掉抽象概念的实体latex"""
        # 1. token 预处理
        tinlist = re.split(',|，|、|;|；|\n|\t', latex2space(instr))
        # 2. 每个断句
        latexlist = []
        for i1 in tinlist:
            # 3.1 分割 关键词 索引。句意级。
            tstr = i1
            for word in self.latex_map:
                # 每个 属性词
                if "n" == self.nominal[self.latex_map[word]]:
                    tstrli = latex_fenci(tstr)
                    tstr = " ".join(tstrli)
            latexlist.append(tstr)
        return latexlist

    def json2space(self, ins_json, basic_space_ins, space_ins):
        write_json = {
            "add": {
                "properobj": [], "triobj": ins_json,
                "quest_properobj": {}, "quest_triobj": {},
            },
            "dele": {"properobj": {}, "triobj": [], "quest_properobj": {}, "quest_triobj": []},
        }
        # 默认： write_json["add"]["triobj"] = [['{线段@BP}', '是', '线段'], ['{线段@PQ}', '是', '线段']]
        # 特殊声明： write_json["add"]["triobj"] = [{'因为': ['{线段@BP}', '是', '线段']}, '所以': ['{线段@PQ}', '是', '线段']}, '求证': ['{线段@PQ}', '是', '线段']}]
        # 1. 先写属性
        space_ins.property_oper(space_ins._proper_trip, addc=write_json["add"]["properobj"],
                                delec=write_json["dele"]["properobj"])
        # 2. 修正元组实体
        # 3. 再写元组
        space_ins._setobj, _, space_ins._stopobj = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                                          space_ins._stopobj,
                                                                          addc=write_json["add"]["triobj"],
                                                                          delec=write_json["dele"]["triobj"])
        logger1.info("write triple: %s" % write_json["add"]["triobj"])

    def fenci2triple(self, fenci_list, space_ins):
        """ 解析：元素属性 词性逻辑关系 到 内存格式"""
        print(88)
        # 1. 问句分割索引
        print(fenci_list)
        newfenci = []
        for i1 in fenci_list:
            newfenci += self.get_step(i1)
        print(newfenci)
        # 2. 三元组提取
        # 如果只有 运算符号，xx 是 表达式
        triobj = []
        quest_triobj = []
        properobj = {}
        quest_properobj = {}
        for idwt, word_tri in enumerate(newfenci):
            lenthwt = len(word_tri)
            print(lenthwt)
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
                # # 普通 字符 暂时略过
                # if index_question != -1:
                #     # 问句类型处理
                #     pass
                # else:
                #     # 陈述句处理
                pass
        # 暂时没想到
        # print(properobj)
        # write_json = {"add": {"properobj": {"test": {"red": 1, "blue": 2, "green": 3}},
        #                       "triobj": [['三角形n', '属于', 'n边形'], ['三角形', '属于', 'n边形']]},
        #               "dele": {"properobj": ["n边形"], "triobj": [['三角形', '属于', 'n边形']]}}
        # write_json = {
        #     "add": {
        #         "properobj": properobj, "triobj": triobj,
        #         "quest_properobj": quest_properobj, "quest_triobj": quest_triobj,
        #     },
        #     "dele": {"properobj": {}, "triobj": [], "quest_properobj": {}, "quest_triobj": []},
        # }
        # print(565656)
        # print(write_json)
        # return write_json
        return triobj


class LogicalInference(object):
    """ 推理内核: """

    def __init__(self):
        self.language = NLPtool()
        self.gstack = GStack()
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        # self.treesig = None
        self.treesig = True
        # self.debugsig = True
        self.debugsig = False
        self.answer_json = []
        if not self.gstack.is_inspace_list(space_name, scene_name, field_name):
            space_ins = BasicalSpace(space_name=space_name, field_name=field_name, scene_name=scene_name)
            self.gstack.loadspace(space_name, scene_name, field_name, space_ins)
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        self.listset_obj = [objset for objset in basic_space_ins._setobj if
                            basic_space_ins._setobj[objset]["结构形式"].startswith("一级列表二级")]
        self.set_obj = [objset for objset in basic_space_ins._setobj if
                        basic_space_ins._setobj[objset]["结构形式"].startswith("一级集合")]
        space_name = "customer"
        if not self.gstack.is_inspace_list(space_name, scene_name, field_name):
            space_ins = BasicalSpace(space_name=space_name, field_name=field_name, scene_name=scene_name)
            self.gstack.loadspace(space_name, scene_name, field_name, space_ins)

    def __call__(self, *args, **kwargs):
        """输入为语言的 list dic 数组: text latex"""
        analist = args[0]
        logger1.info("initial analyzing: %s" % analist)
        # 0. 处理句间 关系，写入部分实体。 基于符号类型的区分标签。结果全部写入内存。
        self.sentence2normal(analist)
        logger1.info("initial clean sentence: %s" % analist)
        return self.inference()

    def analysis_tree(self, nodejson, edgelist, checkpoints=[]):
        # 1.1. 设置默认自动模块, 默认识别的集合
        cannotigore = list(nodejson["求证"]["condjson"].keys())
        self.ignoreset = ["默认集合", "点集合", "角集合", "弧集合", "线段集合", "直线集合", "三角形集合", "锐角集合", "钝角集合"]
        self.ignoreset = [item for item in self.ignoreset if item not in cannotigore]
        # 1.2. 默认连接的知识点
        self.defaultpoint = ["@@已知", "@@求证", "@@同角表示",
                             "@@直角相等", "@@平角相等", "@@自等性质", "@@角度角类型", "@@余角性质",
                             "@@直线得出表达式", "@@直线得出补角", "@@补角属性",
                             "@@平行线间平行线等值", "@@平行属性",
                             "@@垂直性质", "@@垂直直角的属性", "@@垂直直线的线段属性",
                             "@@正方形平行属性", "@@正方形垂直属性", "@@正方形等边属性", "@@正方形直角属性",
                             "@@直角三角形属性必要条件",
                             "@@同位角对顶角内错角属性", "@@等值钝角传递", "@@等值锐角传递",
                             "@@表达式传递", "@@表达式性质",
                             "@@等边三角形充分条件角", "@@等边三角形充分条件边",
                             "@@等腰三角形必要条件角", "@@等腰三角形必要条件边",
                             "@@等腰三角形充分条件角", "@@等腰三角形充分条件边",
                             "@@相似三角形必要条件", "@@相似三角形充分条件",
                             "@@全等三角形充分条件边角边", "@@全等三角形充分条件角边角", "@@全等三角形必要条件",
                             "@@圆内接四边形的性质", "@@圆等弧对等角", "@@圆周角求和关系", "@@圆心角求和关系", "@@圆弧关系",
                             "@@弦切角性质",
                             ]
        for key in self.listset_obj:
            if setobj[key]["结构形式"] in ["一级列表二级集合"] and "二级传递" in setobj[key]["函数"]:
                self.defaultpoint.append("@@{}间传递".format(key.replace("集合", "")))
        self.defaultpoint.append("@@{}{}等反传递".format("平行", "垂直"))
        self.defaultpoint.append("@@{}{}反等传递".format("余角", "等值"))
        self.defaultpoint.append("@@{}{}反等传递".format("补角", "等值"))
        self.defaultpoint.append("@@{}{}反等传递".format("垂直", "平行"))
        self.defaultpoint = [point for point in self.defaultpoint if point not in checkpoints]
        # 1.3 默认 不校验 的 答案 中的某些集合
        self.ignoreerror = ["表达式集合", "角集合"]
        # 2. 答案json，分布到 原始json上，返回报告，仅步骤，不考虑 连通性。
        logger1.info("答案json: {}".format(self.answer_json))
        logger1.info("原始所有节点: {} {}".format(len(nodejson), nodejson))
        defaultnode = []
        defaultpointset = set(self.defaultpoint)
        for node in nodejson:
            if defaultpointset.issuperset(set(nodejson[node]["points"])):
                defaultnode.append(node)
        # 4. 简化默认节点条件
        simple_nodejson, self.answer_json = self.simplify_node(nodejson, self.answer_json, self.ignoreset)
        logger1.info("简化 answer_json：{}".format(self.answer_json))
        logger1.info("简化所有节点：{} {}".format(len(simple_nodejson), simple_nodejson))
        # 5. 节点信息判断
        cond_node, reportjson1, mention_node = self.add_answer2node(simple_nodejson, self.answer_json)
        logger1.info("答案可以导出的考点节点: {}".format(len(mention_node), mention_node))
        usefulnode = mention_node + defaultnode
        logger1.info("答案可用节点: {} {}".format(len(usefulnode), usefulnode))
        # 6. 遍历 原始edgelist，得到最近点的路径，删除每条路径上的默认点，剩余点取数量阈值作为连接的判断。
        reportjson, suminfo = self.find_answer_path(simple_nodejson, usefulnode, mention_node)
        reportjson = reportjson1 + reportjson
        logger1.info("字面报告：{} {} {}".format(len(reportjson), suminfo, reportjson))
        # pprint(reportjson)
        return json.dumps([reportjson, suminfo], ensure_ascii=False)

    def find_answer_path(self, nodejson, usefulnode, mention_node):
        " 只能从已知找，如果中断不连通 由于存在自引循环，无法从另一个联通区域继续往下推 "
        # 2. 连接下行树
        logger1.info("find_answer_path")
        nodename = "已知"
        logger1.info("答案涉及节点：{} {}".format(len(mention_node), {node: nodejson[node] for node in mention_node}))
        answer_nodejson = {node: nodejson[node] for node in nodejson if node in usefulnode}
        logger1.info("答案涉及节点+默认节点：{} {}".format(len(answer_nodejson), answer_nodejson))
        # print(555)
        # for node in nodejson:
        #     if "等值集合" in nodejson[node]["outjson"]:
        #         for tlist in nodejson[node]["outjson"]["等值集合"]:
        #             # sigt1 = 1 if len(set(["{角@BPM}", "{角@NQP}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt2 = 1 if len(set(["{角@MBP}", "{角@NPQ}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt3 = 1 if len(set(["{角@BMP}", "{角@PNQ}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt4 = 1 if len(set(["{线段@MP}", "{线段@NQ}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt5 = 1 if len(set(["{线段@BP}", "{线段@PQ}"]).intersection(set(tlist))) == 2 else 0
        #             sigt6 = 1 if len(set(['{角@DCE}', '{角@CED}']).intersection(set(tlist))) == 2 else 0
        #             sigta = sum([sigt6])
        #             # sigta = sum([sigt1, sigt2, sigt3, sigt4, sigt5, sigt6])
        #             if sigta > 0:
        #                 print(node, nodejson[node])
        #                 break
        sstime = time.time()
        G1, treportjson = self.gene_downtree_semi_from(answer_nodejson, nodename)
        fintime = time.time() - sstime
        logger1.info("生成下行树 耗时 {}s：".format(fintime))
        logger1.info("下行树 点{}，边{}，列表{}".format(len(G1.nodes), len(G1.edges), [nodejson[node] for node in G1.nodes]))
        defaultnode = []
        defaultpointset = set(self.defaultpoint)
        for node in nodejson:
            if defaultpointset.issuperset(set(nodejson[node]["points"])):
                defaultnode.append(node)
        downdefault = [node for node in G1.nodes if node in defaultnode]
        downcheck = [node for node in G1.nodes if node not in downdefault]
        logger1.info("下行树默认节点 {} {}".format(len(downdefault), downdefault))
        logger1.info("下行树考点节点 {} {}".format(len(downcheck), downcheck))
        reportjson = treportjson
        con_des_node = [node for node in G1.nodes if node in mention_node]
        logger1.info(
            "知识点掌握，且 连通已知 的节点：{}, {}".format(len(con_des_node), {node: nodejson[node] for node in con_des_node}))
        nocon_node = [node for node in mention_node if node not in G1.nodes]
        for node in nocon_node:
            tstb = self.nodejson2reportstr(nodejson[node]["condjson"])
            tstt = self.nodejson2reportstr(nodejson[node]["outjson"])
            tstt = "因为:" + tstb + ". 所以:" + tstt
            if tstt != "":
                tpoint = nodejson[node]["points"]
                tmjson = {"content": tstt, "point": ",".join(tpoint), "istrue": "未连通描述正确"}
                reportjson.append(tmjson)
        logger1.info("知识点掌握，但没有 连通已知 的节点：{}, {}".format(len(nocon_node), {node: nodejson[node] for node in nocon_node}))
        nodename = "求证"
        if nodename not in nocon_node and nodename in G1.nodes:
            logger1.info("证明成功")
            suminfo = "此题证明完全正确。"
        else:
            logger1.info("证明失败")
            suminfo = "此题没有完成证明。"
            sstime = time.time()
            # G2 = self.gene_uptree_from(answer_nodejson, nodename)
            G2 = self.gene_uptree_from(nodejson, nodename)
            fintime = time.time() - sstime
            logger1.info("生成上行树 耗时 {}s：".format(fintime))
            unconnectnode = [node for node in G2.nodes if node not in G1.nodes]
            pointnode = [node for node in unconnectnode if
                         set(self.defaultpoint).issuperset(set(nodejson[node]["points"]))]
            logger1.info("未下连通考点 {}，{}".format(len(pointnode), pointnode))
            logger1.info([nodejson[node] for node in pointnode])
            nocon_no_des_node = []
            # nocon_no_des_node = [node for node in pointnode if node not in nocon_node]
            # logger1.info("未下连通且未描述考点 {}，{}".format(len(nocon_no_des_node), nocon_no_des_node))
            for node in nocon_no_des_node:
                tstb = self.nodejson2reportstr(nodejson[node]["condjson"])
                tstt = self.nodejson2reportstr(nodejson[node]["outjson"])
                tstt = "因为:" + tstb + ". 所以:" + tstt
                if tstt != "":
                    tpoint = nodejson[node]["points"]
                    tmjson = {"content": tstt, "point": ",".join(tpoint), "istrue": "可选未连通未描述知识点"}
                    reportjson.append(tmjson)
        # logger1.info("{} {}".format(suminfo, reportjson))
        return reportjson, suminfo
        # # 2. 判断 对错
        # pos = nx.kamada_kawai_layout(G1)
        # plt.figure(figsize=(10, 6))
        # nx.draw(G1, pos, font_color='y', linewidths=0.1, style="dashdot", alpha=0.9, font_size=15, node_size=500,
        #         with_labels=True)
        # plt.axis('on')
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig("../345.svg", dpi=600, format='svg')
        # # plt.show()
        # raise 456
        # # 颜色修改
        # valn_map = {node: "#ff0000" for node in gpath}
        # node_values = [valn_map.get(node, "#0000ff") for node in G1.nodes()]
        # # vale_map = {(gpath[idn], gpath[idn + 1]): "#ff0000" for idn in range(len(gpath) - 1)}
        # vale_map = {edge: "#ff0000" for edge in gedge}
        # edge_values = [vale_map.get(edge, '#000000') for edge in G1.edges()]
        # # 画图
        # pos = nx.kamada_kawai_layout(G1)
        # plt.figure(figsize=(100, 60))
        # nx.draw(G1, pos, node_color=node_values, edge_color=edge_values, font_color='y', linewidths=0.1,
        #         style="dashdot", alpha=0.9, font_size=15, node_size=500, with_labels=True)
        # # nx.draw(G1, pos, node_color='b', edge_color='r', font_size=18, node_size=20, with_labels=True)
        # plt.axis('on')
        # plt.xticks([])
        # plt.yticks([])
        # # plt.savefig("../123.png")
        # plt.savefig("../123.svg", dpi=600, format='svg')
        # plt.show()
        # return None

    def add_answer2node(self, nodejson, answer_json):
        " 包含自引条件 可导出的节点 说明掌握知识点，但可能没找到对应的条件 "
        # 1. 答案字面条件 不存在判断
        reportjson = []
        for item in answer_json:
            tca_item = list(item.values())[0]
            havsig = 0
            for node in nodejson:
                # 1.1 提取 该节点相关的答案
                # 因为 所以 都可以算为后续步骤的 已知条件
                tmc_item = nodejson[node]["condjson"]
                tmo_item = nodejson[node]["outjson"]
                condi_cmmon = self.a_commonset_b(tmc_item, tca_item)
                out_cmmon = self.a_commonset_b(tmo_item, tca_item)
                if condi_cmmon or out_cmmon:
                    havsig = 1
                    break
            if havsig == 0:
                # 得出同一结论可能用到的是隔层原因，所以无法判断知识点。
                tstt = self.nodejson2reportstr(tca_item)
                # print(tstt)
                if tstt != "":
                    tmjson = {"content": tstt, "point": "只有正确才可能判断知识点", "istrue": "描述错误"}
                    reportjson.append(tmjson)
        # 2. 默认条件
        defaultpointset = set(self.defaultpoint)
        logger1.info("defaultpointset: {}".format(defaultpointset))
        tcondilist = []
        for node in nodejson:
            if defaultpointset.issuperset(set(nodejson[node]["points"])):
                tcondilist.append(nodejson[node]["outjson"])
        default_supersets = self.genesuperset(tcondilist)
        # 3. 答案json，分布到 原始精简json上
        cond_node = []
        mention_node = []
        for node in nodejson:
            # 1.1 提取 该节点相关的答案信息
            tcondilist = [default_supersets]
            tmc_item = nodejson[node]["condjson"]
            for item in answer_json:
                # 因为 所以 都可以算为后续步骤的 已知条件
                tca_item = list(item.values())[0]
                common_elem = self.a_commonset_b(tmc_item, tca_item)
                if common_elem:
                    tcondilist.append(common_elem)
            toutlist = []
            tmo_item = nodejson[node]["outjson"]
            for item in answer_json:
                # 只有所以 都可以算为 输出条件
                if "outjson" in item:
                    common_elem = self.a_commonset_b(tmo_item, item["outjson"])
                    if common_elem:
                        toutlist.append(common_elem)
            # 1.2 判断该节点的输入是否有效 生成超集
            condi_supersets = self.genesuperset(tcondilist)
            out_supersets = self.genesuperset(toutlist)
            if self.a_supset_b(condi_supersets, tmc_item) and self.a_supset_b(tmo_item, out_supersets):
                cond_node.append(node)
                # 节点不忽略，只在报告里忽略。
                if not defaultpointset.issuperset(set(nodejson[node]["points"])):
                    # tpoint = [point.replace("@@", "") for point in nodejson[node]["points"]]
                    for onejson in toutlist:
                        tstt = self.nodejson2reportstr(onejson)
                        if tstt != "":
                            mention_node.append(node)
        mention_node = list(set(mention_node))
        # print(reportjson)
        return cond_node, reportjson, mention_node

    def get_condition_tree(self):
        " 根据条件构建 思维树 "
        # 1. 定义空间
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # 2. 使用nodelist
        space_ins._step_node = {str(idn): cont for idn, cont in enumerate(space_ins._step_node)}
        listset_obj = [objset for objset in basic_space_ins._setobj if
                       basic_space_ins._setobj[objset]["结构形式"].startswith("一级列表二级")]
        set_obj = [objset for objset in basic_space_ins._setobj if
                   basic_space_ins._setobj[objset]["结构形式"].startswith("一级集合")]
        for obj in space_ins._initobj:
            if obj in listset_obj:
                for idt, tlist in enumerate(space_ins._initobj[obj]):
                    space_ins._initobj[obj][idt] = list(tlist)
            elif obj in set_obj:
                space_ins._initobj[obj] = list(space_ins._initobj[obj])
        for obj in copy.deepcopy(space_ins._stopobj):
            if obj in listset_obj:
                for idt, tlist in enumerate(space_ins._stopobj[obj]):
                    space_ins._stopobj[obj][idt] = list(tlist)
                    if len(tlist) == 0:
                        del space_ins._stopobj[obj][idt]
                if len(space_ins._stopobj[obj]) == 0:
                    del space_ins._stopobj[obj]
            elif obj in set_obj:
                space_ins._stopobj[obj] = list(space_ins._stopobj[obj])
                if len(space_ins._stopobj[obj]) == 0:
                    del space_ins._stopobj[obj]
        # 用于无条件连接
        space_ins._initobj["默认集合"] = [0]
        space_ins._step_node["已知"] = {"condjson": {}, "points": ["@@已知"], "outjson": space_ins._initobj}
        space_ins._step_node["求证"] = {"condjson": space_ins._stopobj, "points": ["@@求证"], "outjson": {}}
        # 全量节点
        # instra = """"""
        # instra = instra.replace("'", '"').replace("\\", "\\\\")
        # print(instra)
        # space_ins._step_node = json.loads(instra, encoding="utf-8")
        nodejson = space_ins._step_node
        # print(nodejson)
        logger1.info("原始节点数:{}, {}".format(len(nodejson), nodejson))
        # 4. 根据已知节点向下生成树
        nodename = "已知"
        sstime = time.time()
        G1 = self.gene_downtree_from(nodejson, nodename)
        fintime = time.time() - sstime
        logger1.info("下行思维树生成 耗时 {}mins".format(fintime / 60))
        logger1.info("下行树节点{}，边{}".format(len(G1.nodes()), len(G1.edges())))
        nolinknode = [node for node in nodejson if node not in G1.nodes()]
        logger1.info("未连接node {} {}".format(nolinknode, [nodejson[node] for node in nodejson if node in nolinknode]))
        solvelast = [1 for i1 in G1.edges() if i1[1] == "求证"]
        if sum(solvelast) > 0:
            logger1.info("思维树成功生成！")
            # # 耗时过长，忽略精简
            # logger1.info("精简树...")
            # delcounter = 0
            # stopcounter = 9999
            # logger1.info("ori_node num: {}. ori_edge num: {}.".format(len(G1.nodes()), len(G1.edges())))
            # G1 = self.deltree_layer(G1, delcounter, stopcounter)
            # logger1.info("final_node num: {}. final_edge num: {}.".format(len(G1.nodes()), len(G1.edges())))
            # pos = nx.random_layout(G1)
            # pos = nx.kamada_kawai_layout(G1)
            # plt.figure(figsize=(100, 60))
            # nx.draw(G1, pos, font_size=10, with_labels=True)
            # nx.draw(G1, pos, node_color='b', edge_color='#000000', font_color='y', linewidths=1, style="dashdot",
            #         alpha=0.9, font_size=15, node_size=500, with_labels=True)
            # plt.axis('on')
            # plt.xticks([])
            # plt.yticks([])
            # # plt.savefig("../123.png")
            # # plt.savefig("../123.eps", dpi=600,format='eps')
            # plt.savefig("../123.svg", dpi=600, format='svg')
            # # plt.show()
            edglist = [[*edg] for edg in G1.edges()]
            with open("../nodejson.json", "w") as f:
                json.dump(nodejson, f, ensure_ascii=False)
            with open("../edgejson.json", "w") as f:
                json.dump(edglist, f, ensure_ascii=False)
            return json.dumps(nodejson, ensure_ascii=False), json.dumps(edglist, ensure_ascii=False)
        logger1.info("没有发现解答路径！")
        # # 调试部分
        # nodename = "求证"
        # sstime = time.time()
        # G2 = self.gene_uptree_from(nodejson, nodename)
        # fintime = time.time() - sstime
        # logger1.info("上行思维树生成 耗时 {}mins".format(fintime / 60))
        # logger1.info("上行树节点{}，边{}".format(len(G2.nodes()), len(G2.edges())))
        # sstime = time.time()
        # G3 = self.gene_fulltree_from(nodejson)
        # fintime = time.time() - sstime
        # logger1.info("全量思维树生成 耗时 {}mins".format(fintime / 60))
        # logger1.info("全量树节点{}，边{}".format(len(G3.nodes()), len(G3.edges())))
        # mislist1 = [node for node in G2.nodes() if node not in G1.nodes()]
        # mislist2 = [node for node in G2.nodes() if node not in G3.nodes()]
        # commlist = [node for node in G2.nodes() if node in G1.nodes()]
        # print(G1.nodes)
        # print(mislist1)
        # print(mislist2)
        # print(commlist)
        # print(len(mislist1), len(mislist2), len(commlist))
        print(555)
        for node in G1.nodes():
            if "等值集合" in nodejson[node]["outjson"]:
                # print(node, nodejson[node]["outjson"]["等值集合"])
                for tlist in nodejson[node]["outjson"]["等值集合"]:
                    # sigt1 = 1 if len(set(["{角@BPM}", "{角@NQP}"]).intersection(set(tlist))) == 2 else 0
                    # sigt2 = 1 if len(set(["{角@MBP}", "{角@NPQ}"]).intersection(set(tlist))) == 2 else 0
                    # sigt3 = 1 if len(set(["{角@BMP}", "{角@PNQ}"]).intersection(set(tlist))) == 2 else 0
                    sigt4 = 1 if len(set(["{角@CAE}", "{角@DCF}"]).intersection(set(tlist))) == 2 else 0
                    # sigt4 = 1 if len(set(["{角@ABD}", "{角@CED}"]).intersection(set(tlist))) == 2 else 0
                    # sigt5 = 1 if len(set(["{线段@BP}", "{线段@PQ}"]).intersection(set(tlist))) == 2 else 0
                    # sigt6 = 1 if len(set(["{线段@BM}", "{线段@NP}"]).intersection(set(tlist))) == 2 else 0
                    sigta = sum([sigt4])
                    # sigta = sum([sigt1, sigt2, sigt3, sigt4, sigt5, sigt6])
                    if sigta > 0:
                        print(node, nodejson[node])
                        break
        print(556)
        for node in G1.nodes():
            if "等值集合" in nodejson[node]["outjson"]:
                for tlist in nodejson[node]["outjson"]["等值集合"]:
                    sigt4 = 1 if len(set(["{角@CAD}", "{角@DCF}"]).intersection(set(tlist))) == 2 else 0
                    sigta = sum([sigt4])
                    if sigta > 0:
                        print(node, nodejson[node])
                        break
        print(557)
        for node in G1.nodes():
            if "等值集合" in nodejson[node]["outjson"]:
                for tlist in nodejson[node]["outjson"]["等值集合"]:
                    sigt4 = 1 if len(set(["{角@CAE}", "{角@ECF}"]).intersection(set(tlist))) == 2 else 0
                    sigta = sum([sigt4])
                    if sigta > 0:
                        print(node, nodejson[node])
                        break
        print(559)
        for node in G1.nodes():
            if "等值集合" in nodejson[node]["outjson"]:
                for tlist in nodejson[node]["outjson"]["等值集合"]:
                    sigt4 = 1 if len(set(["{角@ABE}", "{角@AEB}"]).intersection(set(tlist))) == 2 else 0
                    sigta = sum([sigt4])
                    if sigta > 0:
                        print(node, nodejson[node])
                        break
        # print(556)
        # for node in G1.nodes():
        #     if nodejson[node]["points"][0].startswith("@@全等三角形充分"):
        #         for tlist in nodejson[node]["outjson"]["全等三角形集合"]:
        #             if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
        #                 print(node, nodejson[node])
        # print(557)
        # for node in G1.nodes():
        #     if "@@全等三角形必要条件" in nodejson[node]["points"]:
        #         for tlist in nodejson[node]["condjson"]["全等三角形集合"]:
        #             if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
        #                 print(node, nodejson[node])
        # print(558)
        # for node in G2.nodes():
        #     if "等值集合" in nodejson[node]["outjson"]:
        #         # print(node, nodejson[node]["outjson"]["等值集合"])
        #         for tlist in nodejson[node]["outjson"]["等值集合"]:
        #             # sigt1 = 1 if len(set(["{角@BPM}", "{角@NQP}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt2 = 1 if len(set(["{角@MBP}", "{角@NPQ}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt3 = 1 if len(set(["{角@BMP}", "{角@PNQ}"]).intersection(set(tlist))) == 2 else 0
        #             sigt4 = 1 if len(set(["{线段@MP}", "{线段@NQ}"]).intersection(set(tlist))) == 2 else 0
        #             sigt5 = 1 if len(set(["{线段@BP}", "{线段@PQ}"]).intersection(set(tlist))) == 2 else 0
        #             sigt6 = 1 if len(set(["{线段@BM}", "{线段@NP}"]).intersection(set(tlist))) == 2 else 0
        #             sigta = sum([sigt4, sigt5, sigt6])
        #             # sigta = sum([sigt1, sigt2, sigt3, sigt4, sigt5, sigt6])
        #             if sigta > 0:
        #                 print(node, nodejson[node])
        #                 break
        # print(559)
        # for node in G2.nodes():
        #     if nodejson[node]["points"][0].startswith("@@全等三角形充分"):
        #         for tlist in nodejson[node]["outjson"]["全等三角形集合"]:
        #             if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
        #                 print(node, nodejson[node])
        # print(560)
        # for node in G2.nodes():
        #     if "@@全等三角形必要条件" in nodejson[node]["points"]:
        #         for tlist in nodejson[node]["condjson"]["全等三角形集合"]:
        #             if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
        #                 print(node, nodejson[node])
        # print(561)
        # for node in G3.nodes():
        #     if "等值集合" in nodejson[node]["outjson"]:
        #         # print(node, nodejson[node]["outjson"]["等值集合"])
        #         for tlist in nodejson[node]["outjson"]["等值集合"]:
        #             # sigt1 = 1 if len(set(["{角@BPM}", "{角@NQP}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt2 = 1 if len(set(["{角@MBP}", "{角@NPQ}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt3 = 1 if len(set(["{角@BMP}", "{角@PNQ}"]).intersection(set(tlist))) == 2 else 0
        #             sigt4 = 1 if len(set(["{线段@MP}", "{线段@NQ}"]).intersection(set(tlist))) == 2 else 0
        #             sigt5 = 1 if len(set(["{线段@BP}", "{线段@PQ}"]).intersection(set(tlist))) == 2 else 0
        #             sigt6 = 1 if len(set(["{线段@BM}", "{线段@NP}"]).intersection(set(tlist))) == 2 else 0
        #             sigta = sum([sigt4, sigt5, sigt6])
        #             # sigta = sum([sigt1, sigt2, sigt3, sigt4, sigt5, sigt6])
        #             if sigta > 0:
        #                 print(node, nodejson[node])
        #                 break
        # print(562)
        # for node in G3.nodes():
        #     if nodejson[node]["points"][0].startswith("@@全等三角形充分"):
        #         for tlist in nodejson[node]["outjson"]["全等三角形集合"]:
        #             if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
        #                 print(node, nodejson[node])
        # print(563)
        # for node in G3.nodes():
        #     if "@@全等三角形必要条件" in nodejson[node]["points"]:
        #         for tlist in nodejson[node]["condjson"]["全等三角形集合"]:
        #             if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
        #                 print(node, nodejson[node])
        # raise 456
        # # 删除非答案的末节点
        # nodename = "求证"
        # parentnodes = self.checknode_parent(G1, nodename, steps=2)
        # print("{} 的父节点有 {} {}".format(nodename, len(parentnodes), parentnodes))
        # nodename = "已知"
        # childnodes = self.checknode_child(G1, nodename, steps=1)
        # print("{} 的子节点有 {} {}".format(nodename, len(childnodes), childnodes))
        return None, None

    def a_commonset_b(self, a, b):
        commenset = {}
        for obja in a:
            for objb in b:
                if objb == obja:
                    if obja in self.listset_obj:
                        commenset[obja] = []
                        for onea in a[obja]:
                            for oneb in b[objb]:
                                comelem = set(onea).issuperset(set(oneb))
                                if comelem:
                                    commenset[obja].append(list(set(oneb)))
                        if len(commenset[obja]) == 0:
                            del commenset[obja]
                    elif obja in self.set_obj:
                        comelem = set(a[obja]).issuperset(set(b[objb]))
                        if comelem:
                            commenset[obja] = list(set(b[objb]))
        if len(commenset) == 0:
            return False
        else:
            return commenset

    def genesuperset(self, listsobj):
        superset = {}
        for nodejso in listsobj:
            for obj in nodejso:
                if obj not in superset:
                    superset[obj] = []
                if obj in self.listset_obj:
                    for onelist in nodejso[obj]:
                        superset[obj].append(onelist)
                elif obj in self.set_obj:
                    superset[obj] += nodejso[obj]
                    superset[obj] = list(set(superset[obj]))
                else:
                    raise Exception("没有考虑到的集合。")
        # superset = self.listlist_deliverall(superset)
        return superset

    def a_supset_b(self, a, b):
        "判断a是b的超集"
        for objb in b:
            if objb not in a:
                return False
            if objb in self.listset_obj:
                for oneb in b[objb]:
                    setsig = 0
                    for onea in a[objb]:
                        if set(onea).issuperset(set(oneb)):
                            setsig = 1
                            break
                    if setsig == 0:
                        return False
            elif objb in self.set_obj:
                if not set(a[objb]).issuperset(set(b[objb])):
                    return False
            else:
                raise Exception("unknow error")
        # 没有发现异常，最后输出相同
        return True

    def nodejson2reportstr(self, innodejson):
        backlist = []
        for item in innodejson:
            if item in self.ignoreerror:
                continue
            if item in self.listset_obj:
                if innodejson[item] == []:
                    continue
                tstrlist = []
                for oneset in innodejson[item]:
                    setstr = ",".join([elem.strip("{}").replace("@", "") for elem in oneset])
                    tstrlist.append(" ".join([setstr, "是", item.replace("集合", "")]))
                backlist.append(";".join(tstrlist))
            elif item in self.set_obj:
                if innodejson[item] == []:
                    continue
                tstrlist = []
                for elem in innodejson[item]:
                    tstrlist.append(" ".join([elem.strip("{}").replace("@", ""), "是", item.replace("集合", "")]))
                backlist.append(",".join(tstrlist))
        return ";".join(backlist)

    def gene_uptree_from(self, nodejson, nodename):
        G = nx.DiGraph()
        knownodes = {nodename}
        waitenodes = list(nodejson.keys())
        waitenodes.remove(nodename)
        edgepairs = []
        oldlenth = -1
        # 生成全树
        while True:
            newlenth = len(knownodes)
            if oldlenth == newlenth:
                break
            oldlenth = newlenth
            # 提取公共元素
            for knownode in copy.deepcopy(knownodes):
                tcondilist = {}
                # 遍历每一层的节点输出端，输出构成待处理节点输入充分条件的，该节点加入连接信息，移除待处理节点。
                for waitenode in copy.deepcopy(waitenodes):
                    if knownode != waitenode:
                        common_elem = self.a_commonset_b(nodejson[waitenode]["outjson"], nodejson[knownode]["condjson"])
                        if common_elem:
                            tcondilist[waitenode] = common_elem
                # 生成超集
                supersets = self.genesuperset(tcondilist.values())
                if self.a_supset_b(supersets, nodejson[knownode]["condjson"]):
                    knownodes |= set(tcondilist.keys())
                    for condnode in tcondilist:
                        if "_".join([condnode, knownode]) not in edgepairs:
                            G.add_edge(condnode, knownode, weight=1)
                            edgepairs.append("_".join([condnode, knownode]))
        return G

    def gene_downtree_from(self, nodejson, nodename):
        G = nx.DiGraph()
        knownodes = {nodename}
        waitenodes = list(nodejson.keys())
        waitenodes.remove(nodename)
        # waitenodes.remove("求证")
        edgepairs = []
        oldlenth = -1
        # 生成全树
        while True:
            newlenth = len(knownodes)
            if oldlenth == newlenth:
                break
            oldlenth = newlenth
            # 遍历每一层的节点输出端，输出构成待处理节点输入充分条件的，该节点加入连接信息，移除待处理节点。
            for waitenode in copy.deepcopy(waitenodes):
                tcondilist = {}
                # 提取公共元素
                for knownode in knownodes:
                    if knownode != waitenode:
                        # a的部分是b的部分，不做交集检查
                        common_elem = self.a_commonset_b(nodejson[knownode]["outjson"], nodejson[waitenode]["condjson"])
                        if common_elem:
                            tcondilist[knownode] = common_elem
                # 生成超集
                supersets = self.genesuperset(tcondilist.values())
                if self.a_supset_b(supersets, nodejson[waitenode]["condjson"]):
                    knownodes.add(waitenode)
                    for condnode in tcondilist:
                        if "_".join([condnode, waitenode]) not in edgepairs:
                            G.add_edge(condnode, waitenode, weight=1)
                            edgepairs.append("_".join([condnode, waitenode]))
        return G

    def gene_downtree_semi_from(self, nodejson, nodename):
        " 默认节点自动连接，考点 答题描述连接，从而生成整棵树。"
        # 0. 默认节点
        reportjson = []
        defaultpointset = set(self.defaultpoint)
        defaultnode = []
        for node in nodejson:
            if defaultpointset.issuperset(set(nodejson[node]["points"])):
                defaultnode.append(node)
        nodejson_dyn = copy.deepcopy(nodejson)
        for node in nodejson_dyn:
            if node not in defaultnode:
                nodejson_dyn[node]["outjson"] = {}
        G = nx.DiGraph()
        knownodes = {nodename}
        waitenodes = list(nodejson.keys())
        waitenodes.remove(nodename)
        edgepairs = []
        print(self.answer_json)
        # 1. 循环查找 描述所以的语句
        for idl in range(len(self.answer_json)):
            if "condjson" in self.answer_json[idl]:
                continue
            oldlenth2 = -1
            while True:
                newlenth2 = len(knownodes)
                if oldlenth2 == newlenth2:
                    break
                oldlenth2 = newlenth2
                # print("loop 2")
                for waitenode in copy.deepcopy(defaultnode):
                    tcondilist = {}
                    # 提取公共元素
                    for knownode in knownodes:
                        if knownode != waitenode:
                            # a的部分是b的部分，不做交集检查
                            common_elem = self.a_commonset_b(nodejson_dyn[knownode]["outjson"],
                                                             nodejson_dyn[waitenode]["condjson"])
                            if common_elem:
                                tcondilist[knownode] = common_elem
                    # 生成超集
                    supersets = self.genesuperset(tcondilist.values())
                    if self.a_supset_b(supersets, nodejson_dyn[waitenode]["condjson"]):
                        knownodes.add(waitenode)
                        for condnode in tcondilist:
                            if "_".join([condnode, waitenode]) not in edgepairs:
                                G.add_edge(condnode, waitenode, weight=1)
                                edgepairs.append("_".join([condnode, waitenode]))
            # 3. 再 循环1内循环3，只找answer json 描述过的，输入或输出 在 know set为已知输出内容，。 输入连接循环
            # answer json 输出 在 所有节点 的输出，匹配加入know set 同时该条写入报告 ，直到没有新knowset节点。
            # 可能作为已知条件的 已知节点的输出
            answer_condi_main_set = {}
            for idn, answer_item in enumerate(self.answer_json):
                if idn >= idl:
                    break
                if "condjson" in answer_item:
                    answer_content = answer_item["condjson"]
                elif "outjson" in answer_item:
                    answer_content = answer_item["outjson"]
                else:
                    raise Exception("答案格式不是预期值。")
                tcondilist = {}
                for knownode in copy.deepcopy(knownodes):
                    out_cmmon = self.a_commonset_b(nodejson[knownode]["outjson"], answer_content)
                    if out_cmmon:
                        tcondilist[knownode] = out_cmmon
                out_supersets = self.genesuperset(tcondilist.values())
                if self.a_supset_b(out_supersets, answer_content):
                    answer_condi_main_set[idn] = list(tcondilist.keys())
            answer_condi_main_keys = list(answer_condi_main_set.keys())
            # 考点节点 非考点节点 所有可作为条件的相关节点。节点的输入 先验条件已经 knownodes 中过滤。
            answer_condi_client_set = set()
            for node in nodejson:
                if self.a_supset_b(nodejson[node]["outjson"], self.answer_json[idl]["outjson"]):
                    answer_condi_client_set.add(node)
            # 合并（书写条件 节点的交集），是 书写答案 的超集
            for outnode in answer_condi_client_set:
                tcondilist = {}
                for idn in range(idl):
                    if idn not in answer_condi_main_keys:
                        continue
                    out_cmmon = self.a_commonset_b(list(self.answer_json[idn].values())[0],
                                                   nodejson[outnode]["condjson"])
                    if out_cmmon:
                        tcondilist[idn] = out_cmmon
                out_supersets = self.genesuperset(tcondilist.values())
                if self.a_supset_b(out_supersets, nodejson[outnode]["condjson"]):
                    # 连边
                    knownodes.add(outnode)
                    tstb = self.nodejson2reportstr(nodejson[outnode]["condjson"])
                    tsts = self.nodejson2reportstr(self.answer_json[idl]["outjson"])
                    tstt = "因为:" + tstb + ". 所以:" + tsts
                    tpoint = nodejson[outnode]["points"]
                    for idn in tcondilist:
                        if idn not in answer_condi_main_set:
                            continue
                        for condnode in answer_condi_main_set[idn]:
                            if "_".join([condnode, outnode]) not in edgepairs:
                                G.add_edge(condnode, outnode, weight=1)
                                edgepairs.append("_".join([condnode, outnode]))
                    if not defaultpointset.issuperset(set(tpoint)) and tstb != "" and tsts != "":
                        # 根据新结论描述，写入条件节点
                        nodejson_dyn[outnode]["outjson"] = self.genesuperset(
                            [nodejson_dyn[outnode]["outjson"], self.answer_json[idl]["outjson"]])
                        tmjson = {"content": tstt, "point": ",".join(tpoint), "istrue": "连通描述正确"}
                        reportjson.append(tmjson)
        # 4. 答案描述结束后，结果判断
        outnode = "求证"
        tcondilist = {}
        for knownode in knownodes:
            out_cmmon = self.a_commonset_b(nodejson[knownode]["outjson"], nodejson["求证"]["condjson"])
            if out_cmmon:
                tcondilist[knownode] = out_cmmon
        out_supersets = self.genesuperset(tcondilist.values())
        if self.a_supset_b(out_supersets, nodejson[outnode]["condjson"]):
            knownodes.add(outnode)
            for condnode in tcondilist:
                if "_".join([condnode, outnode]) not in edgepairs:
                    G.add_edge(condnode, outnode, weight=1)
                    edgepairs.append("_".join([condnode, outnode]))
        print(reportjson)
        return G, reportjson

    def gene_fulltree_from(self, nodejson):
        G = nx.DiGraph()
        # 生成全树
        for targnode in nodejson:
            tcondilist = {}
            # 提取公共元素
            for condnode in nodejson:
                if condnode != targnode:
                    common_elem = self.a_commonset_b(nodejson[condnode]["outjson"], nodejson[targnode]["condjson"])
                    if common_elem:
                        tcondilist[condnode] = common_elem
            # 生成超集
            supersets = self.genesuperset(tcondilist.values())
            if self.a_supset_b(supersets, nodejson[targnode]["condjson"]):
                for condnode in tcondilist:
                    G.add_edge(condnode, targnode, weight=1)
        return G

    def checknode_parent(self, G, nodename, steps=1):
        " 根据需求超父节点 "
        counter = 0
        outnodes = [nodename]
        oldlenth = -1
        while True:
            counter += 1
            newlenth = len(outnodes)
            if steps < counter or oldlenth == newlenth:
                break
            oldlenth = newlenth
            tnodes = []
            for nodepair in G.edges():
                if nodepair[1] in outnodes and nodepair[0] not in outnodes:
                    tnodes.append(nodepair[0])
            outnodes += tnodes
        return outnodes

    def checknode_child(self, G, nodename, steps=1):
        " 根据需求超父节点 "
        counter = 0
        outnodes = [nodename]
        oldlenth = -1
        while True:
            counter += 1
            newlenth = len(outnodes)
            if steps < counter or oldlenth == newlenth:
                break
            oldlenth = newlenth
            tnodes = []
            for nodepair in G.edges():
                if nodepair[0] in outnodes and nodepair[1] not in outnodes:
                    tnodes.append(nodepair[1])
            outnodes += tnodes
        return outnodes

    def deltree_layer(self, G, delcounter, stopcounter):
        " 根据需求删除节点 "
        oldlenth = -1
        newlenth = len(G.edges())
        startt = time.time()
        while True:
            if oldlenth == newlenth or stopcounter == delcounter:
                break
            delcounter += 1
            oldlenth = newlenth
            # print("del info", delcounter, len(G.nodes()), oldlenth)
            for node in copy.deepcopy(G.nodes()):
                midsig = 0
                for nodepair in copy.deepcopy(G.edges()):
                    if node == nodepair[0]:
                        midsig = 1
                if midsig == 0 and node != "求证":
                    for eddg in copy.deepcopy(G.edges()):
                        if node == eddg[1]:
                            G.remove_edge(*eddg)
                    G.remove_node(node)
            newlenth = len(G.edges())
        logger1.info("精简树 use time: {}mins".format((time.time() - startt) / 60))
        return G

    def deriv_basicelement(self, analist):
        # 衍生一级元素
        # 1. 生成汉语法列表
        # print("deriv_basicelement")
        # print(analist)
        newlist = []
        for i1 in analist:
            if isinstance(i1, list):
                newlist.append(i1)
            else:
                newlist += [latex_fenci(i2.strip()) for i2 in re.split(',|，|;|\n|\\\qquad|\\\quad|\t', i1)]
        # 2. 生成语法的内存属性实体、生成语法的关系三元组
        write_json = []
        newoutlist = []
        for i1 in newlist:
            if len(i1) == 0:
                continue
            if not isinstance(i1[0], list):
                outlist, outjson = self.language.latex_default_property(i1)
                write_json += outjson
                newoutlist.append(outlist)
            else:
                newoutlist.append(i1)
        # print(newoutlist)
        # print(write_json)
        return newoutlist, write_json

    def deriv_relationelement(self, analist):
        # 提取所有 已知或求证 的关系
        # 输入: [[['已知'], ['v']], ['{线段@PQ}', '=', '{线段@BP}'], ['{线段@MN}', '\\parallel', '{线段@BC}'], ['{角@BPQ}', '=', '9', '0', '^', '{ \\circ }'], [['求证'], ['v']], ['{线段@BP}', '=', '{线段@PQ}']]
        # 输出:
        purpose_json = []
        # 0. 文本latex标记
        length = len(analist)
        # print("deriv_relationelement")
        # print(analist)
        typesig = "因为"
        for i1 in range(length):
            if isinstance(analist[i1][0], list):
                if analist[i1][0][0] in ["求证"]:
                    # typesig = "所以"
                    typesig = "求证"
            # 每一组只能容纳一个类型
            subjectlist = []
            setstr = ""
            isstr = "是"
            tmp_json = []
            for i2 in range(len(analist[i1])):
                sigmatch = 0
                # print(analist[i1][i2])
                if analist[i1][i2] in step_alist:
                    typesig = "因为"
                    sigmatch = 1
                    tmp_json = []
                elif analist[i1][i2] in step_blist:
                    typesig = "所以"
                    sigmatch = 1
                    tmp_json = []
                else:
                    # 默认为
                    pass
                if analist[i1][i2] in ["=", "等值", '等于']:
                    sigmatch = 1
                    if setstr == "等值" or setstr == "":
                        setstr = "等值"
                    else:
                        raise Exception("等值!={}".format(setstr))
                if analist[i1][i2] in ['\\parallel', '平行']:
                    sigmatch = 1
                    if setstr == "平行" or setstr == "":
                        setstr = "平行"
                    else:
                        raise Exception("平行!={}".format(setstr))
                if analist[i1][i2] in ['\\perp', '垂直']:
                    sigmatch = 1
                    if setstr == "垂直" or setstr == "":
                        setstr = "垂直"
                    else:
                        raise Exception("垂直!={}".format(setstr))
                if analist[i1][i2] in ['\\cong', '全等三角形']:
                    sigmatch = 1
                    if setstr == "全等三角形" or setstr == "":
                        setstr = "全等三角形"
                    else:
                        raise Exception("全等三角形!={}".format(setstr))
                if analist[i1][i2] in ["相似"]:
                    sigmatch = 1
                    if setstr == "相似" or setstr == "":
                        setstr = "相似"
                    else:
                        raise Exception("相似!={}".format(setstr))
                # # 貌似没用
                # if analist[i1][i2] in ["等腰三角形"]:
                #     sigmatch = 1
                #     if setstr == "等腰三角形" or setstr == "":
                #         setstr = "等腰三角形"
                #         print("等腰三角形")
                #     else:
                #         raise Exception("等腰三角形!={}".format(setstr))
                # if analist[i1][i2] in ["等边三角形"]:
                #     sigmatch = 1
                #     if setstr == "等边三角形" or setstr == "":
                #         setstr = "等边三角形"
                #     else:
                #         raise Exception("等边三角形!={}".format(setstr))
                if 1 != sigmatch:
                    tmp_json.append(analist[i1][i2])
                else:
                    if len(tmp_json) != 0:
                        subjectlist.append(tmp_json)
                    tmp_json = []
            if len(tmp_json) != 0 and not isinstance(analist[i1][0], list):
                subjectlist.append(tmp_json)
            subjectlist = [[" ".join(i2) for i2 in subjectlist], isstr, setstr]
            if setstr != "":
                purpose_json.append({typesig: subjectlist})
                # if setstr == "等边三角形":
        return purpose_json

    def get_allkeyproperty(self, analist):
        """text latex 句子间合并, 写入概念属性json，返回取出主题概念的列表"""
        # 1. 展成 同级 list
        # print("in get_allkeyproperty")
        # print(analist)
        analist = list(itertools.chain(*analist))
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [list(sentence.values())[0].strip() for sentence in analist]
        olenth = len(contlist)
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        outlatex = []
        # 2. 提取自带显示属性
        for idn in range(olenth):
            if keylist[idn] == "latex":
                latexlist, propertyjson = self.language.latex_extract_property(contlist[idn])
                outlatex += latexlist
                propertyjson = [{"因为": i1} for i1 in propertyjson]
                self.language.json2space(propertyjson, basic_space_ins, space_ins)
            else:
                tlist = []
                for word in pseg.cut(contlist[idn]):
                    if word.flag not in ["x", "d"]:
                        tlist.append([[word.word], [word.flag]])
                outlatex += tlist
        # 3. 提取默认 字面初级元素，升级属性或新元素为后续工作
        outlatex, propertyjson = self.deriv_basicelement(outlatex)
        propertyjson = [{"因为": i1} for i1 in propertyjson]
        self.language.json2space(propertyjson, basic_space_ins, space_ins)
        # 4. 语法提取 字面关系
        propertyjson = self.deriv_relationelement(outlatex)
        self.language.json2space(propertyjson, basic_space_ins, space_ins)
        # 清理
        space_ins._setobj = self.list_set_shrink_all(space_ins._setobj)
        # 表达式提取
        space_ins._setobj = self.prepare_clean_set(space_ins._setobj)
        # 写入初始化条件
        space_ins._initobj = copy.deepcopy(space_ins._setobj)
        return outlatex

    def get_answerproperty(self, analist):
        """text latex 句子间合并, 根据因为所以 写入不同空间概念属性json，返回取出主题概念的列表"""
        # 1. 展成 同级 list
        # print("in get_answerproperty")
        # print(analist)
        analist = list(itertools.chain(*analist))
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [list(sentence.values())[0].strip() for sentence in analist]
        olenth = len(contlist)
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        outlatex = []
        # 2. 提取自带显示属性
        for idn in range(olenth):
            if keylist[idn] == "latex":
                # print("latex")
                # print(contlist[idn])
                latexlist, propertyjson = self.language.latex_extract_property(contlist[idn])
                outlatex += latexlist
                propertyjson = [{"因为": i1} for i1 in propertyjson]
                self.language2answer(basic_space_ins._setobj, addc=propertyjson)
            else:
                tlist = []
                for word in pseg.cut(contlist[idn]):
                    if word.flag not in ["x", "d"]:
                        tlist.append([[word.word], [word.flag]])
                outlatex += tlist
        # 3. 提取默认 字面初级元素，升级属性或新元素为后续工作
        outlatex, propertyjson = self.deriv_basicelement(outlatex)
        propertyjson = [{"因为": i1} for i1 in propertyjson]
        self.language2answer(basic_space_ins._setobj, addc=propertyjson)
        # 4. 语法提取 字面关系
        # 加入过程是逆序
        outlatex.reverse()
        propertyjson = self.deriv_relationelement(outlatex)
        self.language2answer(basic_space_ins._setobj, addc=propertyjson)
        # 清理
        # self.answer_shrink_all()
        # because_space_ins._setobj = self.list_set_shrink_all(because_space_ins._setobj)
        # 表达式提取
        # print(self.answer_json)
        # raise 123123
        # print("16554")
        # print(self.answer_json)
        self.answer_json = self.answer_clean_set()
        # print(self.answer_json)
        return outlatex

    def answer_clean_set(self):
        # 1. 移动含有表达式的条目。
        expresskey = ["+", "-", "*", "\\frac", "\\times", "\\div"]
        legth = len(self.answer_json)
        for idn in range(legth):
            onesetkey = list(self.answer_json[idn].keys())[0]
            if "等值集合" in self.answer_json[idn][onesetkey]:
                lenth = len(self.answer_json[idn][onesetkey]["等值集合"])
                express_set = set()
                for ids in range(lenth - 1, -1, -1):
                    breaksig = 0
                    for elem in self.answer_json[idn][onesetkey]["等值集合"][ids]:
                        for key in expresskey:
                            if key in elem:
                                express_set.add(" = ".join(list(self.answer_json[idn][onesetkey]["等值集合"][ids])))
                                self.answer_json[idn][onesetkey]["等值集合"].pop(ids)
                                breaksig = 1
                                break
                        if breaksig == 1:
                            break
                if "表达式集合" not in self.answer_json[idn][onesetkey]:
                    self.answer_json[idn][onesetkey]["表达式集合"] = set()
                self.answer_json[idn][onesetkey]["表达式集合"] |= express_set
                if self.answer_json[idn][onesetkey]["表达式集合"] == set():
                    del self.answer_json[idn][onesetkey]["表达式集合"]
                if self.answer_json[idn][onesetkey]["等值集合"] == []:
                    del self.answer_json[idn][onesetkey]["等值集合"]
        # 2. 空间定义
        return self.answer_json

    def answer_shrink_all(self):
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        oldsetobj = self.answer_json
        for objset in basic_space_ins._setobj:
            if basic_space_ins._setobj[objset]["结构形式"] == "一级列表二级集合":
                oldsetobj[objset] = list_set_shrink(oldsetobj[objset])
        newsetobj = copy.deepcopy(oldsetobj)
        return newsetobj

    def simplify_node(self, orinode, ansnode, ignoreset):
        # 1 精简题目json
        simple_nodejson = {}
        for node in orinode:
            tconjson = {}
            for item in orinode[node]["condjson"]:
                if item not in ignoreset:
                    tconjson[item] = orinode[node]["condjson"][item]
            toujson = {}
            for item in orinode[node]["outjson"]:
                if item not in ignoreset:
                    toujson[item] = orinode[node]["outjson"][item]
            simple_nodejson[node] = {"condjson": tconjson, "points": orinode[node]["points"], "outjson": toujson}
        # 2 精简答案json
        anserlenth = len(ansnode)
        for idn in range(anserlenth - 1, -1, -1):
            tconjson = {}
            item = ansnode[idn]
            for key in item:
                # 每个就一条
                tconjson[key] = {}
                for filkey in item[key]:
                    if filkey not in ignoreset:
                        tconjson[key][filkey] = item[key][filkey]
                if tconjson[key] != {}:
                    ansnode[idn] = tconjson
                else:
                    ansnode.pop(idn)
        simple_nodejson["已知"] = orinode["已知"]
        return simple_nodejson, ansnode

    def sentence2normal(self, analist):
        """text latex 句子间合并 按句意合并, 结果全部写入内存。"""
        # 1. 展成 同级 list
        # print("sentence2normal")
        analist = list(itertools.chain(*analist))
        # 2. 去掉空的
        # print(analist)
        analist = [{list(sentence.keys())[0]: list(sentence.values())[0].strip(",，。 \t")} for
                   sentence in analist if list(sentence.values())[0].strip(",，。 \t") != ""]
        # 3. 合并 临近相同的
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [list(sentence.values())[0].strip() for sentence in analist]
        # print(keylist)
        # print(contlist)
        olenth = len(contlist)
        if olenth < 2:
            print("合并 临近相同的 olenth < 2")
            analist = [[{keylist[i1]: contlist[i1]} for i1 in range(olenth)]]
            anastr = self.get_allkeyproperty(analist)
            return anastr
        for i1 in range(olenth - 1, 0, -1):
            if keylist[i1] == keylist[i1 - 1]:
                analist[i1 - 1] = {keylist[i1 - 1]: contlist[i1 - 1] + " ; " + analist[i1][keylist[i1]]}
                del analist[i1]
        # 4. latex text 转化
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        sortkey = list(basic_space_ins._proper_keys) + [trio[0] for trio in basic_space_ins._relation_trip.values()] \
                  + [trio[2] for trio in basic_space_ins._relation_trip.values()]
        sortkey = set(sortkey)
        sortkey = [[onk, len(onk)] for onk in sortkey]
        sortkey = [onk[0] for onk in sorted(sortkey, key=lambda x: -x[1])]
        # 前一个为 text, 已关键字结尾， 且后一个为latex, 以字母开始。则拆分合并。
        ins_json = []
        quire_json = []
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [sentence[list(sentence.keys())[0]].strip() for sentence in analist]
        olenth = len(analist)
        if olenth < 2:
            print("latex text 转化 olenth < 2")
            analist = [[{keylist[i1]: contlist[i1]} for i1 in range(olenth)]]
            anastr = self.get_allkeyproperty(analist)
            return anastr
        quiresig = 0
        for i1 in range(olenth - 1, 0, -1):
            # 目前 仅支持两种模式  1. 如： 正方形 ABCD 2. 如：A B C D 在一条直线上
            # print(i1, contlist[i1], contlist, keylist)
            # print(len(contlist), len(keylist))
            if contlist[i1].endswith("求证") and quiresig == 0:
                quiresig = 1
                # contlist[i1] = contlist[i1].rstrip("求证")
                quire_json = copy.deepcopy(ins_json)
                ins_json = []
                for idn, item in enumerate(quire_json):
                    tnkey = list(quire_json[idn].keys())[0]
                    tnvalue = list(quire_json[idn].values())[0]
                    quire_json[idn]["求证"] = tnvalue
                    del quire_json[idn][tnkey]
            if "" == contlist[i1]:
                del keylist[i1]
                del contlist[i1]
                continue
            if keylist[i1] == "latex" and keylist[i1 - 1] == "text":
                for jsonkey in sortkey:
                    mt = re.sub(u"{}$".format(jsonkey), "", contlist[i1 - 1])
                    mthead = re.match(u"[是|的]{}$".format(jsonkey), contlist[i1 - 1])
                    if not mthead and mt != contlist[i1 - 1]:
                        # 前一个 以属性名结尾
                        se = re.match(r"^(\w|\s)+", contlist[i1])
                        if se is not None:
                            # 后一个 以字母空格开头的 单元字符串 去空
                            # 当前contlist 为一个集合
                            tinlist = self.language.latex_extract_word(contlist[i1])
                            tstrli = latex_fenci(tinlist[0])
                            siglist = [len(i2) for i2 in tstrli]
                            siglenth = len(siglist)
                            posind = -1
                            for i2 in range(siglenth):
                                if siglist[i2] == 1:
                                    posind = i2
                                else:
                                    break
                            if posind == -1:
                                raise Exception("对应实体描述不存在")
                            else:
                                tstrstr = [" ".join(i2) for i2 in tstrli]
                                tconcept_list = []
                                for i2 in range(posind, siglenth):
                                    tnewstr = "{ 点@" + tstrstr[i2] + " }"
                                    tnewstr = self.language.name_normal(tnewstr)
                                    ins_json.append({"因为": [tnewstr, "是", "点"]})
                                    tconcept_list.append(tnewstr)
                                tnewstr = "{ " + jsonkey + "@" + " ".join(tstrstr[0:posind + 1]) + " }"
                                tnewstr = self.language.name_normal(tnewstr)
                                ins_json.append({"因为": [tnewstr, "是", jsonkey]})
                                # 改写拼接
                                length_tinlist = len(tinlist)
                                bstr = ""
                                if length_tinlist > 1:
                                    bstr = " , ".join(tinlist[1:])
                                if siglenth - 1 != posind:
                                    # 是否删除latex部分
                                    contlist[i1] = " , ".join(tstrstr[posind:]) + bstr
                                else:
                                    if bstr == "":
                                        del keylist[i1]
                                        del contlist[i1]
                                    else:
                                        contlist[i1] = bstr
                            # 是否删除文本部分
                            ttext = mt.strip(",，；;:：。 \t")
                            if ttext != "":
                                contlist[i1 - 1] = ttext
                            else:
                                del keylist[i1 - 1]
                                del contlist[i1 - 1]
                            break
            elif keylist[i1] == "text" and keylist[i1 - 1] == "latex":
                ttypelist = ["在一条直线上", "是直径", "是锐角", "是补角", "是圆", "是等腰三角形", "是直角三角形", "的弦切角"]
                matchsig = 0
                for jsonkey in ttypelist:
                    mt = re.sub(u"^{}".format(jsonkey), "", contlist[i1])
                    if mt != contlist[i1]:
                        # print(996)
                        # 是否删除文本部分
                        ttext = mt.strip(",，；;:：。 \t")
                        if ttext != "":
                            contlist[i1] = ttext
                        else:
                            del keylist[i1]
                            del contlist[i1]
                        # 前面的 为一个集合
                        tinlist = self.language.latex_extract_word(contlist[i1 - 1])
                        tstrli = [latex_fenci(i2) for i2 in tinlist]
                        siglist = [len(i2) for i2 in tstrli]
                        siglenth = len(siglist)
                        if jsonkey in ["在一条直线上"]:
                            posind = -1
                            for i2 in range(siglenth - 1, -1, -1):
                                if siglist[i2] == 1:
                                    posind = i2
                                else:
                                    break
                        elif jsonkey in ["是直径"]:
                            posind = -1
                            for i2 in range(siglenth - 1, -1, -1):
                                if siglist[i2] == 3:
                                    posind = i2
                                else:
                                    break
                        elif jsonkey in ["是补角"]:
                            posind = -1
                            countertt = 0
                            for i2 in range(siglenth - 1, -1, -1):
                                if siglist[i2] > 1:
                                    countertt += siglist[i2]
                                if countertt > 3:
                                    posind = i2
                                    break
                        elif jsonkey in ["是锐角", "是等腰三角形", "是直角三角形"]:
                            posind = 0 if siglist[0] > 0 else -1
                        elif jsonkey in ["的弦切角"]:
                            # 模式： OC, \\angle {DCF} $ 是 $\\overset{\\frown} {DC}$的弦切角。
                            tinlist = self.language.latex_extract_word(contlist[i1 - 3])
                            tstrli = [latex_fenci(i2) for i2 in tinlist]
                            siglist = [len(i2) for i2 in tstrli]
                            siglenth = len(siglist)
                            posind = -1
                            for i2 in range(siglenth - 1, -1, -1):
                                if siglist[i2] == 2:
                                    posind = i2
                                    break
                                else:
                                    raise Exception("弦切角格式错误。")
                        if posind == -1:
                            raise Exception("在一条直线上 或 是锐角 前面不应为空")
                        else:
                            tstrstr = [" ".join(i2) for i2 in tstrli]
                            tconcept_list = []
                            if jsonkey in ["在一条直线上"]:
                                for i2 in range(posind, siglenth):
                                    tnewstr = "{ 点@" + tstrstr[i2] + " }"
                                    tnewstr = self.language.name_normal(tnewstr)
                                    ins_json.append({"因为": [tnewstr, "是", "点"]})
                                    tconcept_list.append(tnewstr)
                                ins_json.append({"因为": [tconcept_list, "是", "直线"]})
                                if 0 != posind:
                                    # 是否删除latex部分
                                    contlist[i1 - 1] = " , ".join(tstrstr[0:posind])
                                else:
                                    del keylist[i1 - 1]
                                    del contlist[i1 - 1]
                                matchsig = 1
                            elif jsonkey in ["是直径"]:
                                # print(posind, siglenth, tstrstr, tstrli)
                                for i2 in range(posind, siglenth):
                                    ttconcelist = []
                                    for i3 in tstrli[i2]:
                                        tnewstr = "{ 点@" + i3 + " }"
                                        tnewstr = self.language.name_normal(tnewstr)
                                        ins_json.append({"因为": [tnewstr, "是", "点"]})
                                        ttconcelist.append(tnewstr)
                                    ins_json.append({"因为": [ttconcelist, "是", "直线"]})
                                    tname = self.language.name_symmetric(tstrstr[i2]).replace(" ", "")
                                    tname = "{直径@" + tname + "}"
                                    ins_json.append({"因为": [tname, "是", "直径"]})
                                # print(ins_json[-1],ins_json[-2])
                                if 0 != posind:
                                    # 是否删除latex部分
                                    contlist[i1 - 1] = " , ".join(tstrstr[0:posind])
                                else:
                                    del keylist[i1 - 1]
                                    del contlist[i1 - 1]
                                matchsig = 1
                            elif jsonkey == "是锐角":
                                tstrli = [i2 for i2 in tstrli[-1] if i2 != "\\angle"]
                                typsig = "因为"
                                for i2 in tstrli:
                                    if i2 == "\\therefore":
                                        typsig = "所以"
                                    if i2.startswith("{") and i2.endswith("}"):
                                        tnewstr = "{ 角@" + i2.strip("{}") + " }"
                                        tnewstr = self.language.name_normal(tnewstr)
                                        ins_json.append({typsig: [tnewstr, "是", "锐角"]})
                                        matchsig = 1
                            elif jsonkey == "是补角":
                                tstrli = [tstrli[i2] for i2 in range(posind, siglenth)]
                                typsig = "因为"
                                tlist = set()
                                for i2 in tstrli:
                                    if "\\therefore" in i2:
                                        typsig = "所以"
                                        i2.remove("\\therefore")
                                    elif "\\because" in i2:
                                        typsig = "因为"
                                        i2.remove("\\because")
                                    # 必然含 \\angle 否则报错
                                    i2.remove("\\angle")
                                    if i2[0].startswith("{") and i2[0].endswith("}"):
                                        tnewstr = "{ 角@" + i2[0].strip("{}") + " }"
                                        tnewstr = self.language.name_normal(tnewstr)
                                        tlist.add(tnewstr)
                                ins_json.append({typsig: [list(tlist), "是", jsonkey.lstrip("是")]})
                                matchsig = 1
                            elif jsonkey in ["是等腰三角形", "是直角三角形"]:
                                tstrli = [i2 for i2 in tstrli[-1] if i2 != "\\triangle"]
                                typsig = "因为"
                                for i2 in tstrli:
                                    if i2 == "\\therefore":
                                        typsig = "所以"
                                    if i2.startswith("{") and i2.endswith("}"):
                                        tnewstr = i2.strip("{}")
                                        tnewstr = self.language.name_cyc_one(tnewstr).replace(" ", "")
                                        tnewstr = "{三角形@" + tnewstr + "}"
                                        ins_json.append({typsig: [tnewstr, "是", jsonkey.lstrip("是")]})
                                        matchsig = 1
                            elif jsonkey in ["的弦切角"]:
                                tinlist3 = self.language.latex_extract_word(contlist[i1 - 1])
                                tstrli3 = [latex_fenci(i2) for i2 in tinlist3]
                                if '\\overset{\\frown}' != tstrli3[0][0]:
                                    print(tstrli3)
                                    raise Exception("弦切角格式错误。")
                                tinlist2 = self.language.latex_extract_word(contlist[i1 - 2])
                                if '是' != tinlist2[0]:
                                    print(tinlist2)
                                    raise Exception("弦切角格式错误。")
                                if '\\angle' != tstrli[-1][0]:
                                    print(tstrli[-1])
                                    raise Exception("弦切角格式错误。")
                                contlist[i1 - 1] = ""
                                contlist[i1 - 2] = ""
                                contlist[i1 - 3] = ",".join(re.split(',|，|、|;|；|\n|\t', contlist[i1 - 3])[:-1])
                                tananpair = []
                                tnllist = tstrli3[0][1].strip("{ }").split()
                                tname = self.language.name_symmetric(" ".join(tnllist[1:])).replace(" ", "")
                                tnewstr = "{弧@" + tnllist[0] + tname + "}"
                                tananpair.append(tnewstr)
                                tname = self.language.name_symmetric(tstrli[-1][1].strip("{ }")).replace(" ", "")
                                tnewstr = "{角@" + tname + "}"
                                tananpair.append(tnewstr)
                                typsig = "因为"
                                if "\\therefore" in tstrli:
                                    typsig = "所以"
                                ins_json.append({typsig: [tananpair, "是", jsonkey.lstrip("的")]})
                                # print(ins_json[-1])
                                matchsig = 1
                            else:
                                print(jsonkey)
                                raise Exception("在一条直线上")
                    if matchsig == 1:
                        break
                        # print(ins_json)
        # 5. 写入句间的实例
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        self.language.json2space(ins_json, basic_space_ins, space_ins)
        # print(ins_json)
        # print(space_ins._setobj)
        # print(space_ins._stopobj)
        self.language.json2space(quire_json, basic_space_ins, space_ins)
        # print(quire_json)
        # print(space_ins._setobj)
        # print(space_ins._stopobj)
        # 6. 提取所有 抽象类。对应实例，改变字符。属性
        olenth = len(contlist)
        analist = [[{keylist[i1]: contlist[i1]}] for i1 in range(olenth)]
        # print(analist)
        anastr = self.get_allkeyproperty(analist)
        # print(space_ins._setobj)
        # print(space_ins._stopobj)
        # print("sentence2normal")
        return anastr

    def answer2normal(self, analist):
        """text latex 句子间合并 按句意合并, 结果全部写入内存。"""
        # 1. 展成 同级 list
        # print("answer2normal")
        self.answer_json = []
        analist = list(itertools.chain(*analist))
        # 2. 去掉空的
        analist = [{list(sentence.keys())[0]: list(sentence.values())[0].strip(",，。;； \t")} for
                   sentence in analist if list(sentence.values())[0].strip(",，。;； \t") != ""]
        # print(analist)
        # 3. 合并 临近相同的
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [list(sentence.values())[0].strip() for sentence in analist]
        olenth = len(contlist)
        if olenth < 2:
            print("合并 临近相同的 olenth < 2")
            analist = [[{keylist[i1]: contlist[i1]} for i1 in range(olenth)]]
            anastr = self.get_answerproperty(analist)
            return anastr
        for i1 in range(olenth - 1, 0, -1):
            if keylist[i1] == keylist[i1 - 1]:
                analist[i1 - 1] = {keylist[i1 - 1]: contlist[i1 - 1] + " ; " + analist[i1][keylist[i1]]}
                del analist[i1]
        # 4. latex text 转化
        # print("answer2normal........")
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # 前一个为 text, 已关键字结尾， 且后一个为latex, 以字母开始。则拆分合并。
        ins_json = []
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [sentence[list(sentence.keys())[0]].strip() for sentence in analist]
        # print(keylist)
        # print(contlist)
        olenth = len(analist)
        if olenth < 2:
            print("latex text 转化 olenth < 2")
            analist = [[{keylist[i1]: contlist[i1]} for i1 in range(olenth)]]
            anastr = self.get_answerproperty(analist)
            return anastr
        for i1 in range(olenth - 1, 0, -1):
            # 目前 仅支持两种模式  1. 如： 正方形 ABCD 2. 如：A B C D 在一条直线上
            if keylist[i1] == "latex" and keylist[i1 - 1] == "text":
                sortkey = ['等腰三角形', '等角三角形', '平行四边形', '全等三角形', '等边三角形', '正方形', '四边形', 'n边形',
                           '三角形', '切线', '半径', '矩形', '圆心', '直径', '圆', '边', '3', '弦', '']
                for jsonkey in sortkey:
                    mt = re.sub(u"{}$".format(jsonkey), "", contlist[i1 - 1])
                    if mt != contlist[i1 - 1]:
                        # 前一个 以属性名结尾
                        se = re.match(r"^(\w|\s)+", contlist[i1])
                        if se is not None:
                            # 后一个 以字母空格开头的 单元字符串 去空
                            # 当前contlist 为一个集合
                            tinlist = self.language.latex_extract_word(contlist[i1])
                            tstrli = latex_fenci(tinlist[0])
                            siglist = [len(i2) for i2 in tstrli]
                            siglenth = len(siglist)
                            posind = -1
                            for i2 in range(siglenth):
                                if siglist[i2] == 1:
                                    posind = i2
                                else:
                                    break
                            if posind == -1:
                                raise Exception("对应实体描述不存在")
                            else:
                                tstrstr = [" ".join(i2) for i2 in tstrli]
                                tconcept_list = []
                                for i2 in range(posind, siglenth):
                                    tnewstr = "{ 点@" + tstrstr[i2] + " }"
                                    tnewstr = self.language.name_normal(tnewstr)
                                    ins_json.append({"因为": [tnewstr, "是", "点"]})
                                    tins_json = [{"因为": [tnewstr, "是", "点"]}]
                                    self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                    tconcept_list.append(tnewstr)
                                tnewstr = "{ " + jsonkey + "@" + " ".join(tstrstr[0:posind + 1]) + " }"
                                tnewstr = self.language.name_normal(tnewstr)
                                ins_json.append({"因为": [tnewstr, "是", jsonkey]})
                                tins_json = [{"因为": [tnewstr, "是", jsonkey]}]
                                self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                # 改写拼接
                                length_tinlist = len(tinlist)
                                bstr = ""
                                if length_tinlist > 1:
                                    bstr = " , ".join(tinlist[1:])
                                if siglenth - 1 != posind:
                                    # 是否删除latex部分
                                    contlist[i1] = " , ".join(tstrstr[posind:]) + bstr
                                else:
                                    if bstr == "":
                                        del keylist[i1]
                                        del contlist[i1]
                                    else:
                                        contlist[i1] = bstr
                            # 是否删除文本部分
                            ttext = mt.strip(",，；;:：。 \t")
                            if ttext != "":
                                contlist[i1 - 1] = ttext
                            else:
                                del keylist[i1 - 1]
                                del contlist[i1 - 1]
                            break
            elif keylist[i1] == "text" and keylist[i1 - 1] == "latex":
                ttypelist = ["在一条直线上", "是直径", "是锐角", "是补角", "是等腰三角形", "是直角三角形"]  # 目前仅支持一种模式: \\angle {xxx}
                matchsig = 0
                for jsonkey in ttypelist:
                    mt = re.sub(u"^{}".format(jsonkey), "", contlist[i1])
                    if mt != contlist[i1]:
                        # 是否删除文本部分
                        ttext = mt.strip(",，；;:：。 \t")
                        if ttext != "":
                            contlist[i1] = ttext
                        else:
                            del keylist[i1]
                            del contlist[i1]
                        # 前面的 为一个集合
                        tinlist = self.language.latex_extract_word(contlist[i1 - 1])
                        tstrli = [latex_fenci(i2) for i2 in tinlist]
                        siglist = [len(i2) for i2 in tstrli]
                        siglenth = len(siglist)
                        if jsonkey in ["在一条直线上"]:
                            posind = -1
                            for i2 in range(siglenth - 1, -1, -1):
                                if siglist[i2] == 1:
                                    posind = i2
                                else:
                                    break
                        elif jsonkey in ["是直径"]:
                            posind = -1
                            for i2 in range(siglenth - 1, -1, -1):
                                if siglist[i2] == 3:
                                    posind = i2
                                else:
                                    break
                        elif jsonkey in ["是补角"]:
                            posind = -1
                            countertt = 0
                            for i2 in range(siglenth - 1, -1, -1):
                                if siglist[i2] > 1:
                                    countertt += siglist[i2]
                                if countertt > 3:
                                    posind = i2
                                    break
                        elif jsonkey in ["是锐角", "是等腰三角形", "是直角三角形"]:
                            posind = 0 if siglist[0] > 0 else -1
                        if posind == -1:
                            raise Exception("在一条直线上 或 是锐角 前面不应为空")
                        else:
                            tstrstr = [" ".join(i2) for i2 in tstrli]
                            tconcept_list = []
                            if jsonkey in ["在一条直线上"]:
                                for i2 in range(posind, siglenth):
                                    tnewstr = "{ 点@" + tstrstr[i2] + " }"
                                    tnewstr = self.language.name_normal(tnewstr)
                                    ins_json.append({"因为": [tnewstr, "是", "点"]})
                                    tins_json = [{"因为": [tnewstr, "是", "点"]}]
                                    self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                    tconcept_list.append(tnewstr)
                                ins_json.append({"因为": [tconcept_list, "是", "直线"]})
                                tins_json = [{"因为": [tconcept_list, "是", "直线"]}]
                                self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                if 0 != posind:
                                    # 是否删除latex部分
                                    contlist[i1 - 1] = " , ".join(tstrstr[0:posind])
                                else:
                                    del keylist[i1 - 1]
                                    del contlist[i1 - 1]
                                matchsig = 1
                            elif jsonkey in ["是直径"]:
                                for i2 in range(posind, siglenth):
                                    ttconcelist = []
                                    for i3 in tstrli[i2]:
                                        tnewstr = "{ 点@" + i3 + " }"
                                        tnewstr = self.language.name_normal(tnewstr)
                                        ins_json.append({"因为": [tnewstr, "是", "点"]})
                                        tins_json = [{"因为": [tnewstr, "是", "点"]}]
                                        self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                        tconcept_list.append(tnewstr)
                                        ttconcelist.append(tnewstr)
                                    ins_json.append({"因为": [ttconcelist, "是", "直线"]})
                                    tins_json = [{"因为": [ttconcelist, "是", "直线"]}]
                                    self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                    tname = self.language.name_symmetric(tstrstr[i2]).replace(" ", "")
                                    tname = "{直径@" + tname + "}"
                                    ins_json.append({"因为": [tname, "是", "直径"]})
                                    tins_json = [{"因为": [tname, "是", "直径"]}]
                                    self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                # print(ins_json[-1],ins_json[-2])
                                if 0 != posind:
                                    # 是否删除latex部分
                                    contlist[i1 - 1] = " , ".join(tstrstr[0:posind])
                                else:
                                    del keylist[i1 - 1]
                                    del contlist[i1 - 1]
                                matchsig = 1
                            elif jsonkey == "是补角":
                                tstrli = [tstrli[i2] for i2 in range(posind, siglenth)]
                                typsig = "因为"
                                tlist = set()
                                for i2 in tstrli:
                                    if "\\therefore" in i2:
                                        typsig = "所以"
                                        i2.remove("\\therefore")
                                    elif "\\because" in i2:
                                        typsig = "因为"
                                        i2.remove("\\because")
                                    # 必然含 \\angle 否则报错
                                    i2.remove("\\angle")
                                    if i2[0].startswith("{") and i2[0].endswith("}"):
                                        tnewstr = "{ 角@" + i2[0].strip("{}") + " }"
                                        tnewstr = self.language.name_normal(tnewstr)
                                        tlist.add(tnewstr)
                                ins_json.append({typsig: [list(tlist), "是", jsonkey.lstrip("是")]})
                                tins_json = [{typsig: [list(tlist), "是", jsonkey.lstrip("是")]}]
                                self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                matchsig = 1
                            elif jsonkey == "是锐角":
                                tstrli = [i2 for i2 in tstrli[-1] if i2 != "\\angle"]
                                typsig = "因为"
                                for i2 in tstrli:
                                    if i2 == "\\therefore":
                                        typsig = "所以"
                                    if i2.startswith("{") and i2.endswith("}"):
                                        tnewstr = "{ 角@" + i2.strip("{}") + " }"
                                        tnewstr = self.language.name_normal(tnewstr)
                                        ins_json.append({typsig: [tnewstr, "是", jsonkey.lstrip("是")]})
                                        tins_json = [{typsig: [tnewstr, "是", jsonkey.lstrip("是")]}]
                                        self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                        matchsig = 1
                            elif jsonkey in ["是等腰三角形", "是直角三角形"]:
                                tstrli = [i2 for i2 in tstrli[-1] if i2 != "\\triangle"]
                                typsig = "因为"
                                for i2 in tstrli:
                                    if i2 == "\\therefore":
                                        typsig = "所以"
                                    if i2.startswith("{") and i2.endswith("}"):
                                        tnewstr = i2.strip("{}")
                                        tnewstr = self.language.name_cyc_one(tnewstr).replace(" ", "")
                                        tnewstr = "{三角形@" + tnewstr + "}"
                                        ins_json.append({typsig: [tnewstr, "是", jsonkey.lstrip("是")]})
                                        tins_json = [{typsig: [tnewstr, "是", jsonkey.lstrip("是")]}]
                                        self.language2answer(basic_space_ins._setobj, addc=tins_json)
                                        matchsig = 1
                            else:
                                print(jsonkey)
                                raise Exception("在一条直线上")
                    if matchsig == 1:
                        break
            elif keylist[i1] == "text" and keylist[i1 - 1] == "text":
                raise Exception("NO text text type now!")
            elif keylist[i1] == "latex" and keylist[i1 - 1] == "latex":
                raise Exception("NO latex latex type now!")
            else:
                raise Exception("没有考虑到的情况type now!")
            try:
                analist = [[{keylist[i1]: contlist[i1]}]]
                self.get_answerproperty(analist)
            except Exception as e:
                pass
        # 5. 写入句间的实例
        # self.language2answer(basic_space_ins._setobj, addc=ins_json)
        # 6. 提取所有 抽象类。对应实例，改变字符。属性
        analist = [[{keylist[0]: contlist[0]}]]
        anastr = self.get_answerproperty(analist)
        self.answer_json.reverse()
        # 去重
        orlenth = len(self.answer_json)
        for idm in range(orlenth - 1, 0, -1):
            for idn in range(idm):
                if operator.eq(self.answer_json[idm], self.answer_json[idn]):
                    del self.answer_json[idn]
                    break
        return anastr

    def language2answer(self, basic_set, addc=[]):
        """内存：triple交互操作"""
        keydic = {i1.rstrip("集合"): i1 for i1 in basic_set}
        # print("language2answer")
        for oneitems in addc:
            # print(oneitems)
            tmpobj = {}
            itemsig = ""
            if "因为" in oneitems:
                itemsig = "condjson"
                onetri = oneitems["因为"]
            elif "所以" in oneitems:
                itemsig = "outjson"
                onetri = oneitems["所以"]
            elif "求证" in oneitems:
                print(oneitems)
                raise Exception("答题时不应出现求证")
            else:
                print(oneitems)
                raise Exception("没有考虑的情况")
            if onetri[2] in keydic:
                if keydic[onetri[2]] not in tmpobj:
                    if basic_set[keydic[onetri[2]]]["结构形式"] == "一级集合":
                        tmpobj[keydic[onetri[2]]] = set()
                    elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表":
                        tmpobj[keydic[onetri[2]]] = []
                    elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表二级集合":
                        tmpobj[keydic[onetri[2]]] = []
                    elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表二级列表":
                        tmpobj[keydic[onetri[2]]] = []
                    else:
                        print(onetri)
                        raise Exception("没有考虑的情况")
                if basic_set[keydic[onetri[2]]]["结构形式"] == "一级集合":
                    tmpobj[keydic[onetri[2]]].add(onetri[0])
                elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表":
                    tmpobj[keydic[onetri[2]]].append(onetri[0])
                elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表二级集合":
                    tmpobj[keydic[onetri[2]]].append(set())
                    for i1 in onetri[0]:
                        tmpobj[keydic[onetri[2]]][-1].add(i1)
                elif basic_set[keydic[onetri[2]]]["结构形式"] == "一级列表二级列表":
                    tmpobj[keydic[onetri[2]]].append([])
                    for i1 in onetri[0]:
                        tmpobj[keydic[onetri[2]]][-1].append(i1)
                else:
                    print(onetri)
                    raise Exception("没有考虑的情况")
            else:
                # print(keydic)
                print(onetri)
                raise Exception("没有考虑的情况")
            self.answer_json.append({itemsig: tmpobj})
        return None

    def loadspace(self, bs_ins):
        """加载实体空间: """
        pass

    def math_solver_write(self, injson):
        # 1. 操作空间定义
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # 2. json 标准化
        injson = [{"因为": i1} for i1 in injson]
        # 3. 写入
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=injson, delec=[])
        return copy.deepcopy(space_ins._setobj)

    def step_node_write(self, tripleobjlist):
        """加入步骤节点列表：单个 已知按原集合形式的列表。知识点为一个集合。导出按元素处理
        [ [ [[{"线段1"，"线段2"}],"是"，"等值"] , [已知2] ], {知识点1,知识点2},[{a,b},是,xx] ]
        _step_node = {
        "points":[],
        "condjson":{},
        "outjson":{},
        }
        """
        # return None
        sstime = time.time()
        # 1. 先过滤
        # 2. 元素添加
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        listset_obj = [objset for objset in basic_space_ins._setobj if
                       basic_space_ins._setobj[objset]["结构形式"].startswith("一级列表二级")]
        set_obj = [objset for objset in basic_space_ins._setobj if
                   basic_space_ins._setobj[objset]["结构形式"].startswith("一级集合")]
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)

        def json_deliver_equal(json1, json2):
            "节点上下端的json判断"
            bakjson1 = copy.deepcopy(json1)
            bakjson2 = copy.deepcopy(json2)
            for key1 in json1:
                if key1 not in bakjson2:
                    bakjson2[key1] = []
                    if len(json1[key1]) != 0:
                        return "diff"
                if key1 in listset_obj:
                    lenth1 = len(bakjson1[key1])
                    for id1 in range(lenth1 - 1, -1, -1):
                        lenth2 = len(bakjson2[key1])
                        for id2 in range(lenth2 - 1, -1, -1):
                            if operator.eq(set(bakjson1[key1][id1]), set(bakjson2[key1][id2])):
                                del bakjson1[key1][id1]
                                del bakjson2[key1][id2]
                                break
                    if len(bakjson1[key1]) == 0:
                        del bakjson1[key1]
                    if len(bakjson2[key1]) == 0:
                        del bakjson2[key1]
                elif key1 in set_obj:
                    if operator.eq(set(bakjson1[key1]), set(bakjson2[key1])):
                        del bakjson1[key1]
                        del bakjson2[key1]
            if len(bakjson1) == len(bakjson2) and len(bakjson1) == 0:
                return "same"
            else:
                return "diff"

        def gene_cond_outjson(triplobj, step_node):
            "新增条件输入，新增结果输出， 原始输入输出列表,原始步骤json。没有，返回新条件json, 有 返回None,即跳过。"
            incondilist, inpointkeys, inoutlist = triplobj
            incondijson = condition2json(incondilist)
            inoutjson = out2json([inoutlist])
            # 遍历 同一个 知识点的json
            condihavesig = 0
            bakstep_node = copy.deepcopy(step_node)
            for idn, oneori in enumerate(bakstep_node):
                oricondjson, oripointkeys, orioutjson = oneori["condjson"], oneori["points"], oneori["outjson"]
                if operator.eq(set(inpointkeys), set(oripointkeys)):
                    judgestr = json_deliver_equal(incondijson, oricondjson)
                    if judgestr == "same":
                        condihavesig = 1
                        # 只判断输出 结果输出,合并返回
                        orioutjson = outjson2orijson(inoutjson, orijson=orioutjson)
                        step_node[idn]["outjson"] = orioutjson
                    elif judgestr == "diff":
                        # 后面直接写入 条件输入 结果输出
                        pass
            if condihavesig == 1:
                # 已处理不在处理
                return step_node
            else:
                # 没找到才会到这一步，直接写入 条件输入 结果输出
                tmpjson = {"condjson": incondijson, "points": inpointkeys, "outjson": inoutjson}
                step_node.append(tmpjson)
                return step_node

        def gene_cond_outjson_detail(triplobj, step_node):
            "新增条件输入，新增结果输出， 原始输入输出列表,原始步骤json。没有，返回新条件json, 有 返回None,即跳过。"
            incondilist, inpointkeys, inoutlist = triplobj
            incondijson = condition2json(incondilist)
            inoutjson = out2json([inoutlist])
            # 遍历 同一个 知识点的json
            condihavesig = 0
            bakstep_node = copy.deepcopy(step_node)
            for idn, oneori in enumerate(bakstep_node):
                oricondjson, oripointkeys, orioutjson = oneori["condjson"], oneori["points"], oneori["outjson"]
                if operator.eq(set(inpointkeys), set(oripointkeys)):
                    judgeinstr = json_deliver_equal(incondijson, oricondjson)
                    if judgeinstr == "same":
                        judgeoutstr = json_deliver_equal(inoutjson, orioutjson)
                        if judgeoutstr == "same":
                            condihavesig = 1
                            pass
                        elif judgeoutstr == "diff":
                            # 后面直接写入 条件输入 结果输出
                            pass
                    elif judgeinstr == "diff":
                        # 后面直接写入 条件输入 结果输出
                        pass
            if condihavesig == 1:
                # 已处理不在处理
                return step_node
            else:
                # 没找到才会到这一步，直接写入 条件输入 结果输出
                tmpjson = {"condjson": incondijson, "points": inpointkeys, "outjson": inoutjson}
                step_node.append(tmpjson)
                return step_node

        def condition2json(incondilist):
            "条件输入转json"
            condijson = {}
            for onitem in incondilist:
                tmpkey = onitem[2] + "集合"
                if tmpkey not in condijson:
                    condijson[tmpkey] = []
                condijson[tmpkey] += onitem[0]
            return condijson

        def out2json(incondilist):
            "输出条件 转json"
            condijson = {}
            for onitem in incondilist:
                tmpkey = onitem[2] + "集合"
                if tmpkey not in condijson:
                    condijson[tmpkey] = []
                condijson[tmpkey] += onitem[0]
            return self.listlist_deliverall(condijson)

        def outjson2orijson(inoutjson, orijson={}):
            "输出json 添加到原始json"
            for inkey in inoutjson:
                if inkey not in orijson:
                    orijson[inkey] = []
                if inkey in listset_obj:
                    for initem in inoutjson[inkey]:
                        findsig = 0
                        for oriitem in orijson[inkey]:
                            if set(initem).issubset(set(oriitem)):
                                findsig = 1
                                break
                        if findsig == 0:
                            orijson[inkey].append(initem)
                elif inkey in set_obj:
                    for initem in inoutjson[inkey]:
                        if initem not in orijson[inkey]:
                            orijson[inkey].append(initem)
                else:
                    print(inoutjson)
                    raise Exception(inoutjson)
            orijson = self.listlist_deliverall(orijson)
            return orijson

        if self.debugsig:
            print("tree writing length: {}, old length:{}".format(len(tripleobjlist), len(space_ins._step_node)))
        for tripleobj in tripleobjlist:
            # space_ins._step_node = gene_cond_outjson(tripleobj, space_ins._step_node)
            space_ins._step_node = gene_cond_outjson_detail(tripleobj, space_ins._step_node)
        if self.debugsig:
            print(len(space_ins._step_node))
            print("step time:", time.time() - sstime)
        return None

    def prepare_clean_set(self, oldsetobj):
        # 1. 移动含有表达式的条目。
        expresskey = ["+", "-", "*", "\\frac", "\\times", "\\div"]
        express_set = set()
        lenth = len(oldsetobj["等值集合"])
        for ids in range(lenth - 1, -1, -1):
            breaksig = 0
            for elem in oldsetobj["等值集合"][ids]:
                for key in expresskey:
                    if key in elem:
                        express_set.add(" = ".join(list(oldsetobj["等值集合"][ids])))
                        oldsetobj["等值集合"].pop(ids)
                        breaksig = 1
                        break
                if breaksig == 1:
                    break
        oldsetobj["表达式集合"] |= express_set
        # 2. 空间定义
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # 2. 写入
        space_ins._setobj = oldsetobj
        return copy.deepcopy(oldsetobj)

    def inference(self):
        """推理流程: 三元组 到 三元组"""
        field_name = "数学"
        scene_name = "解题"
        # space_name = "basic"
        # basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        old_space_setobj = copy.deepcopy(space_ins._setobj)
        # print(space_ins._initobj)
        # print(space_ins._setobj)
        # print(space_ins._stopobj)
        # raise 115
        # 1. 查找 具体 属性值
        # old_space_setobj = self.prepare_clean_set(old_space_setobj)
        # space_ins._stopobj={"等腰三角形集合": {"{三角形@AMP}"}}
        # space_ins._stopobj = {"全等三角形集合": [{"{三角形@BMP}", "{三角形@NPQ}"}]}
        # false
        # space_ins._stopobj = {"等值集合": [{"{线段@MP}", "{线段@NQ}"}]}
        # ok
        # space_ins._stopobj = {"等值集合": [{"{线段@BM}", "{线段@NP}"}]}
        # false
        # space_ins._stopobj = {"等值集合": [{"{角@BPM}", "{角@NQP}"}]}
        # ok
        # space_ins._stopobj = {"等值集合": [{"{角@BMP}", "{角@PNQ}"}]}
        # false
        # space_ins._stopobj = {"等值集合": [{"{角@MBP}", "{角@NPQ}"}]}
        step_counter = 0
        steplist = {"0": old_space_setobj}
        starttime = time.time()
        # 初始化集合元素
        old_space_setobj = self.axiom2relation(old_space_setobj)
        while True:
            # 推演步骤 打印出 用到的集合元素属性 和 集合元素属性导出的结果。
            # 根据最终结论，倒寻相关的属性概念。根据年级，忽略非考点的属性，即评判的结果。
            step_counter += 1
            logger1.info("in step {}: {}".format(step_counter, old_space_setobj))
            new_space_setobj = self.step_infere(old_space_setobj)
            steplist[str(step_counter)] = copy.deepcopy(new_space_setobj)
            # 5. 判断终止
            judgeres = self.judge_stop(steplist[str(step_counter - 1)], steplist[str(step_counter)], space_ins._stopobj)
            if step_counter == 90:
                logger1.info("步数超长@{}，停止。".format(step_counter))
                raise Exception("步数超长@{}，停止。".format(step_counter))
            logger1.info("stop inference:{}".format(judgeres[0]))
            if judgeres[1]:
                # 1. 停止操作 写记录
                logger1.info("final step: {}".format(new_space_setobj))
                # 2. 停止操作 变更树写入标记
                logger1.info("writing tree info")
                break
            old_space_setobj = steplist[str(step_counter)]
        logger1.info("树节点生成 use time:{}mins".format((time.time() - starttime) / 60))
        # 6. 生成思维树
        return self.get_condition_tree()

    def list_set_deliver(self, inlistset, key):
        "一级列表二级集合，集合传递缩并。如平行 等值 全等"
        key = key.replace("集合", "")
        tripleobjlist = []
        inlistset = [setins for setins in inlistset if setins != set()]
        lenth_paralist = len(inlistset)
        dictest = {}
        for indmain in range(lenth_paralist - 1, 0, -1):
            for indcli in range(indmain - 1, -1, -1):
                if len(set(inlistset[indcli]).intersection(set(inlistset[indmain]))) > 0:
                    # if key == "等值" and len(set(["{线段@AD}", "{线段@MN}"]).intersection(inlistset[indmain])) == 2:
                    #     print(inlistset[indmain], key)
                    #     raise 456
                    if set(inlistset[indcli]).issuperset(set(inlistset[indmain])):
                        del inlistset[indmain]
                        break
                    if set(inlistset[indmain]).issuperset(set(inlistset[indcli])):
                        inlistset[indcli] = inlistset[indmain]
                        del inlistset[indmain]
                        break
                    if self.treesig:
                        told = [list(inlistset[indcli]), list(inlistset[indmain])]
                        tkstr = "".join(set(["".join(told[0]), "".join(told[1])]))
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join(inlistset[indcli])
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
                            # print(indcli, inlistset[indcli], indmain, inlistset[indmain])
                            tripleobjlist.append(
                                [[[told, "是", key]], ["@@{}间传递".format(key)],
                                 [[list(inlistset[indcli] | inlistset[indmain])], "是", key]])
                            # tripleobjlist.append([[[[0], "是", "默认"]], ["@@{}传递".format(key)], [[list(inlistset[indcli])], "是", key]])
                    inlistset[indcli] |= inlistset[indmain]
                    del inlistset[indmain]
                    break
        if self.treesig:
            if self.debugsig:
                print("{}间传递".format(key))
            self.step_node_write(tripleobjlist)
        return inlistset

    def listset_deliverall(self, allobjset):
        for key in allobjset.keys():
            if setobj[key]["结构形式"] in ["一级列表二级集合"] and "二级传递" in setobj[key]["函数"]:
                allobjset[key] = self.list_set_deliver(allobjset[key], key)
        return allobjset

    def listlist_deliverall(self, allobjset):
        for key in allobjset.keys():
            if setobj[key]["结构形式"] in ["一级列表二级集合"] and "二级传递" in setobj[key]["函数"]:
                # print("key")
                # print(key)
                allobjset[key] = list_list_deliver(allobjset[key])
        return allobjset

    def list_set_antiequal(self, objlistset, tarkey="余角", purposekey="等值"):
        "一级列表二级集合，集合反等传递。如：[余角集合 等值集合] 2d的[垂直集合 平行集合] "
        tarkeyset = tarkey + "集合"
        # purposekeyset = purposekey + "集合"
        objset = [setins for setins in objlistset[tarkeyset] if setins != set()]
        lenth_paralist = len(objset)
        outjson = []
        tripleobjlist = []
        dictest = {}
        for indmain in range(lenth_paralist - 1, 0, -1):
            for indcli in range(indmain - 1, -1, -1):
                sameset = objset[indcli].intersection(objset[indmain])
                if len(sameset) == 2:
                    del objset[indmain]
                    break
                elif len(sameset) == 1:
                    outsame = []
                    outsame += [i1 for i1 in objset[indcli] if i1 not in sameset]
                    outsame += [i1 for i1 in objset[indmain] if i1 not in sameset]
                    outsame = list(set(outsame))
                    tkstr = "".join(set(["".join(objset[indcli]), "".join(objset[indmain])]))
                    if tkstr not in dictest:
                        dictest[tkstr] = []
                    tvstr = "".join(outsame)
                    if tvstr not in dictest[tkstr] and outsame != []:
                        dictest[tkstr].append(tvstr)
                        outjson.append([outsame, "是", purposekey])
                        if self.treesig:
                            tripleobjlist.append([[[[list(objset[indcli]), list(objset[indmain])], "是", tarkey]],
                                                  ["@@{}{}反等传递".format(tarkey, purposekey)],
                                                  [[outsame], "是", purposekey]])
                            # tripleobjlist.append(
                            #     [[[[0], "是", "默认"]], ["@@{}{}传递".format(tarkey, purposekey)], [[outsame], "是", purposekey]])
        if self.treesig:
            if self.debugsig:
                print("{}{}反等传递".format(tarkey, purposekey))
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def list_set_equalanti(self, objlistset, tarkey="平行", purposekey="垂直"):
        "一级列表二级集合，集合等值到反关系。如：2d的[平行集合 垂直集合] "
        tarkeyset = tarkey + "集合"
        purposekeyset = purposekey + "集合"
        paraset = [setins for setins in objlistset[tarkeyset] if setins != set()]
        genelist = [setins for setins in objlistset[purposekeyset] if setins != set()]
        lenth_paralist = len(paraset)
        lenth_vertlist = len(genelist)
        outjson = []
        tripleobjlist = []
        dictest = {}
        for indvert in range(lenth_vertlist):
            for indpara in range(lenth_paralist):
                sameset = paraset[indpara].intersection(genelist[indvert])
                if len(sameset) == 2:
                    raise Exception("不可能有两个反关系对象，对应单关系对象。")
                elif len(sameset) == 1:
                    fkey = [i1 for i1 in genelist[indvert] if i1 not in sameset][0]
                    antiout = [i1 for i1 in paraset[indpara] if i1 not in sameset]
                    for onout in antiout:
                        outjson.append([[fkey, onout], "是", purposekey])
                        if self.treesig:
                            tkstr = "".join(paraset[indpara])
                            if tkstr not in dictest:
                                dictest[tkstr] = []
                            tvstr = "".join(genelist[indvert])
                            if tvstr not in dictest[tkstr]:
                                dictest[tkstr].append(tvstr)
                                tripleobjlist.append(
                                    [[[[list(paraset[indpara])], "是", tarkey],
                                      [[list(genelist[indvert])], "是", purposekey]],
                                     ["@@{}{}等反传递".format(tarkey, purposekey)],
                                     [[list(set([fkey, onout]))], "是", purposekey]])
        if self.treesig:
            if self.debugsig:
                print("{}{}等反传递".format(tarkey, purposekey))
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def step_infere(self, oldsetobj):
        "每步推理的具体操作"
        # 1. 概念属性 衍生关系
        newsetobj = self.conception2element(oldsetobj)
        # 2. 公理 衍生关系
        newsetobj = self.axiom2relation(newsetobj)
        # 3. 属性 提取 概念
        newsetobj = self.element2conception(newsetobj)
        # # 猜谜查找具体 实体
        return newsetobj

    def points_relations(self, oldsetobj):
        "根据所有点 和 直线，得到 线段 角 和 三角形"
        pointslist = [point.rstrip("}").lstrip("{点@") for point in oldsetobj["点集合"]]
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        outjson = []
        tripleobjlist = []
        for line in lineslist:
            polist = []
            for point in line:
                tname = self.language.name_symmetric(" ".join(point)).replace(" ", "")
                polist.append("{点@" + tname + "}")
            outjson.append([polist, "是", "直线"])
            if self.treesig:
                tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[polist], "是", "直线"]])
        # 2. 线段
        for c in combinations(pointslist, 2):
            tname = self.language.name_symmetric(" ".join(c)).replace(" ", "")
            tname = "{线段@" + tname + "}"
            outjson.append([tname, "是", "线段"])
            if self.treesig:
                tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[tname], "是", "线段"]])
            tpname1 = self.language.name_symmetric(" ".join(c[0:1])).replace(" ", "")
            tpname1 = "{点@" + tpname1 + "}"
            tpname2 = self.language.name_symmetric(" ".join(c[1:])).replace(" ", "")
            tpname2 = "{点@" + tpname2 + "}"
            tline = list(set([tpname1, tpname2]))
            outjson.append([tline, "是", "直线"])
            if self.treesig:
                tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[tline], "是", "直线"]])
        # 3. 角 三角形
        for c in combinations(pointslist, 3):
            insig = 0
            for oneline in lineslist:
                if set(c).issubset(set(oneline)):
                    insig = 1
                    break
            if insig != 1:
                tname = self.language.name_symmetric(" ".join([c[0], c[1], c[2]])).replace(" ", "")
                tname = "{角@" + tname + "}"
                outjson.append([tname, "是", "角"])
                if self.treesig:
                    tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[tname], "是", "角"]])
                tname = self.language.name_symmetric(" ".join([c[0], c[1], c[2]])).replace(" ", "")
                tname = "{角@" + tname + "}"
                outjson.append([tname, "是", "角"])
                if self.treesig:
                    tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[tname], "是", "角"]])
                tname = self.language.name_symmetric(" ".join([c[0], c[1], c[2]])).replace(" ", "")
                tname = "{角@" + tname + "}"
                outjson.append([tname, "是", "角"])
                if self.treesig:
                    tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[tname], "是", "角"]])
                tname = self.language.name_cyc_one(" ".join(c)).replace(" ", "")
                tname = "{三角形@" + tname + "}"
                outjson.append([tname, "是", "三角形"])
                if self.treesig:
                    tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[tname], "是", "三角形"]])
        if self.treesig:
            if self.debugsig:
                print("points_relations 已知")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def line2comple_relations(self, oldsetobj):
        " 遍历直线，得到补角 点分割线表达式"
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        pointslist = [point.rstrip("}").lstrip("{点@") for point in oldsetobj["点集合"]]
        outjson = []
        tripleobjlist = []
        dictest = {}
        for idn, oneline in enumerate(lineslist):
            lenth_line = len(oneline)
            if self.treesig:
                tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[oldsetobj["直线集合"][idn]], "是", "直线"]])
            if lenth_line > 2:
                nolinepoint = [point for point in pointslist if point not in oneline]
                line_dic = {point: idp for idp, point in enumerate(oneline)}
                # 平角
                for ang_p in combinations(oneline, 3):
                    ang_plist = [[point, line_dic[point]] for point in ang_p]
                    ang_plist = [item[0] for item in sorted(ang_plist, key=lambda x: x[1])]

                    # inde = [oneline.index(pone) for pone in p3]
                    # inde.remove(max(inde))
                    # inde.remove(min(inde))
                    # point = oneline[inde[0]]
                    # p2 = list(p3).remove(point)
                    segm1 = [ang_plist[0], ang_plist[1]]
                    segm2 = [ang_plist[2], ang_plist[1]]
                    segm3 = [ang_plist[0], ang_plist[2]]
                    segm1 = self.language.name_symmetric(" ".join(segm1)).replace(" ", "")
                    segm1 = "{线段@" + segm1 + "}"
                    segm2 = self.language.name_symmetric(" ".join(segm2)).replace(" ", "")
                    segm2 = "{线段@" + segm2 + "}"
                    segm3 = self.language.name_symmetric(" ".join(segm3)).replace(" ", "")
                    segm3 = "{线段@" + segm3 + "}"
                    expresstr = " ".join([segm3, "=", segm2, "+", segm1])
                    outjson.append([expresstr, "是", "表达式"])
                    if self.treesig:
                        tkstr = "".join(set(oldsetobj["直线集合"][idn]))
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join([expresstr, "表达式"])
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
                            tripleobjlist.append(
                                [[[[oldsetobj["直线集合"][idn]], "是", "直线"]], ["@@直线得出表达式"], [[expresstr], "是", "表达式"]])

                    tname = self.language.name_symmetric(" ".join(ang_plist)).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "平角"])
                    if self.treesig:
                        tkstr = "".join(set(oldsetobj["直线集合"][idn]))
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join([tname, "平角"])
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
                            tripleobjlist.append(
                                [[[[oldsetobj["直线集合"][idn]], "是", "直线"]], ["@@直线得出补角"], [[tname], "是", "平角"]])
                    # 补角
                    for outpoint in nolinepoint:
                        t_ang_plist1 = ang_plist[0:2] + [outpoint]
                        t_ang_plist2 = [outpoint] + ang_plist[1:]
                        tname1 = self.language.name_symmetric(" ".join(t_ang_plist1)).replace(" ", "")
                        tname1 = "{角@" + tname1 + "}"
                        tname2 = self.language.name_symmetric(" ".join(t_ang_plist2)).replace(" ", "")
                        tname2 = "{角@" + tname2 + "}"
                        comlist = [tname1, tname2]
                        outjson.append([comlist, "是", "补角"])
                        if self.treesig:
                            tkstr = "".join(set(oldsetobj["直线集合"][idn]))
                            if tkstr not in dictest:
                                dictest[tkstr] = []
                            tvstr = "".join(comlist)
                            if tvstr not in dictest[tkstr]:
                                dictest[tkstr].append(tvstr)
                                tripleobjlist.append(
                                    [[[[oldsetobj["直线集合"][idn]], "是", "直线"]], ["@@直线得出补角"], [[comlist], "是", "补角"]])
                        c1ttype1 = "钝角" if tname2 in oldsetobj["锐角集合"] else ""
                        c1ttype1 = "锐角" if tname2 in oldsetobj["钝角集合"] else c1ttype1
                        c2ttype2 = "钝角" if tname1 in oldsetobj["锐角集合"] else ""
                        c2ttype2 = "锐角" if tname1 in oldsetobj["钝角集合"] else c2ttype2
                        if c2ttype2 == c1ttype1 and c2ttype2 == "":
                            continue
                        if c1ttype1 == "":
                            c1ttype1 = "锐角" if c2ttype2 == "钝角" else "钝角"
                        if c2ttype2 == "":
                            c2ttype2 = "锐角" if c1ttype1 == "钝角" else "钝角"
                        outjson.append([tname1, "是", c1ttype1])
                        outjson.append([tname2, "是", c2ttype2])
                        if self.treesig:
                            # tripleobjlist.append([[[[tname1], "是", c1ttype1], [[tname], "是", "平角"]], ["@@补角属性"],
                            #                       [[tname2], "是", c2ttype2]])
                            # tripleobjlist.append([[[[tname2], "是", c2ttype2], [[tname], "是", "平角"]], ["@@补角属性"],
                            #                       [[tname1], "是", c1ttype1]])
                            tkstr = "".join(["默认"])
                            if tkstr not in dictest:
                                dictest[tkstr] = []
                            tvstr = "".join([tname2, c2ttype2])
                            if tvstr not in dictest[tkstr]:
                                dictest[tkstr].append(tvstr)
                                tripleobjlist.append([[[[0], "是", "默认"]], ["@@补角属性"], [[tname2], "是", c2ttype2]])
                            tvstr = "".join([tname1, c1ttype1])
                            if tvstr not in dictest[tkstr]:
                                dictest[tkstr].append(tvstr)
                                tripleobjlist.append([[[[0], "是", "默认"]], ["@@补角属性"], [[tname1], "是", c1ttype1]])
        if self.treesig:
            if self.debugsig:
                print("line2comple_relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def lines2angle_equal(self, oldsetobj):
        " 遍历直线，得到 等角 等值集合 "
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        outjson = []
        tripleobjlist = []
        for twoline in combinations(lineslist, 2):
            sameset = set(twoline[0]).intersection(set(twoline[1]))
            if len(sameset) == 1:
                anelem = sameset.pop()
                inde1 = twoline[0].index(anelem)
                inde2 = twoline[1].index(anelem)
                malist1 = [elem for idn, elem in enumerate(twoline[0]) if idn < inde1]
                malist2 = [elem for idn, elem in enumerate(twoline[0]) if idn > inde1]
                cllist1 = [elem for idn, elem in enumerate(twoline[1]) if idn < inde2]
                cllist2 = [elem for idn, elem in enumerate(twoline[1]) if idn > inde2]
                teqlist = set()
                for maelem in malist1:
                    for clelem in cllist1:
                        tname = self.language.name_symmetric(" ".join([maelem, anelem, clelem])).replace(" ", "")
                        tname = "{角@" + tname + "}"
                        teqlist.add(tname)
                if len(teqlist) > 0:
                    teqlist = list(teqlist)
                    outjson.append([teqlist, "是", "等值"])
                    if self.treesig:
                        tripleobjlist.append([[[[0], "是", "默认"]], ["@@已知"], [[teqlist], "是", "等值"]])
                teqlist = set()
                for maelem in malist2:
                    for clelem in cllist1:
                        tname = self.language.name_symmetric(" ".join([maelem, anelem, clelem])).replace(" ", "")
                        tname = "{角@" + tname + "}"
                        teqlist.add(tname)
                if len(teqlist) > 0:
                    teqlist = list(teqlist)
                    outjson.append([teqlist, "是", "等值"])
                    if self.treesig:
                        tripleobjlist.append([[[[0], "是", "默认"]], ["@@同角表示"], [[teqlist], "是", "等值"]])
                teqlist = set()
                for maelem in malist1:
                    for clelem in cllist2:
                        tname = self.language.name_symmetric(" ".join([maelem, anelem, clelem])).replace(" ", "")
                        tname = "{角@" + tname + "}"
                        teqlist.add(tname)
                if len(teqlist) > 0:
                    teqlist = list(teqlist)
                    outjson.append([teqlist, "是", "等值"])
                    if self.treesig:
                        tripleobjlist.append([[[[0], "是", "默认"]], ["@@同角表示"], [[teqlist], "是", "等值"]])
                teqlist = set()
                for maelem in malist2:
                    for clelem in cllist2:
                        tname = self.language.name_symmetric(" ".join([maelem, anelem, clelem])).replace(" ", "")
                        tname = "{角@" + tname + "}"
                        teqlist.add(tname)
                if len(teqlist) > 0:
                    teqlist = list(teqlist)
                    outjson.append([teqlist, "是", "等值"])
                    if self.treesig:
                        tripleobjlist.append([[[[0], "是", "默认"]], ["@@同角表示"], [[teqlist], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("lines2angle_equal")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def vert2Rt_relations(self, oldsetobj):
        " 遍历垂直，得到直角，直角三角形 "
        # 如果 线段的点 全在一条直线上，两条线上的任意一对都垂直。如果垂直的有 共同点，改组为直角。改代表角为直角三角形
        outjson = []
        tripleobjlist = []
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        vertlist = [[segm.rstrip("}").lstrip("{线段@") for segm in segms] for segms in oldsetobj["垂直集合"]]
        # print(vertlist)
        vertlist = [[latex_fenci(latex2space(item2)) for item2 in item1] for item1 in vertlist]
        dictest = {}
        for segm1, segm2 in vertlist:
            segmlist1, segmlist2 = [segm1], [segm2]
            tseedname1 = self.language.name_symmetric(" ".join(segm1)).replace(" ", "")
            tseedname2 = self.language.name_symmetric(" ".join(segm2)).replace(" ", "")
            tseedname1 = "{线段@" + tseedname1 + "}"
            tseedname2 = "{线段@" + tseedname2 + "}"
            # 根据直线获取垂直组
            for oneline in lineslist:
                if set(segm1).issubset(set(oneline)):
                    segmlist1 = [segm for segm in combinations(oneline, 2)]
                if set(segm2).issubset(set(oneline)):
                    segmlist2 = [segm for segm in combinations(oneline, 2)]
            # print(segmlist1)
            # print(segmlist2)
            # print("segmlist")
            # 垂直组 互相组合
            for vertsegm1, vertsegm2 in itertools.product(segmlist1, segmlist2):
                vertsegm1 = list(vertsegm1)
                vertsegm2 = list(vertsegm2)
                tname1 = self.language.name_symmetric(" ".join(vertsegm1)).replace(" ", "")
                tname2 = self.language.name_symmetric(" ".join(vertsegm2)).replace(" ", "")
                tname1 = "{线段@" + tname1 + "}"
                tname2 = "{线段@" + tname2 + "}"
                tkstr = "".join(["默认"])
                if tkstr not in dictest:
                    dictest[tkstr] = []
                tvstr = "".join(set([tname1, tname2, "垂直"]))
                if tvstr not in dictest[tkstr]:
                    dictest[tkstr].append(tvstr)
                    outjson.append([[tname1, tname2], "是", "垂直"])
                    if self.treesig:
                        tripleobjlist.append([[[[0], "是", "默认"]], ["@@垂直直线的线段属性"],
                                              [[list(set([tname1, tname2]))], "是", "垂直"]])
                        # tripleobjlist.append([[[[[tseedname1, tseedname2]], "是", "垂直"]], ["@@垂直直线的线段属性"],
                        #                       [[list(set([tname1, tname2]))], "是", "垂直"]])
                insetlist = list(set(vertsegm1).intersection(set(vertsegm2)))
                # 有公共点 生成角和三角形
                if len(insetlist) == 1:
                    vertsegm1.remove(insetlist[0])
                    vertsegm2.remove(insetlist[0])
                    tanlgelist = vertsegm1 + insetlist + vertsegm2
                    tname = self.language.name_symmetric(" ".join(tanlgelist)).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    tkstr = "".join(["默认"])
                    if tkstr not in dictest:
                        dictest[tkstr] = []
                    tvstr = "".join(set([tname, "直角"]))
                    if tvstr not in dictest[tkstr]:
                        dictest[tkstr].append(tvstr)
                        outjson.append([tname, "是", "直角"])
                        if self.treesig:
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@垂直直角的属性"], [[tname], "是", "直角"]])
                            # tripleobjlist.append([[[[[tseedname1, tseedname2]], "是", "垂直"]], ["@@垂直直角的属性"],
                            #                       [[tname], "是", "直角"]])
                    tname = self.language.name_symmetric(" ".join(insetlist + vertsegm1 + vertsegm2)).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "角"])
                    tkstr = "".join(["默认"])
                    if tkstr not in dictest:
                        dictest[tkstr] = []
                    tvstr = "".join(set([tname, "锐角"]))
                    if tvstr not in dictest[tkstr]:
                        dictest[tkstr].append(tvstr)
                        outjson.append([tname, "是", "锐角"])
                        if self.treesig:
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@余角性质"], [[tname], "是", "锐角"]])
                            # tripleobjlist.append(
                            #     [[[[[tseedname1, tseedname2]], "是", "垂直"]], ["@@余角性质"], [[tname], "是", "锐角"]])
                    tname = self.language.name_symmetric(" ".join(vertsegm1 + vertsegm2 + insetlist)).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "角"])
                    tkstr = "".join(["默认"])
                    if tkstr not in dictest:
                        dictest[tkstr] = []
                    tvstr = "".join(set([tname, "锐角"]))
                    if tvstr not in dictest[tkstr]:
                        dictest[tkstr].append(tvstr)
                        outjson.append([tname, "是", "锐角"])
                        if self.treesig:
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@余角性质"], [[tname], "是", "锐角"]])
                            # tripleobjlist.append(
                            #     [[[[[tseedname1, tseedname2]], "是", "垂直"]], ["@@余角性质"], [[tname], "是", "锐角"]])
                    tname = self.language.name_cyc_one(" ".join(tanlgelist)).replace(" ", "")
                    tname = "{三角形@" + tname + "}"
                    tkstr = "".join(["默认"])
                    if tkstr not in dictest:
                        dictest[tkstr] = []
                    tvstr = "".join(set([tname, "直角三角形"]))
                    if tvstr not in dictest[tkstr]:
                        dictest[tkstr].append(tvstr)
                        outjson.append([tname, "是", "直角三角形"])
                        if self.treesig:
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@垂直性质"], [[tname], "是", "直角三角形"]])
                            # tripleobjlist.append(
                            #     [[[[[tseedname1, tseedname2]], "是", "垂直"]], ["@@垂直性质"], [[tname], "是", "直角三角形"]])
        if self.treesig:
            if self.debugsig:
                print("vert2Rt_relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def parali2segm_relations(self, oldsetobj):
        " 平行传递。线段是元素，多点直线作为多个元素处理。不同组间有重复的元素，则合并 "
        outjson = []
        tripleobjlist = []
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        paralist = [[segm.rstrip("}").lstrip("{线段@") for segm in segms] for segms in oldsetobj["平行集合"]]
        paralist = [[latex_fenci(latex2space(item2)) for item2 in item1] for item1 in paralist]
        dictest = {}
        for onegroup in paralist:
            newgroup = []
            idlinegroup = []
            for segmi in onegroup:
                newgroup.append(segmi)
                for idtn, oneline in enumerate(lineslist):
                    if set(segmi).issubset(set(oneline)):
                        newgroup += [segm for segm in combinations(oneline, 2)]
                        idlinegroup.append(oldsetobj["直线集合"][idtn])
                        break
            strlist = []
            for segmi in newgroup:
                strlist.append(self.language.name_symmetric(" ".join(segmi)).replace(" ", ""))
            strlist = ["{线段@" + segmi + "}" for segmi in strlist]
            tkstr = "".join(["默认"])
            if tkstr not in dictest:
                dictest[tkstr] = []
            tvstr = "".join(set(strlist))
            if tvstr not in dictest[tkstr]:
                dictest[tkstr].append(tvstr)
                outjson.append([strlist, "是", "平行"])
                if self.treesig:
                    # tripleobjlist.append([[[[strlist[0]], "是", "平行"], [idlinegroup, "是", "直线"]], ["@@平行属性"],
                    #                       [[list(set(strlist))], "是", "平行"]])
                    tripleobjlist.append([[[[0], "是", "默认"]], ["@@平行属性"], [[strlist], "是", "平行"]])
        if self.treesig:
            if self.debugsig:
                print("parali2segm_relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def parapara2equal(self, oldsetobj):
        "平行集 交 平行集 2 等值集"
        outjson = []
        tripleobjlist = []
        paralist = [[segm.rstrip("}").lstrip("{线段@") for segm in segms] for segms in oldsetobj["平行集合"]]
        paralist = [[latex_fenci(latex2space(item2)) for item2 in item1] for item1 in paralist]
        for segmlist1, segmlist2 in combinations(paralist, 2):
            for pairseg11, pairseg12 in combinations(segmlist1, 2):
                for pairseg21, pairseg22 in combinations(segmlist2, 2):
                    if len(set(pairseg21 + pairseg22 + pairseg11 + pairseg12)) == 4:
                        tname11 = self.language.name_symmetric(" ".join(pairseg11)).replace(" ", "")
                        tname12 = self.language.name_symmetric(" ".join(pairseg12)).replace(" ", "")
                        tname21 = self.language.name_symmetric(" ".join(pairseg21)).replace(" ", "")
                        tname22 = self.language.name_symmetric(" ".join(pairseg22)).replace(" ", "")
                        pairseg1 = ["{线段@" + tname11 + "}", "{线段@" + tname12 + "}"]
                        pairseg2 = ["{线段@" + tname21 + "}", "{线段@" + tname22 + "}"]
                        outjson.append([pairseg1, "是", "等值"])
                        outjson.append([pairseg2, "是", "等值"])
                        if self.treesig:
                            tripleobjlist.append([[[[pairseg1, pairseg2], "是", "平行"]], ["@@平行线间平行线等值"],
                                                  [[pairseg1, pairseg2], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("parapara2equal")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def corresangles2relations(self, oldsetobj):
        " 遍历平行，对顶角 "
        outjson = []
        tripleobjlist = []
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        paralist = [[segm.rstrip("}").lstrip("{线段@") for segm in segms] for segms in oldsetobj["平行集合"]]
        paralist = [[latex_fenci(latex2space(item2)) for item2 in item1] for item1 in paralist]
        for idl, line in enumerate(lineslist):
            noselflines = copy.deepcopy(lineslist)
            del noselflines[idl]
            for idp, onegroup in enumerate(paralist):
                # 5.1得出 一条主线 与 一组平行线 的 交点
                corres_updn_12ang = []
                posi_intersec = {ide: elem for ide, elem in enumerate(line) if elem in set(itertools.chain(*onegroup))}
                # 5.2与一个线段生成上位角和下位角。
                # 5.3不同 交点 的 上位角或下位角 如果同是锐角或钝角则相同。
                for posi in posi_intersec:
                    tlist_ap = line[:posi]
                    tlist_an = line[posi + 1:]
                    for segmi in onegroup:
                        if posi_intersec[posi] in segmi and not set(segmi).issubset(set(line)):
                            # 相交的线段，必存在直线。且不属于原直线
                            segline = [tmline for tmline in noselflines if set(segmi).issubset(set(tmline))][0]
                            seglinedic = {elem: ide for ide, elem in enumerate(segline)}
                            ide_b = seglinedic[posi_intersec[posi]]
                            tlist_bp = segline[:ide_b]
                            tlist_bn = segline[ide_b + 1:]
                            # 对顶角1
                            inner_upp = []
                            for pair in list(itertools.product(tlist_ap, tlist_bp)):
                                tlist = [pair[0], posi_intersec[posi], pair[1]]
                                tname = self.language.name_symmetric(" ".join(tlist)).replace(" ", "")
                                inner_upp.append("{角@" + tname + "}")
                            inner_dnn = []
                            for pair in list(itertools.product(tlist_an, tlist_bn)):
                                tlist = [pair[0], posi_intersec[posi], pair[1]]
                                tname = self.language.name_symmetric(" ".join(tlist)).replace(" ", "")
                                inner_dnn.append("{角@" + tname + "}")
                            outjson.append([inner_upp + inner_dnn, "是", "等值"])
                            if self.treesig:
                                if inner_upp + inner_dnn != []:
                                    tripleobjlist.append(
                                        [[[[oldsetobj["直线集合"][idl]], "是", "直线"],
                                          [[list(oldsetobj["平行集合"][idp])], "是", "平行"]],
                                         ["@@同位角对顶角内错角属性"], [[inner_upp + inner_dnn], "是", "等值"]])
                                    # tripleobjlist.append(
                                    #     [[[[0], "是", "默认"]], ["@@定位角对顶角内错角属性"], [[inner_upp + inner_dnn], "是", "等值"]])
                            # 对顶角2
                            inner_upn = []
                            for pair in list(itertools.product(tlist_ap, tlist_bn)):
                                tlist = [pair[0], posi_intersec[posi], pair[1]]
                                tname = self.language.name_symmetric(" ".join(tlist)).replace(" ", "")
                                inner_upn.append("{角@" + tname + "}")
                            inner_dnp = []
                            for pair in list(itertools.product(tlist_an, tlist_bp)):
                                tlist = [pair[0], posi_intersec[posi], pair[1]]
                                tname = self.language.name_symmetric(" ".join(tlist)).replace(" ", "")
                                inner_dnp.append("{角@" + tname + "}")
                            outjson.append([inner_dnp + inner_upn, "是", "等值"])
                            if self.treesig:
                                if inner_dnp + inner_upn != []:
                                    tripleobjlist.append(
                                        [[[[oldsetobj["直线集合"][idl]], "是", "直线"],
                                          [[list(oldsetobj["平行集合"][idp])], "是", "平行"]],
                                         ["@@同位角对顶角内错角属性"], [[inner_dnp + inner_upn], "是", "等值"]])
                                    # tripleobjlist.append(
                                    #     [[[[0], "是", "默认"]], ["@@定位角对顶角内错角属性"], [[inner_dnp + inner_upn], "是", "等值"]])
                            corres_updn_12ang.append([inner_upp, inner_upn, inner_dnp, inner_dnn])
                # 根据类型判断相等
                acute_set = []
                obtuse_set = []
                for onegroup in corres_updn_12ang:
                    for angli in onegroup:
                        if len(set(angli).intersection(oldsetobj["锐角集合"])) > 0:
                            acute_set += angli
                        elif len(set(angli).intersection(oldsetobj["钝角集合"])) > 0:
                            obtuse_set += angli
                        else:
                            pass
                outjson.append([acute_set, "是", "等值"])
                outjson.append([obtuse_set, "是", "等值"])
                if self.treesig:
                    if acute_set != []:
                        tripleobjlist.append(
                            [[[[oldsetobj["直线集合"][idl]], "是", "直线"], [acute_set, "是", "锐角"]], ["@@同位角对顶角内错角属性"],
                             [[acute_set], "是", "等值"]])
                    if obtuse_set != []:
                        tripleobjlist.append(
                            [[[[oldsetobj["直线集合"][idl]], "是", "直线"], [obtuse_set, "是", "钝角"]], ["@@同位角对顶角内错角属性"],
                             [[obtuse_set], "是", "等值"]])
                        # tripleobjlist.append([[[[0], "是", "默认"]], ["@@定位角对顶角内错角属性"], [[acute_set], "是", "等值"]])
                        # tripleobjlist.append([[[[0], "是", "默认"]], ["@@定位角对顶角内错角属性"], [[obtuse_set], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("corresangles2relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def addconstantequal(self, oldsetobj):
        " 加入恒等选项 "
        outjson = []
        tripleobjlist = []
        # oldsetobj["等值集合"] += [set([elem]) for elem in oldsetobj["线段集合"]]
        # oldsetobj["等值集合"] += [set([elem]) for elem in oldsetobj["角集合"]]
        # oldsetobj["等值集合"].append(copy.deepcopy(oldsetobj["直角集合"]))
        # oldsetobj["等值集合"].append(copy.deepcopy(oldsetobj["平角集合"]))
        outjson.append([copy.deepcopy(oldsetobj["直角集合"]), "是", "等值"])
        outjson.append([copy.deepcopy(oldsetobj["平角集合"]), "是", "等值"])
        if self.treesig:
            tripleobjlist.append([[[[0], "是", "默认"]], ["@@平角相等"], [[list(oldsetobj["平角集合"])], "是", "等值"]])
            tripleobjlist.append([[[[0], "是", "默认"]], ["@@直角相等"], [[list(oldsetobj["直角集合"])], "是", "等值"]])
        for elem in oldsetobj["线段集合"] | oldsetobj["角集合"]:
            outjson.append([[elem], "是", "等值"])
            if self.treesig:
                tripleobjlist.append([[[[0], "是", "默认"]], ["@@自等性质"], [[[elem]], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("addconstantequal")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def rttriang2remain_relations(self, oldsetobj):
        " 遍历直角三角形 垂直已导出角可以略过，得到余角 "
        outjson = []
        oldsetobj["直角三角形集合"] = list(oldsetobj["直角三角形集合"])
        rtlist = [elems.rstrip("}").lstrip("{三角形@") for elems in oldsetobj["直角三角形集合"]]
        rtlist = [latex_fenci(latex2space(angli)) for angli in rtlist]
        tripleobjlist = []
        for idn, points in enumerate(rtlist):
            strlist = []
            tpoilist = points + points[0:2]
            strlist.append(self.language.name_symmetric(" ".join(tpoilist[0:3])).replace(" ", ""))
            strlist.append(self.language.name_symmetric(" ".join(tpoilist[1:4])).replace(" ", ""))
            strlist.append(self.language.name_symmetric(" ".join(tpoilist[2:5])).replace(" ", ""))
            strlist = ["{角@" + angli + "}" for angli in strlist]
            strlist = [angli for angli in strlist if angli not in oldsetobj["直角集合"]]
            outjson.append([strlist, "是", "余角"])
            outjson.append([strlist[0], "是", "锐角"])
            outjson.append([strlist[1], "是", "锐角"])
            if self.treesig:
                tripleobjlist.append(
                    [[[[oldsetobj["直角三角形集合"][idn]], "是", "直角三角形"]], ["@@直角三角形属性必要条件"], [[strlist], "是", "余角"]])
                tripleobjlist.append(
                    [[[[oldsetobj["直角三角形集合"][idn]], "是", "直角三角形"]], ["@@直角三角形属性必要条件"], [[strlist[0]], "是", "锐角"]])
                tripleobjlist.append(
                    [[[[oldsetobj["直角三角形集合"][idn]], "是", "直角三角形"]], ["@@直角三角形属性必要条件"], [[strlist[1]], "是", "锐角"]])
        if self.treesig:
            if self.debugsig:
                print("rttriang2remain_relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def equall2dunrui_relations(self, oldsetobj):
        outjson = []
        tripleobjlist = []
        dictest = {}
        for ruijiao in oldsetobj["锐角集合"]:
            for equals in oldsetobj["等值集合"]:
                if ruijiao in equals:
                    for elem in equals:
                        tkstr = "".join(["默认"])
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join([elem])
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
                            outjson.append([elem, "是", "锐角"])
                            if self.treesig:
                                # tripleobjlist.append(
                                #     [[[[[elem, ruijiao]], "是", "等值"], [[ruijiao], "是", "锐角"]], ["@@等值钝角传递"],
                                #      [[elem], "是", "锐角"]])
                                tripleobjlist.append([[[[0], "是", "默认"]], ["@@等值锐角传递"], [[elem], "是", "锐角"]])
        for dunjiao in oldsetobj["钝角集合"]:
            for equals in oldsetobj["等值集合"]:
                if dunjiao in equals:
                    for elem in equals:
                        tkstr = "".join(["默认"])
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join([elem])
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
                            outjson.append([elem, "是", "钝角"])
                            if self.treesig:
                                # tripleobjlist.append(
                                #     [[[[[elem, dunjiao]], "是", "等值"], [[ruijiao], "是", "钝角"]], ["@@等值钝角传递"],
                                #      [[elem], "是", "钝角"]])
                                tripleobjlist.append([[[[0], "是", "默认"]], ["@@等值钝角传递"], [[elem], "是", "钝角"]])
        if self.treesig:
            if self.debugsig:
                print("equall2dunrui_relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def degree2angle_relations(self, oldsetobj):
        " 找等于90度和180度的，作为直角 直角三角形 补角 直线。"
        # print("degree2angle_relations")
        outjson = []
        tripleobjlist = []
        for oneset in oldsetobj["等值集合"]:
            # 每个集合中找非属性的表达式，如果计算值小于误差，则为直角 或 平角
            findsig = 0
            for elem in oneset:
                if "@" not in elem:
                    vastr = solve_latex_formula2(elem, varlist=["x"], const_dic={"\\pi": "3.14"})
                    if abs(vastr[0]["x"] - 1.57) < 1e-3:
                        findsig = "直角"
                    if abs(vastr[0]["x"] - 3.14) < 1e-3:
                        findsig = "平角"
            if findsig != 0:
                for elem in oneset:
                    if "@" in elem:
                        outjson.append([elem, "是", findsig])
                        # print(outjson[-1])
                        if self.treesig:
                            # tripleobjlist.append([[[list(oneset), "是", "表达式"]], ["@@角度角类型"], [[elem], "是", findsig]])
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@角度角类型"], [[elem], "是", findsig]])
                        tpoilist = latex_fenci(latex2space(elem.rstrip("}").lstrip("{角@")))
                        if "平角" == findsig:
                            strlist = []
                            for point in tpoilist:
                                strlist.append(self.language.name_symmetric(" ".join(point)).replace(" ", ""))
                            strlist = ["{点@" + point + "}" for point in strlist]
                            outjson.append([strlist, "是", "直线"])
                            if self.treesig:
                                # tripleobjlist.append(
                                #     [[[list(oneset), "是", "表达式"]], ["@@角度角类型"], [[strlist], "是", "直线"]])
                                tripleobjlist.append([[[[0], "是", "默认"]], ["@@角度角类型"], [[strlist], "是", "直线"]])
                        else:
                            pass
        if self.treesig:
            if self.debugsig:
                print("degree2angle_relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def express2compleremain_relations(self, oldsetobj):
        " 找表达式含度的，含90度和180度的，移项 根据 其他表达式或 等值关系，得出 补角 余角。"
        # print("express2compleremain_relations")
        outjson = []
        tripleobjlist = []
        for oneset in oldsetobj["表达式集合"]:
            if "^ { \\circ }" in oneset:
                # print("表达式:")
                # print(oneset)
                anglelems = [elem for elem in oneset.split()]
                anglelems = ["1 8 0 ^ { \\circ }" if elem in oldsetobj["平角集合"] else elem for elem in anglelems]
                anglelems = ["9 0 ^ { \\circ }" if elem in oldsetobj["直角集合"] else elem for elem in anglelems]
                newoneset = " ".join(anglelems)
                # print(anglelems)
                # print(newoneset)
                anglelems = [elem for elem in newoneset.split() if elem.startswith("{角@")]
                # print(anglelems)
                vastr = solve_latex_equation(newoneset, varlist=anglelems, const_dic={"\\pi": "3.14"})
                tmpkeys = []
                for angobj in vastr:
                    tkey = list(angobj.keys())[0]
                    tmpkeys.append(tkey)
                # print(tmpkeys)
                # print(vastr)
                for angobj in vastr:
                    tkey = list(angobj.keys())[0]
                    tvalue = str(angobj[tkey])
                    tvalue = tvalue.replace("1.0*", "")
                    tangl = tvalue.replace("1.57 - ", "")
                    # print(123)
                    # print(tkey)
                    # print(tvalue)
                    # print(tangl)
                    if tangl in tmpkeys and tangl != tkey:
                        outjson.append([[tangl, tkey], "是", "余角"])
                        # print(outjson[-1])
                        if self.treesig:
                            # tripleobjlist.append([[[[0], "是", "默认"]], ["@@表达式性质"], [[oneset], "是", "表达式"]])
                            tripleobjlist.append([[[[oneset], "是", "表达式"]], ["@@表达式性质"], [[[tangl, tkey]], "是", "余角"]])
                            # tripleobjlist.append([[[[0], "是", "默认"]], ["@@表达式性质"], [[[tangl, tkey]], "是", "余角"]])
                    tangl = tvalue.replace("3.14 - ", "")
                    if tangl in tmpkeys:
                        outjson.append([[tangl, tkey], "是", "补角"])
                        # print(outjson[-1])
                        if self.treesig:
                            # tripleobjlist.append([[[[0], "是", "默认"]], ["@@表达式性质"], [[oneset], "是", "表达式"]])
                            tripleobjlist.append([[[[oneset], "是", "表达式"]], ["@@表达式性质"], [[[tangl, tkey]], "是", "补角"]])
                            # tripleobjlist.append([[[[0], "是", "默认"]], ["@@表达式性质"], [[[tangl, tkey]], "是", "补角"]])
        if self.treesig:
            if self.debugsig:
                print("express2compleremain_relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def express2relations(self, oldsetobj):
        " 找表达式 含度的，含90度和180度的，移项 根据 其他表达式或 等值关系，得出 反传递关系。"

        def expres2list(inexpre):
            "等式标准化 x = y + z 如果不能写成等号 单边 1项 另边 2项 的形式，当做复杂表达式处理，临时忽略。不忽略的只考虑+-分割，两个一样剩下的就相同"
            inexpre = inexpre.replace(" ", "")
            sidelist = inexpre.split("=")
            if len(sidelist) != 2:
                return None
            orilist = re.split('\-|\+|=', inexpre)
            if len(orilist) > 3:
                return None
            # else:
            #     return orilist
            sidem1 = sidelist[0].split("-")
            sidem2 = sidelist[1].split("-")
            sidep1 = sidelist[0].split("+")
            sidep2 = sidelist[1].split("+")
            if len(sidem1) > 1:
                sidep2.append(sidem1[-1])
                sidem1.pop()
            if len(sidem2) > 1:
                sidep1.append(sidem2[-1])
                sidem2.pop()
            if len(sidep1) > 1:
                sidepa = sidep2 + sidep1
                sidepa = [latex_fenci(ite)[0] if "@" not in ite else ite for ite in sidepa]
                return sidepa
            if len(sidep2) > 1:
                sidepa = sidep1 + sidep2
                sidepa = [latex_fenci(ite)[0] if "@" not in ite else ite for ite in sidepa]
                return sidepa

        outjson = []
        tripleobjlist = []
        expgrouplist = []
        for oneexpre in oldsetobj["表达式集合"]:
            # 表达式分解后单项
            explist = expres2list(oneexpre)
            if explist:
                explenth = len(explist)
                if explenth == 2:
                    raise Exception("表达式尚未考虑。")
                elif explenth == 3:
                    expgrouplist.append(explist)
                else:
                    raise Exception("表达式尚未考虑。")
        for comlist in combinations(expgrouplist, 2):
            # 遍历每一组表达式 (3) + (3)
            counter = [0] * 5
            eqlist0 = set([comlist[0][0], comlist[1][0]])
            eqlist1 = set([comlist[0][1], comlist[1][1]])
            eqlist2 = set([comlist[0][2], comlist[1][2]])
            eqlist3 = set([comlist[0][1], comlist[1][2]])
            eqlist4 = set([comlist[0][2], comlist[1][1]])
            for equset in oldsetobj["等值集合"]:
                if len(eqlist0.intersection(equset)) > 1:
                    counter[0] = 1
                if len(eqlist1.intersection(equset)) > 1:
                    counter[1] = 1
                if len(eqlist2.intersection(equset)) > 1:
                    counter[2] = 1
                if len(eqlist3.intersection(equset)) > 1:
                    counter[3] = 1
                if len(eqlist4.intersection(equset)) > 1:
                    counter[4] = 1
            eqlist0 = list(eqlist0)
            eqlist1 = list(eqlist1)
            eqlist2 = list(eqlist2)
            eqlist3 = list(eqlist3)
            eqlist4 = list(eqlist4)
            if counter[0] == 1 and counter[1] == 1:
                outjson.append([eqlist2, "是", "等值"])
                if self.treesig:
                    tripleobjlist.append([[[[eqlist0, eqlist1], "是", "等值"]], ["@@表达式传递"], [[eqlist2], "是", "等值"]])
            if counter[0] == 1 and counter[2] == 1:
                outjson.append([eqlist1, "是", "等值"])
                if self.treesig:
                    tripleobjlist.append([[[[eqlist0, eqlist2], "是", "等值"]], ["@@表达式传递"], [[eqlist1], "是", "等值"]])
            if counter[0] == 1 and counter[3] == 1:
                outjson.append([eqlist4, "是", "等值"])
                if self.treesig:
                    tripleobjlist.append([[[[eqlist0, eqlist3], "是", "等值"]], ["@@表达式传递"], [[eqlist4], "是", "等值"]])
            if counter[0] == 1 and counter[4] == 1:
                outjson.append([eqlist3, "是", "等值"])
                if self.treesig:
                    tripleobjlist.append([[[[eqlist0, eqlist4], "是", "等值"]], ["@@表达式传递"], [[eqlist3], "是", "等值"]])
            if counter[1] == 1 and counter[2] == 1:
                outjson.append([eqlist0, "是", "等值"])
                if self.treesig:
                    tripleobjlist.append([[[[eqlist1, eqlist2], "是", "等值"]], ["@@表达式传递"], [[eqlist0], "是", "等值"]])
            if counter[3] == 1 and counter[4] == 1:
                outjson.append([eqlist0, "是", "等值"])
                if self.treesig:
                    tripleobjlist.append([[[[eqlist3, eqlist4], "是", "等值"]], ["@@表达式传递"], [[eqlist0], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("express2relations")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def axiom2relation(self, oldsetobj):
        " 精确概念的自洽 "
        logger1.info("in axiom2relation")
        # 0. 空间定义
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # 0. 直线合并
        oldsetobj["直线集合"] = lines_deliver(oldsetobj["直线集合"])
        # 1. 遍历点，得到 线段 角 和 三角形
        oldsetobj = self.points_relations(oldsetobj)
        oldsetobj["直线集合"] = lines_deliver(oldsetobj["直线集合"])
        # 2. 遍历直线，得到补角
        space_ins._setobj = self.line2comple_relations(oldsetobj)
        space_ins._setobj["直线集合"] = lines_deliver(space_ins._setobj["直线集合"])
        oldsetobj = space_ins._setobj
        # 3. 遍历垂直，得到直角，直角三角形
        oldsetobj = self.vert2Rt_relations(oldsetobj)
        # self.list_set_deliver(space_ins._setobj["等值集合"], "等值集合")
        # 4. 平行间传递平行。线段是元素，多点直线作为多个元素处理。不同组间有重复的元素，则合并。
        # space_ins._setobj = self.parali2segm_relations(oldsetobj)
        oldsetobj = self.parali2segm_relations(oldsetobj)
        # self.list_set_deliver(space_ins._setobj["等值集合"], "等值集合")
        # 5. 平行传递垂直。
        oldsetobj = self.list_set_equalanti(oldsetobj, tarkey="平行", purposekey="垂直")
        # 集合缩并
        oldsetobj = self.listset_deliverall(oldsetobj)
        # 6. 平行夹平行相等。
        oldsetobj = self.parapara2equal(oldsetobj)
        # oldsetobj = self.listset_deliverall(oldsetobj)
        # 7. 遍历平行，对顶角, 同位角
        oldsetobj["直线集合"] = lines_deliver(oldsetobj["直线集合"])
        oldsetobj = self.corresangles2relations(oldsetobj)
        oldsetobj = self.listset_deliverall(oldsetobj)
        # 8. 所有直角 平角 导入 等值集合
        oldsetobj = self.addconstantequal(oldsetobj)
        # 9. 等值传递。不同组间有重复的元素，则合并。余角后面和直角三角形一起做
        # 钝角锐角 根据等值传递
        oldsetobj = self.equall2dunrui_relations(oldsetobj)
        oldsetobj = self.degree2angle_relations(oldsetobj)
        # print(sys.getsizeof(space_ins._setobj))
        # 11. 遍历直角三角形 垂直已导出角可以略过，得到余角
        oldsetobj = self.rttriang2remain_relations(oldsetobj)
        oldsetobj = self.listset_deliverall(oldsetobj)
        # 12. 表达式得出补角余角集合
        oldsetobj = self.express2compleremain_relations(oldsetobj)
        oldsetobj = self.express2relations(oldsetobj)
        # 13. 余角 反等传递
        oldsetobj = self.list_set_antiequal(oldsetobj, tarkey="余角", purposekey="等值")
        # 14. 补角 反等传递
        oldsetobj = self.list_set_antiequal(oldsetobj, tarkey="补角", purposekey="等值")
        # 15. 垂直 反等传递
        oldsetobj = self.list_set_antiequal(oldsetobj, tarkey="垂直", purposekey="平行")
        oldsetobj = self.listset_deliverall(oldsetobj)
        # 16. 非平行直线组成 角 的等值集合
        oldsetobj = self.lines2angle_equal(oldsetobj)
        # 17. 删除空的集合
        space_ins._setobj = self.list_set_shrink_all(oldsetobj)
        newsetobj = copy.deepcopy(space_ins._setobj)
        return newsetobj

    def list_set_shrink_all(self, oldsetobj):
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        for objset in basic_space_ins._setobj:
            if basic_space_ins._setobj[objset]["结构形式"] == "一级列表二级集合":
                oldsetobj[objset] = list_set_shrink(oldsetobj[objset])
        newsetobj = copy.deepcopy(oldsetobj)
        return newsetobj

    def square2elements(self, onesetobj):
        tripleobjlist = []
        outjson = []
        for obj in onesetobj:
            tname = obj.rstrip("}").lstrip("{正方形@")
            tlist = latex_fenci(latex2space(tname))
            tlist = [i1.replace(" ", "") for i1 in tlist]
            # 点
            for point in tlist:
                tname = self.language.name_symmetric(point).replace(" ", "")
                tname = "{点@" + tname + "}"
                outjson.append([tname, "是", "点"])
            # 线段
            tseglist = tlist + tlist[0:1]
            last4seg = []
            for idseg in range(4):
                tname = self.language.name_symmetric(" ".join(tseglist[idseg:idseg + 2])).replace(" ", "")
                tname = "{线段@" + tname + "}"
                last4seg.append(tname)
                outjson.append([tname, "是", "线段"])
                tpname1 = self.language.name_symmetric(" ".join(tseglist[idseg:idseg + 1])).replace(" ", "")
                tpname1 = "{点@" + tpname1 + "}"
                tpname2 = self.language.name_symmetric(" ".join(tseglist[idseg + 1:idseg + 2])).replace(" ", "")
                tpname2 = "{点@" + tpname2 + "}"
                outjson.append([[tpname1, tpname2], "是", "直线"])
            tripleobjlist.append(
                [[[[obj], "是", "正方形"]], ["@@正方形平行属性"], [[list(set([last4seg[-1], last4seg[-3]]))], "是", "平行"]])
            tripleobjlist.append(
                [[[[obj], "是", "正方形"]], ["@@正方形平行属性"], [[list(set([last4seg[-2], last4seg[-4]]))], "是", "平行"]])
            tripleobjlist.append(
                [[[[obj], "是", "正方形"]], ["@@正方形垂直属性"], [[list(set([last4seg[-1], last4seg[-2]]))], "是", "垂直"]])
            tripleobjlist.append(
                [[[[obj], "是", "正方形"]], ["@@正方形垂直属性"], [[list(set([last4seg[-2], last4seg[-3]]))], "是", "垂直"]])
            tripleobjlist.append(
                [[[[obj], "是", "正方形"]], ["@@正方形垂直属性"], [[list(set([last4seg[-3], last4seg[-4]]))], "是", "垂直"]])
            tripleobjlist.append(
                [[[[obj], "是", "正方形"]], ["@@正方形垂直属性"], [[list(set([last4seg[-4], last4seg[-1]]))], "是", "垂直"]])
            tripleobjlist.append([[[[obj], "是", "正方形"]], ["@@正方形等边属性"], [[list(set(last4seg))], "是", "等值"]])
            outjson.append([[last4seg[-1], last4seg[-3]], "是", "平行"])
            outjson.append([[last4seg[-2], last4seg[-4]], "是", "平行"])
            outjson.append([[last4seg[-1], last4seg[-2]], "是", "垂直"])
            outjson.append([[last4seg[-2], last4seg[-3]], "是", "垂直"])
            outjson.append([[last4seg[-3], last4seg[-4]], "是", "垂直"])
            outjson.append([[last4seg[-4], last4seg[-1]], "是", "垂直"])
            outjson.append([last4seg, "是", "等值"])

            # 角
            equangle = []
            tanglist = tlist + tlist[0:2]
            for idangle in range(4):
                tname = self.language.name_symmetric(" ".join(tanglist[idangle:idangle + 3])).replace(" ", "")
                tname = "{角@" + tname + "}"
                outjson.append([tname, "是", "角"])
                outjson.append([tname, "是", "直角"])
                tripleobjlist.append([[[[obj], "是", "正方形"]], ["@@正方形直角属性"], [[tname], "是", "直角"]])
                tname = self.language.name_symmetric(" ".join([tanglist[idangle + 1], tanglist[idangle],
                                                               tanglist[idangle + 2]])).replace(" ", "")
                tname = "{角@" + tname + "}"
                outjson.append([tname, "是", "角"])
                outjson.append([tname, "是", "锐角"])
                equangle.append(tname)
                tname = self.language.name_symmetric(" ".join([tanglist[idangle], tanglist[idangle + 2],
                                                               tanglist[idangle + 1]])).replace(" ", "")
                tname = "{角@" + tname + "}"
                outjson.append([tname, "是", "角"])
                outjson.append([tname, "是", "锐角"])
                equangle.append(tname)
                tname = self.language.name_cyc_one(" ".join(tanglist[idangle:idangle + 3])).replace(" ", "")
                tname = "{三角形@" + tname + "}"
                outjson.append([tname, "是", "三角形"])
                outjson.append([tname, "是", "直角三角形"])
                tripleobjlist.append([[[[obj], "是", "正方形"]], ["@@正方形直角属性"], [[tname], "是", "直角三角形"]])
            equangle = list(set(equangle))
            outjson.append([equangle, "是", "等值"])
            tripleobjlist.append([[[[obj], "是", "正方形"]], ["@@正方形等边属性"], [[equangle], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("square2elements")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def triangle2elements(self, onesetobj):
        outjson = []
        for obj in onesetobj:
            tname = obj.rstrip("}").lstrip("{三角形@")
            tlist = latex_fenci(latex2space(tname))
            tlist = [i1.replace(" ", "") for i1 in tlist]
            # 点
            for point in tlist:
                tname = self.language.name_symmetric(point).replace(" ", "")
                tname = "{点@" + tname + "}"
                outjson.append([tname, "是", "点"])
            # 线段
            tseglist = tlist + tlist[0:1]
            last4seg = []
            for idseg in range(3):
                tname = self.language.name_symmetric(" ".join(tseglist[idseg:idseg + 2])).replace(" ", "")
                tname = "{线段@" + tname + "}"
                last4seg.append(tname)
                outjson.append([tname, "是", "线段"])
                tpname1 = self.language.name_symmetric(" ".join(tseglist[idseg:idseg + 1])).replace(" ", "")
                tpname1 = "{点@" + tpname1 + "}"
                tpname2 = self.language.name_symmetric(" ".join(tseglist[idseg + 1:idseg + 2])).replace(" ", "")
                tpname2 = "{点@" + tpname2 + "}"
                outjson.append([[tpname1, tpname2], "是", "直线"])
            # 角
            tanglist = tlist + tlist[0:2]
            for idangle in range(3):
                tname = self.language.name_symmetric(" ".join(tanglist[idangle:idangle + 3])).replace(" ", "")
                tname = "{角@" + tname + "}"
                outjson.append([tname, "是", "角"])
        return self.math_solver_write(outjson)

    def simila_triangle2elements(self, onesetobj, equalsetobj):
        " 相似三角形必要条件 可以导出的 "
        tripleobjlist = []
        outjson = []
        for objlist in onesetobj:
            # 概念元素输入
            for obj in objlist:
                tname = obj.rstrip("}").lstrip("{三角形@")
                tlist = latex_fenci(latex2space(tname))
                tlist = [i1.replace(" ", "") for i1 in tlist]
                # 点
                for point in tlist:
                    tname = self.language.name_symmetric(point).replace(" ", "")
                    tname = "{点@" + tname + "}"
                    outjson.append([tname, "是", "点"])
                # 线段
                tseglist = tlist + tlist[0:1]
                last4seg = []
                for idseg in range(3):
                    tname = self.language.name_symmetric(" ".join(tseglist[idseg:idseg + 2])).replace(" ", "")
                    tname = "{线段@" + tname + "}"
                    last4seg.append(tname)
                    outjson.append([tname, "是", "线段"])
                    tpname1 = self.language.name_symmetric(" ".join(tseglist[idseg:idseg + 1])).replace(" ", "")
                    tpname1 = "{点@" + tpname1 + "}"
                    tpname2 = self.language.name_symmetric(" ".join(tseglist[idseg + 1:idseg + 2])).replace(" ", "")
                    tpname2 = "{点@" + tpname2 + "}"
                    outjson.append([[tpname1, tpname2], "是", "直线"])
                # 角
                tanglist = tlist + tlist[0:2]
                for idangle in range(3):
                    tname = self.language.name_symmetric(" ".join(tanglist[idangle:idangle + 3])).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "角"])
            # 特性提取
            angles_list = []
            triang_pointlist = [elems.rstrip("}").lstrip("{三角形@") for elems in objlist]
            triang_pointlist = [latex_fenci(latex2space(angli)) for angli in triang_pointlist]
            for onetriangle in triang_pointlist:
                for elem in onetriangle:
                    elelist = copy.deepcopy(onetriangle)
                    elelist.remove(elem)
                    elea = elelist[0]
                    eleb = elelist[1]
                    tname = self.language.name_symmetric(" ".join([elea, elem, eleb])).replace(" ", "")
                    tanle0 = "{角@" + tname + "}"
                    tname = self.language.name_symmetric(" ".join([elem, elea, eleb])).replace(" ", "")
                    tanle1 = "{角@" + tname + "}"
                    tname = self.language.name_symmetric(" ".join([elea, eleb, elem])).replace(" ", "")
                    tanle2 = "{角@" + tname + "}"
                    tname = self.language.name_cyc_one(" ".join(onetriangle)).replace(" ", "")
                    ttrian = "{三角形@" + tname + "}"
                    angles_list.append([tanle0, tanle1, tanle2, ttrian])
            comb_lenth = len(angles_list)
            for idmain in range(comb_lenth):
                for idcli in range(idmain + 1, comb_lenth):
                    outjson.append([[eae_list[idmain][0], eae_list[idcli][1]], "是", "等值"])
                    if self.treesig:
                        tklist = list(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                        tripleobjlist.append([[[[tklist], "是", "相似三角形"]], ["@@相似三角形必要条件"], [[tvlist], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("simila_triangle2elements")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def congruent_triangle2elements(self, onesetobj, equalsetobj):
        "全等三角形必要条件 可以导出的"
        tripleobjlist = []
        outjson = []
        dictest = {}
        for objlist in onesetobj:
            # 概念元素输入
            for obj in objlist:
                tname = obj.rstrip("}").lstrip("{三角形@")
                tlist = latex_fenci(latex2space(tname))
                tlist = [i1.replace(" ", "") for i1 in tlist]
                # 点
                for point in tlist:
                    tname = self.language.name_symmetric(point).replace(" ", "")
                    tname = "{点@" + tname + "}"
                    outjson.append([tname, "是", "点"])
                # 线段
                tseglist = tlist + tlist[0:1]
                last4seg = []
                for idseg in range(3):
                    tname = self.language.name_symmetric(" ".join(tseglist[idseg:idseg + 2])).replace(" ", "")
                    tname = "{线段@" + tname + "}"
                    last4seg.append(tname)
                    outjson.append([tname, "是", "线段"])
                    tpname1 = self.language.name_symmetric(" ".join(tseglist[idseg:idseg + 1])).replace(" ", "")
                    tpname1 = "{点@" + tpname1 + "}"
                    tpname2 = self.language.name_symmetric(" ".join(tseglist[idseg + 1:idseg + 2])).replace(" ", "")
                    tpname2 = "{点@" + tpname2 + "}"
                    outjson.append([[tpname1, tpname2], "是", "直线"])
                # 角
                tanglist = tlist + tlist[0:2]
                for idangle in range(3):
                    tname = self.language.name_symmetric(" ".join(tanglist[idangle:idangle + 3])).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "角"])
            # 特性提取
            eae_list = []
            aea_list = []
            triang_pointlist = [elems.rstrip("}").lstrip("{三角形@") for elems in objlist]
            triang_pointlist = [latex_fenci(latex2space(angli)) for angli in triang_pointlist]
            for onetriangle in triang_pointlist:
                for elem in onetriangle:
                    elelist = copy.deepcopy(onetriangle)
                    elelist.remove(elem)
                    elea = elelist[0]
                    eleb = elelist[1]
                    tname = self.language.name_symmetric(" ".join([elea, elem, eleb])).replace(" ", "")
                    tanle0 = "{角@" + tname + "}"
                    tname = self.language.name_symmetric(" ".join([elea, elem])).replace(" ", "")
                    tseg1 = "{线段@" + tname + "}"
                    tname = self.language.name_symmetric(" ".join([elem, eleb])).replace(" ", "")
                    tseg2 = "{线段@" + tname + "}"
                    tname = self.language.name_symmetric(" ".join([elem, elea, eleb])).replace(" ", "")
                    tanle1 = "{角@" + tname + "}"
                    tname = self.language.name_symmetric(" ".join([elea, eleb, elem])).replace(" ", "")
                    tanle2 = "{角@" + tname + "}"
                    tname = self.language.name_symmetric(" ".join([elea, eleb])).replace(" ", "")
                    tseg0 = "{线段@" + tname + "}"
                    tname = self.language.name_cyc_one(" ".join(onetriangle)).replace(" ", "")
                    ttrian = "{三角形@" + tname + "}"
                    eae_list.append([tseg1, tseg2, tanle0, ttrian])
                    aea_list.append([tseg0, tanle1, tanle2, ttrian])
            comb_lenth = len(aea_list)
            for idmain in range(comb_lenth):
                for idcli in range(idmain + 1, comb_lenth):
                    if aea_list[idmain][-1] == aea_list[idcli][-1]:
                        continue
                    # 会根据每个三角形的3元素 遍历3遍
                    #  判断边
                    aea_sig = [0, 0, 0, 0, 0]
                    eae_sig = [0, 0, 0, 0, 0]
                    for equset in equalsetobj:
                        if aea_list[idmain][-1] != aea_list[idcli][-1]:
                            # 角边角
                            judgequllist = [aea_list[idmain][0], aea_list[idcli][0]]
                            if set(judgequllist).issubset(equset):
                                aea_sig[0] = 1
                            judgequllist = [aea_list[idmain][1], aea_list[idcli][1]]
                            if set(judgequllist).issubset(equset):
                                aea_sig[1] = 1
                            judgequllist = [aea_list[idmain][1], aea_list[idcli][2]]
                            if set(judgequllist).issubset(equset):
                                aea_sig[2] = 1
                            judgequllist = [aea_list[idmain][2], aea_list[idcli][1]]
                            if set(judgequllist).issubset(equset):
                                aea_sig[3] = 1
                            judgequllist = [aea_list[idmain][2], aea_list[idcli][2]]
                            if set(judgequllist).issubset(equset):
                                aea_sig[4] = 1
                            if aea_sig[0] == 1 and (aea_sig[1] + aea_sig[4] == 2 or aea_sig[2] + aea_sig[3] == 2):
                                # outjson 可以只写 aea 之外的等值关系。tripleobjlist 需要全量写
                                outjson.append([[eae_list[idmain][2], eae_list[idcli][2]], "是", "等值"])
                                if self.treesig:
                                    tklist = list(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                    tkstr = "".join(tklist)
                                    if tkstr not in dictest:
                                        dictest[tkstr] = []
                                    tvlist = list(set([eae_list[idmain][2], eae_list[idcli][2]]))
                                    tvstr = "".join(tvlist)
                                    if tvstr not in dictest[tkstr]:
                                        dictest[tkstr].append(tvstr)
                                        tripleobjlist.append(
                                            [[[[tklist], "是", "全等三角形"]],
                                             ["@@全等三角形必要条件"],
                                             [[tvlist], "是", "等值"]])
                                    tvlist = list(set([aea_list[idmain][0], aea_list[idcli][0]]))
                                    tvstr = "".join(tvlist)
                                    if tvstr not in dictest[tkstr]:
                                        dictest[tkstr].append(tvstr)
                                        tripleobjlist.append(
                                            [[[[tklist], "是", "全等三角形"]],
                                             ["@@全等三角形必要条件"],
                                             [[tvlist], "是", "等值"]])
                                if aea_sig[1] == 1 and aea_sig[4] == 1:
                                    outjson.append([[eae_list[idmain][1], eae_list[idcli][1]], "是", "等值"])
                                    outjson.append([[eae_list[idmain][0], eae_list[idcli][0]], "是", "等值"])
                                    if self.treesig:
                                        tklist = list(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                        tkstr = "".join(tklist)
                                        if tkstr not in dictest:
                                            dictest[tkstr] = []
                                        tvlist = list(set([aea_list[idmain][1], aea_list[idcli][1]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([eae_list[idmain][1], eae_list[idcli][1]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([eae_list[idmain][0], eae_list[idcli][0]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([aea_list[idmain][2], aea_list[idcli][2]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                elif aea_sig[2] == 1 and aea_sig[3] == 1:
                                    outjson.append([[eae_list[idmain][1], eae_list[idcli][0]], "是", "等值"])
                                    outjson.append([[eae_list[idmain][0], eae_list[idcli][1]], "是", "等值"])
                                    if self.treesig:
                                        tklist = list(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                        tkstr = "".join(tklist)
                                        if tkstr not in dictest:
                                            dictest[tkstr] = []
                                        tvlist = list(set([eae_list[idmain][1], eae_list[idcli][0]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([aea_list[idmain][1], aea_list[idcli][2]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([eae_list[idmain][0], eae_list[idcli][1]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([aea_list[idmain][2], aea_list[idcli][1]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                            # 边角边
                            judgequllist = [eae_list[idmain][0], eae_list[idcli][0]]
                            if set(judgequllist).issubset(equset):
                                eae_sig[0] = 1
                            judgequllist = [eae_list[idmain][0], eae_list[idcli][1]]
                            if set(judgequllist).issubset(equset):
                                eae_sig[1] = 1
                            judgequllist = [eae_list[idmain][1], eae_list[idcli][0]]
                            if set(judgequllist).issubset(equset):
                                eae_sig[2] = 1
                            judgequllist = [eae_list[idmain][1], eae_list[idcli][1]]
                            if set(judgequllist).issubset(equset):
                                eae_sig[3] = 1
                            judgequllist = [eae_list[idmain][2], eae_list[idcli][2]]
                            if set(judgequllist).issubset(equset):
                                eae_sig[4] = 1
                            if eae_sig[4] == 1 and (eae_sig[0] + eae_sig[3] == 2 or eae_sig[1] + eae_sig[2] == 2):
                                # outjson 可以只写 eae 之外的等值关系。tripleobjlist 需要全量写
                                if eae_sig[0] == 1 and eae_sig[3] == 1:
                                    outjson.append([[aea_list[idmain][1], aea_list[idcli][1]], "是", "等值"])
                                    outjson.append([[aea_list[idmain][2], aea_list[idcli][2]], "是", "等值"])
                                    if self.treesig:
                                        tklist = list(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                        tkstr = "".join(tklist)
                                        if tkstr not in dictest:
                                            dictest[tkstr] = []
                                        tvlist = list(set([aea_list[idmain][1], aea_list[idcli][1]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([eae_list[idmain][1], eae_list[idcli][1]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([aea_list[idmain][2], aea_list[idcli][2]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([eae_list[idmain][0], eae_list[idcli][0]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                elif eae_sig[1] == 1 and eae_sig[2] == 1:
                                    outjson.append([[aea_list[idmain][1], aea_list[idcli][2]], "是", "等值"])
                                    outjson.append([[aea_list[idmain][2], aea_list[idcli][1]], "是", "等值"])
                                    if self.treesig:
                                        tklist = list(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                        tkstr = "".join(tklist)
                                        if tkstr not in dictest:
                                            dictest[tkstr] = []
                                        tvlist = list(set([aea_list[idmain][1], aea_list[idcli][2]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([eae_list[idmain][1], eae_list[idcli][0]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([aea_list[idmain][2], aea_list[idcli][1]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                        tvlist = list(set([eae_list[idmain][0], eae_list[idcli][1]]))
                                        tvstr = "".join(tvlist)
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[tklist], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[tvlist], "是", "等值"]])
                                outjson.append([[aea_list[idmain][0], aea_list[idcli][0]], "是", "等值"])
                                if self.treesig:
                                    tklist = list(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                    tkstr = "".join(tklist)
                                    if tkstr not in dictest:
                                        dictest[tkstr] = []
                                    tvlist = list(set([aea_list[idmain][0], aea_list[idcli][0]]))
                                    tvstr = "".join(tvlist)
                                    if tvstr not in dictest[tkstr]:
                                        dictest[tkstr].append(tvstr)
                                        tripleobjlist.append(
                                            [[[[tklist], "是", "全等三角形"]],
                                             ["@@全等三角形必要条件"],
                                             [[tvlist], "是", "等值"]])
                                    tvlist = list(set([eae_list[idmain][2], eae_list[idcli][2]]))
                                    tvstr = "".join(tvlist)
                                    if tvstr not in dictest[tkstr]:
                                        dictest[tkstr].append(tvstr)
                                        tripleobjlist.append(
                                            [[[[tklist], "是", "全等三角形"]],
                                             ["@@全等三角形必要条件"],
                                             [[tvlist], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("congruent_triangle2elements")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def isosceles_triangle2elements(self, onesetobj, equalsetobj):
        "等腰三角形必要条件 可以导出的 衍生 元素 "
        logger1.info("in isosceles_triangle2elements")
        # 1. 得出 等腰或等边三角形
        onesetobj = list(onesetobj)
        triang_pointlist = [elems.rstrip("}").lstrip("{三角形@") for elems in onesetobj]
        triang_pointlist = [latex_fenci(latex2space(angli)) for angli in triang_pointlist]
        tripleobjlist = []
        outjson = []
        for idn, onetriangle in enumerate(triang_pointlist):
            point1, point2, point3 = onetriangle
            tname = self.language.name_symmetric(" ".join([point1, point2, point3])).replace(" ", "")
            tanle2 = "{角@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point3, point1, point2])).replace(" ", "")
            tanle1 = "{角@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point2, point3, point1])).replace(" ", "")
            tanle3 = "{角@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point1, point3])).replace(" ", "")
            tseg2 = "{线段@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point2, point3])).replace(" ", "")
            tseg1 = "{线段@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point1, point2])).replace(" ", "")
            tseg3 = "{线段@" + tname + "}"
            for equset in equalsetobj:
                edgeset1 = set([tseg3, tseg2])
                edgeset2 = set([tseg3, tseg1])
                edgeset3 = set([tseg2, tseg1])
                anglset1 = set([tanle2, tanle3])
                anglset2 = set([tanle1, tanle3])
                anglset3 = set([tanle1, tanle2])
                # print(onesetobj[idn], edgeset1, edgeset2, edgeset3, anglset1, anglset2, anglset3)
                if len(edgeset1.intersection(equset)) == 2:
                    outjson.append([list(anglset1), "是", "等值"])
                    if self.treesig:
                        # print(onesetobj[idn], outjson[-1])
                        tripleobjlist.append([[[[onesetobj[idn]], "是", "等腰三角形"]], ["@@等腰三角形必要条件角"],
                                              [[list(anglset1)], "是", "等值"]])
                if len(edgeset2.intersection(equset)) == 2:
                    outjson.append([list(anglset2), "是", "等值"])
                    if self.treesig:
                        # print(onesetobj[idn], outjson[-1])
                        tripleobjlist.append([[[[onesetobj[idn]], "是", "等腰三角形"]], ["@@等腰三角形必要条件角"],
                                              [[list(anglset2)], "是", "等值"]])
                if len(edgeset3.intersection(equset)) == 2:
                    outjson.append([list(anglset3), "是", "等值"])
                    if self.treesig:
                        # print(onesetobj[idn], outjson[-1])
                        tripleobjlist.append([[[[onesetobj[idn]], "是", "等腰三角形"]], ["@@等腰三角形必要条件角"],
                                              [[list(anglset3)], "是", "等值"]])
                if len(anglset1.intersection(equset)) == 2:
                    outjson.append([list(edgeset1), "是", "等值"])
                    if self.treesig:
                        tripleobjlist.append([[[[onesetobj[idn]], "是", "等腰三角形"]], ["@@等腰三角形必要条件边"],
                                              [[list(edgeset1)], "是", "等值"]])
                if len(anglset2.intersection(equset)) == 2:
                    outjson.append([list(edgeset2), "是", "等值"])
                    if self.treesig:
                        tripleobjlist.append([[[[onesetobj[idn]], "是", "等腰三角形"]], ["@@等腰三角形必要条件边"],
                                              [[list(edgeset2)], "是", "等值"]])
                if len(anglset3.intersection(equset)) == 2:
                    outjson.append([list(edgeset3), "是", "等值"])
                    if self.treesig:
                        tripleobjlist.append([[[[onesetobj[idn]], "是", "等腰三角形"]], ["@@等腰三角形必要条件边"],
                                              [[list(edgeset3)], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("congruent_triangle2elements")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def circle2elements(self, oldsetobj):
        " 圆的必要条件 可以导出的 衍生 元素 "
        logger1.info("in circle2elements")
        # 1. 得出 圆 的相关属性
        onesetobj = list(oldsetobj["圆集合"])
        tanangobj = list(oldsetobj["弦切角集合"])
        cir_pointlist = [elems.rstrip("}").lstrip("{圆@") for elems in onesetobj]
        cir_pointlist = [latex_fenci(latex2space(angli)) for angli in cir_pointlist]
        tripleobjlist = []
        outjson = []
        # 2. 遍历每个圆
        for idn, onecir in enumerate(cir_pointlist):
            lenthp = len(onecir) - 1
            tnlist = onecir[1:] * 2
            # 3.1. 直径直角性质
            for one_diam in oldsetobj["直径集合"]:
                onobj = latex_fenci(latex2space(one_diam.lstrip("{直径@").rstrip("}")))
                if onobj[1] != onecir[0]:
                    break
                sidpoint = [point for point in onecir if point not in onobj]
                for pooint in sidpoint:
                    tname = self.language.name_symmetric(" ".join([onobj[0], pooint, onobj[1]])).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "直角"])
                    if self.treesig:
                        tripleobjlist.append([[[[one_diam], "是", "直径"]], ["@@直径的性质"], [[tname], "是", "直角"]])
            # 3.2 找出弦切角 所有跟该圆相关的 角
            arclist = []
            anglist = []
            for paircir in tanangobj:
                arc_now = paircir[0].rstrip("}").lstrip("{弧@")
                arc_now = latex_fenci(latex2space(arc_now))
                if arc_now[0] == onecir[0]:
                    arclist.append(paircir[0])
                    ang_now = paircir[1].rstrip("}").lstrip("{角@")
                    ang_now = latex_fenci(latex2space(ang_now))
                    anglist.append(ang_now)
            # 3.3 等弧点循环 n 点 2点弧 ~ n-1 点弧
            for inum in range(2, lenthp):
                # 3.3.1 等点弧 组合循环 n 个
                for iindex in range(lenthp):
                    tname = self.language.name_symmetric(" ".join(tnlist[iindex:iindex + inum])).replace(" ", "")
                    arcname = "{弧@" + onecir[0] + tname + "}"
                    outjson.append([arcname, "是", "弧"])
                    # 3.3.2 弦切角获取
                    anglts = []
                    if arcname in arclist:
                        tind = arclist.index(arcname)
                        angpps = anglist[tind]
                        linet1 = []
                        linet2 = []
                        # print(oldsetobj["直线集合"])
                        # print(angpps)
                        for tline in oldsetobj["直线集合"]:
                            tline = [elem.lstrip("{点@").rstrip("}") for elem in tline]
                            if set(tline).issuperset(set(angpps[:2])):
                                linet1 = tline
                            if set(tline).issuperset(set(angpps[1:])):
                                linet2 = tline
                        if linet1 == [] or linet2 == []:
                            print(linet1, linet2)
                            raise Exception("没有需要匹配直线")
                        indt1 = linet1.index(angpps[0])
                        indtc1 = linet1.index(angpps[1])
                        indtc2 = linet2.index(angpps[1])
                        indt2 = linet2.index(angpps[2])
                        linet1 = linet1[indtc1 + 1:] if indt1 > indtc1 else linet1[:indtc1]
                        linet2 = linet2[indtc2 + 1:] if indt2 > indtc2 else linet2[:indtc2]
                        for tp1 in linet1:
                            for tp2 in linet2:
                                tname = self.language.name_symmetric(" ".join([tp1, angpps[1], tp2])).replace(" ", "")
                                tname = "{角@" + tname + "}"
                                anglts.append(tname)
                    # 3.3.3 单弧 角 循环 n - inum
                    leavepoints = [point for point in onecir[1:] if point not in tnlist[iindex:iindex + inum]]
                    tequlist = set()
                    # 圆心角
                    tname = self.language.name_symmetric(
                        " ".join([tnlist[iindex], onecir[0], tnlist[iindex + inum - 1]])).replace(" ", "")
                    canglname = "{角@" + tname + "}"
                    for angpoint in leavepoints:
                        # 3.3.4 圆周角
                        tname = self.language.name_symmetric(
                            " ".join([tnlist[iindex], angpoint, tnlist[iindex + inum - 1]])).replace(" ", "")
                        anglname = "{角@" + tname + "}"
                        outjson.append([anglname, "是", "角"])
                        tequlist.add(anglname)
                        bei_expangstr = canglname + " = 2 * " + anglname
                        outjson.append([bei_expangstr, "是", "表达式"])
                        if anglts != []:
                            fnlist = anglts + [anglname]
                            outjson.append([fnlist, "是", "等值"])
                            for tang in anglts:
                                if self.treesig:
                                    tripleobjlist.append(
                                        [[[[onesetobj[idn]], "是", "圆"]], ["@@弦切角性质"],
                                         [[list(set([tang, anglname]))], "是", "等值"]])
                        if self.treesig:
                            tripleobjlist.append(
                                [[[[onesetobj[idn]], "是", "圆"]], ["@@圆心角圆周角关系"], [[bei_expangstr], "是", "表达式"]])
                    tequlist = list(tequlist)
                    outjson.append([tequlist, "是", "等值"])
                    if self.treesig:
                        tripleobjlist.append(
                            [[[[onesetobj[idn]], "是", "圆"]], ["@@圆等弧对等角"], [[tequlist], "是", "等值"]])
            # 跨点等弧 循环
            for iarc in range(lenthp - 2):  # 角点循环
                for arc in range(lenthp):
                    angpli = [point for point in onecir[1:] if point not in tnlist[arc:arc + iarc + 2]]
                    tname = self.language.name_symmetric(" ".join(tnlist[arc:arc + iarc + 2])).replace(" ", "")
                    arcname = "{弧@" + onecir[0] + tname + "}"
                    outjson.append([arcname, "是", "弧"])
                    # 遍历弧内点的表达式
                    if iarc > 0:
                        if lenthp > 3:
                            # 2. 圆内接四边形的性质
                            for inpoint in range(iarc):
                                tinname = self.language.name_symmetric(
                                    " ".join([tnlist[arc], tnlist[arc + inpoint + 1], tnlist[arc + iarc + 1]])).replace(
                                    " ", "")
                                tinname = "{角@" + tinname + "}"
                                for ap in angpli:
                                    toutname = self.language.name_symmetric(
                                        " ".join([tnlist[arc], ap, tnlist[arc + iarc + 1]])).replace(" ", "")
                                    toutname = "{角@" + toutname + "}"
                                    toutlist = list(set([tinname, toutname]))
                                    outjson.append([toutlist, "是", "补角"])
                                    if self.treesig:
                                        tripleobjlist.append(
                                            [[[[onesetobj[idn]], "是", "圆"]], ["@@圆内接四边形的性质"], [[toutlist], "是", "补角"]])
                        exparcstr = arcname + " = "
                        tarclist = []
                        for ipoint in range(iarc):
                            tname = self.language.name_symmetric(
                                " ".join([tnlist[arc + ipoint], tnlist[arc + ipoint + 1]])).replace(" ", "")
                            tarclist.append("{弧@" + tname + "}")
                        tname = self.language.name_symmetric(
                            " ".join([tnlist[arc + iarc], tnlist[arc + iarc + 1]])).replace(" ", "")
                        tarclist.append("{弧@" + tname + "}")
                        exparcstr += " + ".join(tarclist)
                        outjson.append([exparcstr, "是", "表达式"])
                        if self.treesig:
                            tripleobjlist.append(
                                [[[[onesetobj[idn]], "是", "圆"]], ["@@圆弧关系"], [[exparcstr], "是", "表达式"]])
                    # 圆心
                    tname = self.language.name_symmetric(
                        " ".join([tnlist[arc], onecir[0], tnlist[arc + iarc + 1]])).replace(" ", "")
                    canglname = "{角@" + tname + "}"
                    outjson.append([canglname, "是", "角"])
                    tcname = self.language.name_cyc_one(
                        " ".join([tnlist[arc], onecir[0], tnlist[arc + iarc + 1]])).replace(" ", "")
                    ctrianglname = "{三角形@" + tcname + "}"
                    outjson.append([ctrianglname, "是", "三角形"])
                    # tequlist = set()
                    for ap in angpli:
                        tname = self.language.name_symmetric(
                            " ".join([tnlist[arc], ap, tnlist[arc + iarc + 1]])).replace(" ", "")
                        anglname = "{角@" + tname + "}"
                        # outjson.append([anglname, "是", "角"])
                        # tequlist.add(anglname)
                        tname = self.language.name_cyc_one(" ".join([tnlist[arc], ap, tnlist[arc + iarc + 1]])).replace(
                            " ", "")
                        trianglname = "{三角形@" + tname + "}"
                        outjson.append([trianglname, "是", "三角形"])
                        # 遍历弧外点的表达式
                        if iarc > 0:
                            expangstr = anglname + " = "
                            cexpangstr = canglname + " = "
                            tanlist = []
                            ctanlist = []
                            for ipoint in range(iarc):
                                tname = self.language.name_symmetric(
                                    " ".join([tnlist[arc + ipoint], ap, tnlist[arc + ipoint + 1]])).replace(" ", "")
                                tanlist.append("{角@" + tname + "}")
                                tnamec = self.language.name_symmetric(
                                    " ".join([tnlist[arc + ipoint], onecir[0], tnlist[arc + ipoint + 1]])).replace(" ",
                                                                                                                   "")
                                ctanlist.append("{角@" + tnamec + "}")
                            tname = self.language.name_symmetric(
                                " ".join([tnlist[arc + iarc], ap, tnlist[arc + iarc + 1]])).replace(" ", "")
                            tanlist.append("{角@" + tname + "}")
                            # 圆周角大角 = 圆周分角的和
                            expangstr += " + ".join(tanlist)
                            outjson.append([expangstr, "是", "表达式"])
                            # 圆心角大角 = 圆心分角的和
                            cexpangstr += " + ".join(ctanlist)
                            outjson.append([cexpangstr, "是", "表达式"])
                            bei_expangstr = canglname + " = 2 * " + tanlist[-1]
                            outjson.append([bei_expangstr, "是", "表达式"])
                            if self.treesig:
                                tripleobjlist.append(
                                    [[[[onesetobj[idn]], "是", "圆"]], ["@@圆周角求和关系"], [[expangstr], "是", "表达式"]])
                                tripleobjlist.append(
                                    [[[[onesetobj[idn]], "是", "圆"]], ["@@圆心角求和关系"], [[cexpangstr], "是", "表达式"]])
                                # tripleobjlist.append(
                                #     [[[[onesetobj[idn]], "是", "圆"]], ["@@圆心角圆周角关系"], [[bei_expangstr], "是", "表达式"]])
                                # tequlist = list(tequlist)
                                # outjson.append([tequlist, "是", "等值"])
                                # if self.treesig:
                                #     tripleobjlist.append([[[[onesetobj[idn]], "是", "圆"]], ["@@圆等弧对等角"], [[tequlist], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("circle2elements")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def conception2element(self, oldsetobj):
        " 根据概念属性 衍生，点 线段 角 三角形，去掉顺序差异，再根据直线 衍生等值角 "
        logger1.info("in conception2element")
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        for oneset in oldsetobj:
            if oneset == "正方形集合":
                self.square2elements(oldsetobj[oneset])
            if oneset == "三角形集合":
                self.triangle2elements(oldsetobj[oneset])
            if oneset == "全等三角形集合":
                self.congruent_triangle2elements(oldsetobj[oneset], oldsetobj["等值集合"])
            if oneset == "相似三角形集合":
                self.simila_triangle2elements(oldsetobj[oneset], oldsetobj["等值集合"])
            if oneset == "等腰三角形集合":
                self.isosceles_triangle2elements(oldsetobj[oneset], oldsetobj["等值集合"])
            if oneset == "圆集合":
                self.circle2elements(oldsetobj)
        return space_ins._setobj

    def element2conception(self, oldsetobj):
        " 元素衍生概念 "
        logger1.info("in element2conception")
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # 1. 得出 全等三角形
        for oneset in oldsetobj:
            if oneset == "正方形集合":
                pass
            if oneset == "三角形集合":
                pass
            if oneset == "全等三角形集合":
                self.elements2congruent_triangle(oldsetobj["三角形集合"], oldsetobj["等值集合"])
            if oneset == "相似三角形集合":
                self.elements2similar_triangle(oldsetobj["三角形集合"], oldsetobj["等值集合"])
            if oneset == "等腰三角形集合":
                self.elements2isosceles_triangle(oldsetobj["三角形集合"], oldsetobj["等值集合"])
        space_ins._setobj = self.listset_deliverall(space_ins._setobj)
        return space_ins._setobj

    def elements2isosceles_triangle(self, onesetobj, equalsetobj):
        "元素衍生等腰三角形"
        logger1.info("in elements2isosceles_triangle")
        # 1. 得出 等腰或等边三角形
        onesetobj = list(onesetobj)
        triang_pointlist = [elems.rstrip("}").lstrip("{三角形@") for elems in onesetobj]
        triang_pointlist = [latex_fenci(latex2space(angli)) for angli in triang_pointlist]
        tripleobjlist = []
        outjson = []
        for idn, onetriangle in enumerate(triang_pointlist):
            point1, point2, point3 = onetriangle
            tname = self.language.name_symmetric(" ".join([point1, point2, point3])).replace(" ", "")
            tanle2 = "{角@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point3, point1, point2])).replace(" ", "")
            tanle1 = "{角@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point2, point3, point1])).replace(" ", "")
            tanle3 = "{角@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point1, point3])).replace(" ", "")
            tseg2 = "{线段@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point2, point3])).replace(" ", "")
            tseg1 = "{线段@" + tname + "}"
            tname = self.language.name_symmetric(" ".join([point1, point2])).replace(" ", "")
            tseg3 = "{线段@" + tname + "}"
            for equset in equalsetobj:
                edgeset = set([tseg3, tseg2, tseg1])
                anglset = set([tanle1, tanle2, tanle3])
                sameedge = edgeset.intersection(equset)
                sameangl = anglset.intersection(equset)
                if len(sameedge) > 1 or len(sameangl) > 1:
                    outjson.append([onesetobj[idn], "是", "等腰三角形"])
                    if len(sameedge) > 1:
                        if self.treesig:
                            tripleobjlist.append([[[[list(sameedge)], "是", "等值"], [[onesetobj[idn]], "是", "三角形"]],
                                                  ["@@等腰三角形充分条件边"], [[onesetobj[idn]], "是", "等腰三角形"]])
                    if len(sameangl) > 1:
                        if self.treesig:
                            tripleobjlist.append([[[[list(sameangl)], "是", "等值"], [[onesetobj[idn]], "是", "三角形"]],
                                                  ["@@等腰三角形充分条件角"], [[onesetobj[idn]], "是", "等腰三角形"]])
                if len(sameedge) == 3 or len(sameangl) == 3:
                    outjson.append([onesetobj[idn], "是", "等边三角形"])
                    if len(sameedge) == 3:
                        if self.treesig:
                            tripleobjlist.append([[[[list(edgeset)], "是", "等值"], [[onesetobj[idn]], "是", "三角形"]],
                                                  ["@@等边三角形充分条件边"], [[onesetobj[idn]], "是", "等边三角形"]])
                    if len(sameangl) == 3:
                        if self.treesig:
                            tripleobjlist.append([[[[list(anglset)], "是", "等值"], [[onesetobj[idn]], "是", "三角形"]],
                                                  ["@@等边三角形充分条件角"], [[onesetobj[idn]], "是", "等边三角形"]])
        if self.treesig:
            if self.debugsig:
                print("elements2isosceles_triangle")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def elements2similar_triangle(self, onesetobj, equalsetobj):
        " 元素衍生相似三角形 "
        logger1.info("in elements2similar_triangle")
        # 1. 得出 相似三角形
        triang_pointlist = [elems.rstrip("}").lstrip("{三角形@") for elems in onesetobj]
        triang_pointlist = [latex_fenci(latex2space(angli)) for angli in triang_pointlist]
        tripleobjlist = []
        outjson = []
        angles_list = []
        for onetriangle in triang_pointlist:
            elelist = onetriangle * 2
            tname = self.language.name_cyc_one(" ".join(onetriangle)).replace(" ", "")
            ttrian = "{三角形@" + tname + "}"
            tname = self.language.name_symmetric(" ".join(elelist[0:3])).replace(" ", "")
            tanle0 = "{角@" + tname + "}"
            tname = self.language.name_symmetric(" ".join(elelist[1:4])).replace(" ", "")
            tanle1 = "{角@" + tname + "}"
            tname = self.language.name_symmetric(" ".join(elelist[2:5])).replace(" ", "")
            tanle2 = "{角@" + tname + "}"
            angles_list.append([tanle0, tanle1, tanle2, ttrian])
        comb_lenth = len(angles_list)
        for idmain in range(comb_lenth - 1, 0, -1):
            for idcli in range(idmain - 1, -1, -1):
                tmlist = copy.deepcopy(angles_list[idmain])
                tclist = copy.deepcopy(angles_list[idcli])
                tequal = []
                for equset in equalsetobj:
                    findm = set(tmlist).intersection(equset)
                    findc = set(tclist).intersection(equset)
                    if len(findm) > 0 and len(findc) > 0:
                        for antgl in findm:
                            tmlist.remove(antgl)
                        for antgl in findc:
                            tclist.remove(antgl)
                        tequal = [[list(findm | findc), "是", "等值"]]
                if len(tclist) < 3 and len(tmlist) < 3:
                    tsetl = list(set([angles_list[idmain][-1], angles_list[idcli][-1]]))
                    outjson.append([tsetl, "是", "相似三角形"])
                    if self.treesig:
                        tripleobjlist.append([tequal, ["@@相似三角形充分条件"], [[tsetl], "是", "相似三角形"]])
        if self.treesig:
            if self.debugsig:
                print("elements2similar_triangle")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def elements2congruent_triangle(self, onesetobj, equalsetobj):
        " 元素衍生全等三角形 "
        logger1.info("in elements2congruent_triangle")
        # 1. 得出 全等三角形
        triang_pointlist = [elems.rstrip("}").lstrip("{三角形@") for elems in onesetobj]
        triang_pointlist = [latex_fenci(latex2space(angli)) for angli in triang_pointlist]
        # 1.1 边角边
        tripleobjlist = []
        outjson = []
        eae_list = []
        aea_list = []
        eee_list = []
        dictest = {}
        for onetriangle in triang_pointlist:
            for elem in onetriangle:
                elelist = copy.deepcopy(onetriangle)
                elelist.remove(elem)
                elea = elelist[0]
                eleb = elelist[1]
                tname = self.language.name_symmetric(" ".join([elea, elem, eleb])).replace(" ", "")
                tanle0 = "{角@" + tname + "}"
                tname = self.language.name_symmetric(" ".join([elea, elem])).replace(" ", "")
                tseg1 = "{线段@" + tname + "}"
                tname = self.language.name_symmetric(" ".join([elem, eleb])).replace(" ", "")
                tseg2 = "{线段@" + tname + "}"
                tname = self.language.name_symmetric(" ".join([elem, elea, eleb])).replace(" ", "")
                tanle1 = "{角@" + tname + "}"
                tname = self.language.name_symmetric(" ".join([elea, eleb, elem])).replace(" ", "")
                tanle2 = "{角@" + tname + "}"
                tname = self.language.name_symmetric(" ".join([elea, eleb])).replace(" ", "")
                tseg0 = "{线段@" + tname + "}"
                tname = self.language.name_cyc_one(" ".join(onetriangle)).replace(" ", "")
                ttrian = "{三角形@" + tname + "}"
                eae_list.append([tseg1, tseg2, tanle0, ttrian])
                aea_list.append([tseg0, tanle1, tanle2, ttrian])
                eee_list.append([tseg0, tseg1, tseg2, ttrian])
        comb_lenth = len(aea_list)
        for idmain in range(comb_lenth - 1, 0, -1):
            for idcli in range(idmain - 1, -1, -1):
                # 判断边
                aea_sig = [0, 0, 0, 0, 0]
                eae_sig = [0, 0, 0, 0, 0]
                eee_sig = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 00 11 22 01 10 02 20 12 21
                for equset in equalsetobj:
                    if aea_list[idmain][-1] != aea_list[idcli][-1]:
                        # 角边角
                        judgequllist = [aea_list[idmain][0], aea_list[idcli][0]]
                        if set(judgequllist).issubset(equset):
                            aea_sig[0] = 1
                        judgequllist = [aea_list[idmain][1], aea_list[idcli][1]]
                        if set(judgequllist).issubset(equset):
                            aea_sig[1] = 1
                        judgequllist = [aea_list[idmain][1], aea_list[idcli][2]]
                        if set(judgequllist).issubset(equset):
                            aea_sig[2] = 1
                        judgequllist = [aea_list[idmain][2], aea_list[idcli][1]]
                        if set(judgequllist).issubset(equset):
                            aea_sig[3] = 1
                        judgequllist = [aea_list[idmain][2], aea_list[idcli][2]]
                        if set(judgequllist).issubset(equset):
                            aea_sig[4] = 1
                        if aea_sig[0] == 1 and (aea_sig[1] + aea_sig[4] == 2 or aea_sig[2] + aea_sig[3] == 2):
                            outjson.append([[aea_list[idmain][-1], aea_list[idcli][-1]], "是", "全等三角形"])
                            if self.treesig:
                                t0list = list(set([aea_list[idmain][0], aea_list[idcli][0]]))
                                keyelem = ["".join(t0list)]
                                taea_equal = []
                                taea_equal.append([[t0list], "是", "等值"])
                                if aea_sig[1] + aea_sig[4] == 2:
                                    t1list = list(set([aea_list[idmain][1], aea_list[idcli][1]]))
                                    t2list = list(set([aea_list[idmain][2], aea_list[idcli][2]]))
                                    taea_equal.append([[t1list], "是", "等值"])
                                    taea_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t2list))
                                    keyelem.append("".join(t1list))
                                if aea_sig[2] + aea_sig[3] == 2:
                                    t1list = list(set([aea_list[idmain][1], aea_list[idcli][2]]))
                                    t2list = list(set([aea_list[idmain][2], aea_list[idcli][1]]))
                                    taea_equal.append([[t1list], "是", "等值"])
                                    taea_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t2list))
                                    keyelem.append("".join(t1list))
                                tkstr = "".join(set(keyelem))
                                if tkstr not in dictest:
                                    dictest[tkstr] = []
                                tvstr = "".join([aea_list[idmain][-1], aea_list[idcli][-1]])
                                if tvstr not in dictest[tkstr]:
                                    dictest[tkstr].append(tvstr)
                                    tripleobjlist.append([taea_equal, ["@@全等三角形充分条件角边角"],
                                                          [[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是",
                                                           "全等三角形"]])
                        # 边角边
                        judgequllist = [eae_list[idmain][0], eae_list[idcli][0]]
                        if set(judgequllist).issubset(equset):
                            eae_sig[0] = 1
                        judgequllist = [eae_list[idmain][0], eae_list[idcli][1]]
                        if set(judgequllist).issubset(equset):
                            eae_sig[1] = 1
                        judgequllist = [eae_list[idmain][1], eae_list[idcli][0]]
                        if set(judgequllist).issubset(equset):
                            eae_sig[2] = 1
                        judgequllist = [eae_list[idmain][1], eae_list[idcli][1]]
                        if set(judgequllist).issubset(equset):
                            eae_sig[3] = 1
                        judgequllist = [eae_list[idmain][2], eae_list[idcli][2]]
                        if set(judgequllist).issubset(equset):
                            eae_sig[4] = 1
                        if eae_sig[4] == 1 and (eae_sig[0] + eae_sig[3] == 2 or eae_sig[1] + eae_sig[2] == 2):
                            outjson.append([[eae_list[idmain][-1], eae_list[idcli][-1]], "是", "全等三角形"])
                            if self.treesig:
                                t0list = list(set([eae_list[idmain][2], eae_list[idcli][2]]))
                                keyelem = ["".join(t0list)]
                                teae_equal = []
                                teae_equal.append([[t0list], "是", "等值"])
                                if eae_sig[0] + eae_sig[3] == 2:
                                    t1list = list(set([eae_list[idmain][1], eae_list[idcli][1]]))
                                    t2list = list(set([eae_list[idmain][0], eae_list[idcli][0]]))
                                    teae_equal.append([[t1list], "是", "等值"])
                                    teae_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t2list))
                                    keyelem.append("".join(t1list))
                                elif eae_sig[1] + eae_sig[2] == 2:
                                    t1list = list(set([eae_list[idmain][0], eae_list[idcli][1]]))
                                    t2list = list(set([eae_list[idmain][1], eae_list[idcli][0]]))
                                    # tkstr = "".join(keyelem)
                                    teae_equal.append([[t1list], "是", "等值"])
                                    teae_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t2list))
                                    keyelem.append("".join(t1list))
                                tkstr = "".join(set(keyelem))
                                if tkstr not in dictest:
                                    dictest[tkstr] = []
                                tvstr = "".join([aea_list[idmain][-1], aea_list[idcli][-1]])
                                if tvstr not in dictest[tkstr]:
                                    dictest[tkstr].append(tvstr)
                                    tripleobjlist.append([teae_equal, ["@@全等三角形充分条件边角边"],
                                                          [[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是",
                                                           "全等三角形"]])
                        # 边边边  # 00 11 22 01 10 02 20 12 21
                        judgequllist = [eee_list[idmain][0], eee_list[idcli][0]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[0] = 1
                        judgequllist = [eee_list[idmain][1], eee_list[idcli][1]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[1] = 1
                        judgequllist = [eee_list[idmain][2], eee_list[idcli][2]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[2] = 1
                        judgequllist = [eee_list[idmain][0], eee_list[idcli][1]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[3] = 1
                        judgequllist = [eee_list[idmain][1], eee_list[idcli][0]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[4] = 1
                        judgequllist = [eee_list[idmain][0], eee_list[idcli][2]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[5] = 1
                        judgequllist = [eee_list[idmain][2], eee_list[idcli][0]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[6] = 1
                        judgequllist = [eee_list[idmain][1], eee_list[idcli][2]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[7] = 1
                        judgequllist = [eee_list[idmain][2], eee_list[idcli][1]]
                        if set(judgequllist).issubset(equset):
                            eee_sig[8] = 1
                        if eee_sig[0] + eee_sig[1] + eee_sig[2] == 3 or eee_sig[0] + eee_sig[7] + eee_sig[8] == 3 or \
                                                        eee_sig[3] + eee_sig[7] + eee_sig[6] == 3 or eee_sig[1] + \
                                eee_sig[5] + eee_sig[6] == 3 or \
                                                        eee_sig[4] + eee_sig[5] + eee_sig[8] == 3 or eee_sig[2] + \
                                eee_sig[3] + eee_sig[4] == 3:
                            outjson.append([[eee_list[idmain][-1], eee_list[idcli][-1]], "是", "全等三角形"])
                            if self.treesig:
                                keyelem = []
                                teee_equal = []
                                if eee_sig[0] + eee_sig[1] + eee_sig[2] == 3:
                                    t0list = list(set([eee_list[idmain][0], eee_list[idcli][0]]))
                                    t1list = list(set([eee_list[idmain][1], eee_list[idcli][1]]))
                                    t2list = list(set([eee_list[idmain][2], eee_list[idcli][2]]))
                                    teee_equal.append([[t0list], "是", "等值"])
                                    teee_equal.append([[t1list], "是", "等值"])
                                    teee_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t0list))
                                    keyelem.append("".join(t1list))
                                    keyelem.append("".join(t2list))
                                elif eee_sig[0] + eee_sig[7] + eee_sig[8] == 3:
                                    t0list = list(set([eee_list[idmain][0], eee_list[idcli][0]]))
                                    t1list = list(set([eee_list[idmain][1], eee_list[idcli][2]]))
                                    t2list = list(set([eee_list[idmain][2], eee_list[idcli][1]]))
                                    teee_equal.append([[t0list], "是", "等值"])
                                    teee_equal.append([[t1list], "是", "等值"])
                                    teee_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t0list))
                                    keyelem.append("".join(t1list))
                                    keyelem.append("".join(t2list))
                                elif eee_sig[3] + eee_sig[7] + eee_sig[6] == 3:
                                    t0list = list(set([eee_list[idmain][0], eee_list[idcli][1]]))
                                    t1list = list(set([eee_list[idmain][1], eee_list[idcli][2]]))
                                    t2list = list(set([eee_list[idmain][2], eee_list[idcli][0]]))
                                    teee_equal.append([[t0list], "是", "等值"])
                                    teee_equal.append([[t1list], "是", "等值"])
                                    teee_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t0list))
                                    keyelem.append("".join(t1list))
                                    keyelem.append("".join(t2list))
                                elif eee_sig[1] + eee_sig[5] + eee_sig[6] == 3:
                                    t0list = list(set([eee_list[idmain][1], eee_list[idcli][1]]))
                                    t1list = list(set([eee_list[idmain][0], eee_list[idcli][2]]))
                                    t2list = list(set([eee_list[idmain][2], eee_list[idcli][0]]))
                                    teee_equal.append([[t0list], "是", "等值"])
                                    teee_equal.append([[t1list], "是", "等值"])
                                    teee_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t0list))
                                    keyelem.append("".join(t1list))
                                    keyelem.append("".join(t2list))
                                elif eee_sig[4] + eee_sig[5] + eee_sig[8] == 3:
                                    t0list = list(set([eee_list[idmain][1], eee_list[idcli][0]]))
                                    t1list = list(set([eee_list[idmain][0], eee_list[idcli][2]]))
                                    t2list = list(set([eee_list[idmain][2], eee_list[idcli][1]]))
                                    teee_equal.append([[t0list], "是", "等值"])
                                    teee_equal.append([[t1list], "是", "等值"])
                                    teee_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t0list))
                                    keyelem.append("".join(t1list))
                                    keyelem.append("".join(t2list))
                                elif eee_sig[2] + eee_sig[3] + eee_sig[4] == 3:
                                    t0list = list(set([eee_list[idmain][2], eee_list[idcli][2]]))
                                    t1list = list(set([eee_list[idmain][1], eee_list[idcli][0]]))
                                    t2list = list(set([eee_list[idmain][0], eee_list[idcli][1]]))
                                    teee_equal.append([[t0list], "是", "等值"])
                                    teee_equal.append([[t1list], "是", "等值"])
                                    teee_equal.append([[t2list], "是", "等值"])
                                    keyelem.append("".join(t0list))
                                    keyelem.append("".join(t1list))
                                    keyelem.append("".join(t2list))
                                tkstr = "".join(set(keyelem))
                                if tkstr not in dictest:
                                    dictest[tkstr] = []
                                tvstr = "".join([eee_list[idmain][-1], eee_list[idcli][-1]])
                                if tvstr not in dictest[tkstr]:
                                    dictest[tkstr].append(tvstr)
                                    tripleobjlist.append([teee_equal, ["@@全等三角形充分条件边边边"],
                                                          [[[eee_list[idmain][-1], eee_list[idcli][-1]]], "是",
                                                           "全等三角形"]])
        if self.treesig:
            if self.debugsig:
                print("elements2congruent_triangle")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def judge_stop(self, oldsetobj, newsetobj, stopobj):
        "每步推理的具体操作 true为应该结束"
        nomatchsig = 0
        for key in newsetobj:
            if setobj[key]["结构形式"] in ["一级集合", "一级列表"]:
                for i1 in oldsetobj[key]:
                    if i1 not in newsetobj[key]:
                        nomatchsig = 1
                        break
                if nomatchsig == 1:
                    break
            elif setobj[key]["结构形式"] in ["一级列表二级集合", "一级列表二级列表"]:
                for tnlist in newsetobj[key]:
                    havesig = 0
                    for tolist in oldsetobj[key]:
                        if set(tnlist).issuperset(set(tolist)) and len(tolist) == len(tnlist):
                            havesig = 1
                            break
                    if havesig == 0:
                        nomatchsig = 1
                        break
                if nomatchsig == 1:
                    break
        if nomatchsig == 0:
            message = "已知条件无法进一步推理"
            return message, True
        # if operator.eq(oldsetobj, newsetobj):
        #     message = "已知条件无法进一步推理"
        #     return message, True
        for key in stopobj.keys():
            if setobj[key]["结构形式"] in ["一级集合", "一级列表"]:
                if not stopobj[key].issubset(newsetobj[key]):
                    message = "待求 {}: {}".format(key, stopobj[key])
                    return message, False
            elif setobj[key]["结构形式"] in ["一级列表二级集合", "一级列表二级列表"]:
                for i1 in stopobj[key]:
                    # 找到不匹配 为 0
                    findsig = 0
                    for i2 in newsetobj[key]:
                        if i1.issubset(i2):
                            findsig = 1
                            break
                    if findsig == 0:
                        message = "待求 {}: {}".format(key, stopobj[key])
                        return message, False
        message = "推理完成生成全量条件中..."
        return message, False


def recog_str2list(instr):
    " 输入为 text 、latex "
    # 1. 答案字符标准化
    if isinstance(instr, str):
        # 1.1 行级 处理单元
        # printstr3 = printstr3.replace("\\\\", "\\").replace("\\n", "\n")
        handestr3 = instr.replace("\\\n", "\n")
        # print(handestr3)
        # 1.2 句组级 处理单元
        sentenc_list = re.split('。|\?|？|！|；|。|;|\n', handestr3)
        ans_inlist = []
        for sentence in sentenc_list:
            sp_list = sentence.strip(" ,，.。\t").split("$")
            ans_inlist.append([{"text": sp} if id % 2 == 0 else {"latex": sp} for id, sp in enumerate(sp_list)])
    else:
        ans_inlist = instr
    return ans_inlist


def title_latex_prove(instr):
    "输入：题目字符串，输出：序列化条件，序列化的树"
    # 1. 问题字符标准化
    ans_inlist = recog_str2list(instr)
    # 2. 分析问句
    li_ins = LogicalInference()
    nodejson, edgelist = li_ins(ans_inlist)
    return nodejson, edgelist


def answer_latex_prove(instr, inconditon, intree, checkpoints=[]):
    "输入：解答字符串，输出：序列化要素，相关知识点报告"
    # 1. 答案字符标准化
    ans_inlist = recog_str2list(instr)
    # 2. 分解答案
    li_ins = LogicalInference()
    li_ins.answer2normal(ans_inlist)
    outjson = li_ins.analysis_tree(inconditon, intree, checkpoints=checkpoints)
    return outjson


if __name__ == '__main__':
    """
    印刷体的提问规范：
      1. 写出四则运算表达式（公式题目类型指定, 指定精度） 
      2. 列出方程 并求解 x 代表小明 y 代表时间（方程题 指明变量, 指定精度）
      3. pi 取 3.1415 （常数需赋值）
      4. varlist 至简从单字符变量（a） 到多字符变量 （a_{2}, a_{2}b_{3}），不包含前后缀如 角度
    手写输入为: 印刷体的输出
      5. 用三点表示角。圆上弧 按弧上的点依次描述。直径需要三点描述 AOB。
    """
    # jsonkey = "直径"
    # mthead = re.match(u"[是|的]{}$".format(jsonkey), "直径")
    # print(mthead)
    # raise 333
    # ss = " 。asdfb.,".strip(",，。 ")
    # print(ss)
    # 3. latex 证明
    # ss = "@aa12 d ~ f"
    # se = re.match(r"^(\w|\s)+", ss)
    # print(se.group())
    # se = re.sub(r"^(\w|\s)+","", ss)
    # print(se.string)
    # print(se)
    # 一、中等难度 多边形
    printstr3 = "已知：正方形 $ABCD, A、P、C $ 在一条直线上。$MN \\parallel BC, \\angle {BPQ} =90 ^{\\circ},A、M、B $ 在一条直线上，$M、P、N $ 在一条直线上，$\\angle {APM}$是锐角，$\\angle {NPQ} +\\angle {BPQ} +\\angle {BPM} =180^{\\circ }, \\angle {ACB}$是锐角。 $C、Q、 N、D $ 在一条直线上。求证 $PB = PQ$"
    # 1. 正确，不同路径的demo1. （考点为 等腰三角形 全等三角形 表达式传递）
    handestr3 = "$ \\because \\angle {MAP} = \\angle {MPA}, \\therefore \\triangle {AMP} $是等腰三角形, $ \\therefore AM=PM, \\because AB=MN,\\therefore MB=PN,\\because \\angle {BPQ}=90 ^ {\\circ},\\therefore \\angle {BPM} + \\angle {NPQ} = 90 ^ {\\circ},\\because \\angle {NPQ} + \\angle {NQP} = 90 ^ {\\circ},\\therefore \\angle {MBP} = \\angle {NPQ},\\because \\triangle {BPM}$ 是直角三角形。$\\because \\triangle {NPQ}$ 是直角三角形$ \\therefore \\angle {BMP} = \\angle {PNQ}, \\therefore \\triangle {BPM} \\cong \\triangle {NPQ},\\therefore PB = PQ $"
    # 2. 正确，不同路径的demo1. （考点为 等腰三角形 全等三角形 表达式传递）
    handestr3 = "$ \\because \\angle {NCP} = \\angle {NPC}, \\therefore \\triangle {NPC} $是等腰三角形, $ \\therefore CN=PN, \\because CN=MB,\\therefore MB=PN,\\because \\angle {BPQ}=90 ^ {\\circ},\\therefore \\angle {BPM} + \\angle {NPQ} = 90 ^ {\\circ},\\because \\angle {NPQ} + \\angle {NQP} = 90 ^ {\\circ},\\therefore \\angle {MBP} = \\angle {NPQ},\\because \\triangle {BPM}$ 是直角三角形。$\\because \\triangle {NPQ}$ 是直角三角形$ \\therefore \\angle {BMP} = \\angle {PNQ}, \\therefore \\triangle {BPM} \\cong \\triangle {NPQ},\\therefore PB = PQ $"
    # 3. 考点描述不全, 证明有断层。（未描述等腰三角形）
    handestr3 = "$ \\because \\angle {MAP} = \\angle {MPA}, \\therefore AM=PM, \\because AB=MN,\\therefore MB=PN,\\because \\angle {BPQ}=90 ^ {\\circ},\\therefore \\angle {BPM} + \\angle {NPQ} = 90 ^ {\\circ},\\because \\angle {NPQ} + \\angle {NQP} = 90 ^ {\\circ},\\therefore \\angle {MBP} = \\angle {NPQ},\\because \\triangle {BPM}$ 是直角三角形。$\\because \\triangle {NPQ}$ 是直角三角形$ \\therefore \\angle {BMP} = \\angle {PNQ}, \\therefore \\triangle {BPM} \\cong \\triangle {NPQ},\\therefore PB = PQ $"
    # 4. 答题文字如3 但题目考点不考等腰三角形，证明成功。
    handestr3 = "$ \\because \\angle {MAP} = \\angle {MPA}, \\therefore AM=PM, \\because AB=MN,\\therefore MB=PN,\\because \\angle {BPQ}=90 ^ {\\circ},\\therefore \\angle {BPM} + \\angle {NPQ} = 90 ^ {\\circ},\\because \\angle {NPQ} + \\angle {NQP} = 90 ^ {\\circ},\\therefore \\angle {MBP} = \\angle {NPQ},\\because \\triangle {BPM}$ 是直角三角形。$\\because \\triangle {NPQ}$ 是直角三角形$ \\therefore \\angle {BMP} = \\angle {PNQ}, \\therefore \\triangle {BPM} \\cong \\triangle {NPQ},\\therefore PB = PQ $"
    # 5. 描述错误，但不影响答案。
    handestr3 = "$ \\because \\angle {MAF} = \\angle {MPA}, \\because \\angle {MAP} = \\angle {MPA}, \\therefore \\triangle {AMP} $是等腰三角形, $ \\therefore AM=PM, \\because AB=MN,\\therefore MB=PN,\\because \\angle {BPQ}=90 ^ {\\circ},\\therefore \\angle {BPM} + \\angle {NPQ} = 90 ^ {\\circ},\\because \\angle {NPQ} + \\angle {NQP} = 90 ^ {\\circ},\\therefore \\angle {MBP} = \\angle {NPQ},\\because \\triangle {BPM}$ 是直角三角形。$\\because \\triangle {NPQ}$ 是直角三角形$ \\therefore \\angle {BMP} = \\angle {PNQ}, \\therefore \\triangle {BPM} \\cong \\triangle {NPQ},\\therefore PB = PQ $"
    checkpoints = [
        # "@@求证",
        "@@表达式传递",
        # "@@全等三角形间传递",
        # "@@全等三角形充分条件边角边",
        "@@全等三角形充分条件角边角", "@@全等三角形必要条件",
        # "@@等边三角形充分条件角", "@@等边三角形充分条件边",
        # "@@等腰三角形充分条件边",
        "@@等腰三角形充分条件角",
        "@@等腰三角形必要条件边",
        # "@@等腰三角形必要条件角",
    ]
    # 二、难度未知 圆
    printstr3 = "已知：$\\triangle {ABC} , AB=AC, \\bigodot {OABDE}, B、D、C $ 在一条直线上，$AOB $ 是直径，$A、E、C$ 在一条直线上。求证 $\\triangle {CDE} $ 是等腰三角形。"
    # 1. 正确，不同路径的demo1. （考点为 等腰三角形 全等三角形 表达式传递）
    # handestr3 = "$\\because \\bigodot {OABDE} ; \\therefore \\angle {AED}, \\angle {ABD} $是补角。 $ \\because AB = AC, \\therefore \\triangle {ABC}$ 是等腰三角形。$\\therefore \\angle {ABC} = \\angle {ACB}, \\therefore \\angle {DEC} = \\angle {ACB}, \\because \\angle {DEC} = \\angle {DCE}, \\therefore \\triangle {DEC}$ 是等腰三角形。"
    handestr3 = "$\\because \\bigodot {OABDE} $。$\\therefore \\angle {AED}, \\angle {ABD} $是补角。 $ \\because AB = AC, \\therefore \\triangle {ABC}$ 是等腰三角形。$\\therefore \\angle {ABC} = \\angle {ACB}, \\therefore \\angle {DEC} = \\angle {ACB}, \\because \\angle {DEC} = \\angle {DCE}, \\therefore \\triangle {DEC}$ 是等腰三角形。"
    checkpoints = [
        # "@@表达式传递",
        "@@圆内接四边形的性质",
        # # "@@圆等弧对等角",
        # # "@@圆周角求和关系",
        # # "@@圆心角求和关系",
        # # "@@圆弧关系",
        "@@等腰三角形充分条件边",
        "@@等腰三角形必要条件角",
        "@@等腰三角形充分条件角",
        # "@@等腰三角形必要条件边",
    ]
    # 三、难度未知 弦切角
    printstr3 = "已知：$\\bigodot {OABCD}, A、D、E、F $ 在一条直线上，$AOB $ 是直径，$B、C、E$ 在一条直线上。$CF \\perp AE， CF \\perp OC， \\angle {DCF} $ 是 $\\overset{\\frown} {ODC}$的弦切角。求证 $\\triangle {ABE} $ 是等腰三角形。"
    # 1. 正确，不同路径的demo1. （考点为 等腰三角形 全等三角形 表达式传递）
    handestr3 = "$\\because \\bigodot {OABCD} $。$\\therefore \\angle {BCD}, \\angle {BAD} $是补角。 $\\therefore \\angle {CDE}, \\angle {ABC} $是补角。 $ \\therefore \\triangle {ABC} = \\triangle {CDE}, \\triangle {BAD} = \\triangle {DCE}, \\because \\angle {ACE} $ 是直角。$ \\therefore \\angle {ECF} = \\angle {CAE}, \\therefore \\angle {DCF} $ 是 $ \\overset{\\frown} {DC}$的弦切角。 $\\therefore \\angle {CAE} = \\angle {DCF}, \\therefore \\angle {AEB} = \\angle {ABE}, \\therefore \\triangle {DEC}$ 是等腰三角形。"
    checkpoints = [
        # # "@@表达式传递",
        # "@@圆内接四边形的性质",
        # "@@弦切角性质",
        # # # "@@圆等弧对等角",
        # # # "@@圆周角求和关系",
        # # # "@@圆心角求和关系",
        # # # "@@圆弧关系",
        # "@@等腰三角形充分条件边",
        # "@@等腰三角形必要条件角",
        # "@@等腰三角形充分条件角",
        # # "@@等腰三角形必要条件边",
    ]
    # 题目
    outelem, outtree = title_latex_prove(printstr3)
    raise 123
    # print("原答案")
    # print(handestr3)
    print(handestr3)
    inconditon = "../nodejson.json"
    intree = "../edgejson.json"
    nodejson = json.load(open(inconditon, "r"))
    edgelist = json.load(open(intree, "r"))
    outreport = answer_latex_prove(handestr3, nodejson, edgelist, checkpoints=checkpoints)
    print("end")
    exit()

    # 1. 单行测验
    # printstr1 = "$\\therefore \\angle{ECA}=\\angle{CED}+\\angle{CDE}$"
    printstr1 = "$\\therefore CE=AE=\\frac{1}{2}AB$"
    print(printstr1)
    # 先按概念实例合并表达式，再按句意分割，合并最小单元
    varlist = ["CE", "AE", "AB"]
    tmplist = latex2list_P(printstr1, varlist=varlist)
    print(tmplist)
    # # 3. 解析堆栈形式
    postfix = postfix_convert_P(tmplist)
    print(postfix)
    # res = re.findall(r"/h3>([\s\S]+?)</div", instr)
