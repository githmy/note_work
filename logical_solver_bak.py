# coding:utf-8
"""
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

# 0. 设计 demo
# 1. 衍生一级元素
# 2. 关系提取 
# 3. 根据规则，写入三元组
  属性值为 函数，则该主体 的 实例必有 集合属性，作为函数的参数。
# 3. 三元组条件 空间步骤推演
# 3.1 正推 概念 衍生 条件
# 3.2 反推 条件 得出 概念 
# 4. 学生条件的反推，可以得到概念
   条件全 没到概念，不理解
   有概念 没到掌握，不理解
   低step 掌握 高step 没掌握, 逻辑思维不足 


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
            self._setobj = {"锐角集合": set(), "钝角集合": set(), "等价集合": [], "全等集合": [], "全等三角形集合": [],
                            "垂直集合": [], "平行集合": [], "直角集合": set(), "平角集合": set(), "直角三角形集合": set(),
                            "余角集合": [], "补角集合": [], "表达式集合": set()}
            self._stopobj = {}
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
                # print(keydic)
                print(onetri)
                raise Exception("没有考虑的情况")
        # for onetri in delec:
        #     onetri
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
            for word in self.latex_map:
                # 每个 属性词
                if "n" == self.nominal[self.latex_map[word]]:
                    tstrli = latex_fenci(tstr)
                    tstr, tjson = self.get_extract(tstrli, word)
                    outjson += tjson
            latexlist.append(tstr)
        return latexlist, outjson

    def latex_extract_word(self, instr):
        """单句： 空格标准分组后 返还 去掉抽象概念的实体latex"""
        # 1. token 预处理
        tinlist = re.split(',|，|、|\n|\t', latex2space(instr))
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
        logger1.info("write property: %s" % write_json["add"]["properobj"])
        # 2. 修正元组实体
        # print(113)
        # print(write_json["add"]["triobj"])
        # write_json["add"]["triobj"] = [
        #     [latex2unit(online[0], varlist=varlist), online[1], latex2unit(online[2], varlist=varlist)] for online in
        #     write_json["add"]["triobj"]]
        # print(write_json["add"]["triobj"])
        # 3. 再写元组
        space_ins._setobj, _, space_ins._stopobj = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                                          space_ins._stopobj,
                                                                          addc=write_json["add"]["triobj"],
                                                                          delec=write_json["dele"]["triobj"])
        # space_ins.tri2set_oper(basic_space_ins, addc=write_json["add"]["triobj"],
        #                        delec=write_json["dele"]["triobj"])
        # space_ins.triple_oper(space_ins._proper_trip, addc=write_json["add"]["triobj"],
        #                       delec=write_json["dele"]["triobj"])
        logger1.info("write triple: %s" % write_json["add"]["triobj"])
        # space_ins.property_oper(space_ins._properobj, addc=write_json["add"]["properobj"], delec=write_json["dele"]["properobj"])
        # space_ins.triple_oper(space_ins._triobj, addc=write_json["add"]["triobj"], delec=write_json["dele"]["triobj"])

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

    def nature2space(self, instrs, gstack):
        """ 解析：自然语言 到 空间三元组。按因果 分步骤"""
        # 1. 语言录入初始化, 添加语句
        instr_list = re.split(',|，|;|\n|\\\qquad|\\\quad|\t', instrs)
        # stand_fenci_list = [[i1.strip()] for i1 in instr_list]
        stand_fenci_list = [latex_fenci(i1.strip()) for i1 in instr_list]
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
        print(basic_space_ins._proper_trip)
        print(basic_space_ins._relation_trip)
        print(space_ins._proper_trip)
        print(space_ins._relation_trip)
        triobj = self.fenci2triple(stand_fenci_list, basic_space_ins)
        logger1.info("json write: %s" % triobj)
        # 5. 写入空间, 先写 属性再根据属性 合并 三元组
        # propertyjson = [{"因为": i1} for i1 in propertyjson]
        self.json2space(triobj, basic_space_ins, space_ins)
        print(basic_space_ins._proper_trip)
        print(basic_space_ins._relation_trip)
        print(space_ins._proper_trip)
        print(space_ins._relation_trip)
        print("check ok")


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
        self.debugsig = True
        # self.debugsig = False
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
        # 0. 处理句间 关系，写入部分实体。 基于符号类型的区分标签。结果全部写入内存。
        anastr = self.sentence2normal(analist)
        logger1.info("initial clean sentence: %s" % analist)
        # 1. 循环处理每句话 生成空间解析 和 步骤list
        # for sentence in analist:
        #     logger1.info("sentence: %s" % sentence)
        # 2. 基于因果的符号标签
        # self.analyize_strs(anastr)
        # 2. 内存推理，基于之前的步骤条件
        self.get_condition_tree()
        return self.inference()

    def analysis_tree(self, anastr, inconditon, intree):
        # todo: 1. 解析答案元素 加入树 2. 对比 get_condition_tree 对应树上的报告
        nodejson = json.loads(inconditon, encoding="utf-8")
        edgelist = json.loads(intree, encoding="utf-8")
        reportjson = {}
        return json.dumps(reportjson, ensure_ascii=False)

    def get_condition_tree(self):
        " 根据条件构建 思维树 "
        # 1. 定义空间
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # print(space_ins._step_node)
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
        for obj in space_ins._stopobj:
            if obj in listset_obj:
                for idt, tlist in enumerate(space_ins._stopobj[obj]):
                    space_ins._stopobj[obj][idt] = list(tlist)
            elif obj in set_obj:
                space_ins._stopobj[obj] = list(space_ins._stopobj[obj])
        # 用于无条件连接
        space_ins._initobj["默认集合"] = [0]
        space_ins._step_node["已知"] = {"condjson": {}, "points": ["@@已知"], "outjson": space_ins._initobj}
        space_ins._step_node["求证"] = {"condjson": space_ins._stopobj, "points": ["@@求证"], "outjson": {}}
        instra = """{'0': {'condjson': {'正方形集合': ['{正方形@ABCD}']}, 'points': ['@@正方形平行属性'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@BC}'], ['{线段@CD}', '{线段@AB}']]}}, '1': {'condjson': {'正方形集合': ['{正方形@ABCD}']}, 'points': ['@@正方形垂直属性'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@CD}'], ['{线段@CD}', '{线段@BC}'], ['{线段@BC}', '{线段@AB}'], ['{线段@AD}', '{线段@AB}']]}}, '2': {'condjson': {'正方形集合': ['{正方形@ABCD}']}, 'points': ['@@正方形等边属性'], 'outjson': {'等值集合': [['{线段@BC}', '{线段@AD}', '{线段@AB}', '{线段@CD}']]}}, '3': {'condjson': {'正方形集合': ['{正方形@ABCD}']}, 'points': ['@@正方形直角属性'], 'outjson': {'直角集合': ['{角@ABC}', '{角@BCD}', '{角@ADC}', '{角@BAD}'], '直角三角形集合': ['{三角形@ABC}', '{三角形@BCD}', '{三角形@ACD}', '{三角形@ABD}']}}, '4': {'condjson': {'直线集合': [['{点@D}', '{点@N}', '{点@Q}', '{点@C}']]}, 'points': ['@@直线得出表达式'], 'outjson': {'表达式集合': ['{线段@DQ} = {线段@NQ} + {线段@DN}', '{线段@CD} = {线段@CN} + {线段@DN}', '{线段@CD} = {线段@CQ} + {线段@DQ}', '{线段@CN} = {线段@CQ} + {线段@NQ}']}}, '5': {'condjson': {'直线集合': [['{点@D}', '{点@N}', '{点@Q}', '{点@C}']]}, 'points': ['@@直线得出补角'], 'outjson': {'平角集合': ['{角@DNQ}', '{角@CND}', '{角@CQD}', '{角@CQN}'], '补角集合': [['{角@AND}', '{角@ANQ}'], ['{角@BND}', '{角@BNQ}'], ['{角@DNP}', '{角@PNQ}'], ['{角@DNM}', '{角@MNQ}'], ['{角@AND}', '{角@ANC}'], ['{角@BND}', '{角@BNC}'], ['{角@DNP}', '{角@CNP}'], ['{角@DNM}', '{角@CNM}'], ['{角@AQD}', '{角@AQC}'], ['{角@BQD}', '{角@BQC}'], ['{角@DQP}', '{角@CQP}'], ['{角@DQM}', '{角@CQM}'], ['{角@AQN}', '{角@AQC}'], ['{角@BQN}', '{角@BQC}'], ['{角@NQP}', '{角@CQP}'], ['{角@MQN}', '{角@CQM}']]}}, '6': {'condjson': {'直线集合': [['{点@M}', '{点@P}', '{点@N}']]}, 'points': ['@@直线得出表达式'], 'outjson': {'表达式集合': ['{线段@MN} = {线段@NP} + {线段@MP}']}}, '7': {'condjson': {'直线集合': [['{点@M}', '{点@P}', '{点@N}']]}, 'points': ['@@直线得出补角'], 'outjson': {'平角集合': ['{角@MPN}'], '补角集合': [['{角@APM}', '{角@APN}'], ['{角@BPM}', '{角@BPN}'], ['{角@MPQ}', '{角@NPQ}'], ['{角@CPM}', '{角@CPN}'], ['{角@DPM}', '{角@DPN}']]}}, '8': {'condjson': {'默认集合': [0]}, 'points': ['@@补角属性'], 'outjson': {'钝角集合': ['{角@APN}', '{角@CPM}', '{角@ANQ}', '{角@BND}', '{角@ANC}', '{角@AQC}', '{角@BQD}', '{角@BQN}', '{角@AMC}', '{角@BMD}', '{角@CQP}', '{角@CQM}', '{角@BPN}', '{角@MPQ}', '{角@DPM}'], '锐角集合': ['{角@APM}', '{角@AND}', '{角@BNQ}', '{角@BNC}', '{角@AQD}', '{角@BQC}', '{角@AQN}', '{角@CPN}', '{角@BMC}', '{角@AMD}', '{角@DQP}', '{角@DQM}', '{角@NQP}', '{角@MQN}', '{角@BPM}', '{角@NPQ}', '{角@DPN}']}}, '9': {'condjson': {'直线集合': [['{点@A}', '{点@M}', '{点@B}']]}, 'points': ['@@直线得出表达式'], 'outjson': {'表达式集合': ['{线段@AB} = {线段@BM} + {线段@AM}']}}, '10': {'condjson': {'直线集合': [['{点@A}', '{点@M}', '{点@B}']]}, 'points': ['@@直线得出补角'], 'outjson': {'平角集合': ['{角@AMB}'], '补角集合': [['{角@AMP}', '{角@BMP}'], ['{角@AMQ}', '{角@BMQ}'], ['{角@AMN}', '{角@BMN}'], ['{角@AMC}', '{角@BMC}'], ['{角@AMD}', '{角@BMD}']]}}, '11': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']]}, 'points': ['@@直线得出表达式'], 'outjson': {'表达式集合': ['{线段@AC} = {线段@CP} + {线段@AP}']}}, '12': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']]}, 'points': ['@@直线得出补角'], 'outjson': {'平角集合': ['{角@APC}'], '补角集合': [['{角@APB}', '{角@BPC}'], ['{角@APQ}', '{角@CPQ}'], ['{角@APM}', '{角@CPM}'], ['{角@APN}', '{角@CPN}'], ['{角@APD}', '{角@CPD}']]}}, '13': {'condjson': {'默认集合': [0]}, 'points': ['@@垂直直线的线段属性'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DN}'], ['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@CD}'], ['{线段@AD}', '{线段@NQ}'], ['{线段@AD}', '{线段@CN}'], ['{线段@AD}', '{线段@CQ}'], ['{线段@BC}', '{线段@DN}'], ['{线段@BC}', '{线段@DQ}'], ['{线段@CD}', '{线段@BC}'], ['{线段@BC}', '{线段@NQ}'], ['{线段@CN}', '{线段@BC}'], ['{线段@BC}', '{线段@CQ}'], ['{线段@BC}', '{线段@AM}'], ['{线段@BC}', '{线段@AB}'], ['{线段@BC}', '{线段@BM}'], ['{线段@AD}', '{线段@AM}'], ['{线段@AD}', '{线段@AB}'], ['{线段@AD}', '{线段@BM}'], ['{线段@MP}', '{线段@DN}'], ['{线段@MN}', '{线段@DN}'], ['{线段@NP}', '{线段@DN}'], ['{线段@MP}', '{线段@DQ}'], ['{线段@MN}', '{线段@DQ}'], ['{线段@NP}', '{线段@DQ}'], ['{线段@CD}', '{线段@MP}'], ['{线段@CD}', '{线段@MN}'], ['{线段@CD}', '{线段@NP}'], ['{线段@NQ}', '{线段@MP}'], ['{线段@NQ}', '{线段@MN}'], ['{线段@NQ}', '{线段@NP}'], ['{线段@CN}', '{线段@MP}'], ['{线段@CN}', '{线段@MN}'], ['{线段@CN}', '{线段@NP}'], ['{线段@MP}', '{线段@CQ}'], ['{线段@MN}', '{线段@CQ}'], ['{线段@NP}', '{线段@CQ}'], ['{线段@AM}', '{线段@MP}'], ['{线段@AM}', '{线段@MN}'], ['{线段@NP}', '{线段@AM}'], ['{线段@AB}', '{线段@MP}'], ['{线段@AB}', '{线段@MN}'], ['{线段@AB}', '{线段@NP}'], ['{线段@BM}', '{线段@MP}'], ['{线段@BM}', '{线段@MN}'], ['{线段@NP}', '{线段@BM}']]}}, '14': {'condjson': {'默认集合': [0]}, 'points': ['@@垂直直角的属性'], 'outjson': {'直角集合': ['{角@ADN}', '{角@ADQ}', '{角@ADC}', '{角@BCD}', '{角@BCN}', '{角@BCQ}', '{角@ABC}', '{角@CBM}', '{角@DAM}', '{角@BAD}', '{角@DNM}', '{角@DNP}', '{角@MNQ}', '{角@PNQ}', '{角@CNM}', '{角@CNP}', '{角@AMP}', '{角@AMN}', '{角@BMP}', '{角@BMN}']}}, '15': {'condjson': {'默认集合': [0]}, 'points': ['@@余角性质'], 'outjson': {'锐角集合': ['{角@DAN}', '{角@AND}', '{角@DAQ}', '{角@AQD}', '{角@CAD}', '{角@ACD}', '{角@BDC}', '{角@CBD}', '{角@BNC}', '{角@CBN}', '{角@BQC}', '{角@CBQ}', '{角@ACB}', '{角@BAC}', '{角@BCM}', '{角@BMC}', '{角@ADM}', '{角@AMD}', '{角@ADB}', '{角@ABD}', '{角@MDN}', '{角@DMN}', '{角@NDP}', '{角@DPN}', '{角@MQN}', '{角@NMQ}', '{角@NQP}', '{角@NPQ}', '{角@MCN}', '{角@CMN}', '{角@NCP}', '{角@CPN}', '{角@MAP}', '{角@APM}', '{角@MAN}', '{角@ANM}', '{角@MBP}', '{角@BPM}', '{角@MBN}', '{角@BNM}']}}, '16': {'condjson': {'默认集合': [0]}, 'points': ['@@垂直性质'], 'outjson': {'直角三角形集合': ['{三角形@ADN}', '{三角形@ADQ}', '{三角形@ACD}', '{三角形@BCD}', '{三角形@BCN}', '{三角形@BCQ}', '{三角形@ABC}', '{三角形@BCM}', '{三角形@ADM}', '{三角形@ABD}', '{三角形@DMN}', '{三角形@DNP}', '{三角形@MNQ}', '{三角形@NPQ}', '{三角形@CMN}', '{三角形@CNP}', '{三角形@AMP}', '{三角形@AMN}', '{三角形@BMP}', '{三角形@BMN}']}}, '17': {'condjson': {'默认集合': [0]}, 'points': ['@@平行属性'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}'], ['{线段@CD}', '{线段@DN}', '{线段@DQ}', '{线段@CD}', '{线段@NQ}', '{线段@CN}', '{线段@CQ}', '{线段@AB}', '{线段@AM}', '{线段@AB}', '{线段@BM}']]}}, '18': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@CD}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@BC}']]}}, '19': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@AD}', '{线段@CD}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@AB}']]}}, '20': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@CD}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AM}']]}}, '21': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@CD}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@MN}']]}}, '22': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@CD}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@AD}']]}}, '23': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@CD}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@AB}']]}}, '24': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@CD}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@MN}']]}}, '25': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CD}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AM}']]}}, '26': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@BC}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@MN}']]}}, '27': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@AB}']]}}, '28': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@BC}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@CD}']]}}, '29': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@BC}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@MN}']]}}, '30': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BC}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AM}']]}}, '31': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@AB}']]}}, '32': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@AD}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@CD}']]}}, '33': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AM}']]}}, '34': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DN}']]}}, '35': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AM}']]}}, '36': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}']]}}, '37': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@AM}']]}}, '38': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@NQ}']]}}, '39': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AM}']]}}, '40': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@CN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@BC}']]}}, '41': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@CN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AM}']]}}, '42': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@CQ}']]}}, '43': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AM}']]}}, '44': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@BC}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MN}', '{线段@DN}']]}}, '45': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DN}']]}}, '46': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@BC}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MN}', '{线段@DN}']]}}, '47': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BC}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AM}']]}}, '48': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@BC}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MN}', '{线段@DQ}']]}}, '49': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}']]}}, '50': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@BC}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MN}', '{线段@DQ}']]}}, '51': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BC}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@BM}'], ['{线段@BC}', '{线段@AM}']]}}, '52': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}']]}}, '53': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@NQ}']]}}, '54': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}']]}}, '55': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AM}']]}}, '56': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@CN}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@MN}']]}}, '57': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@CN}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@AD}']]}}, '58': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@CN}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@MN}']]}}, '59': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CN}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AM}']]}}, '60': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MN}', '{线段@CQ}']]}}, '61': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@CQ}']]}}, '62': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MN}', '{线段@CQ}']]}}, '63': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AM}']]}}, '64': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@BC}', '{线段@AM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AM}', '{线段@MN}']]}}, '65': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@AM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@AM}']]}}, '66': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@BC}', '{线段@AM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AM}', '{线段@MN}']]}}, '67': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BC}', '{线段@AM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@BM}']]}}, '68': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}']], '垂直集合': [['{线段@BC}', '{线段@BM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BM}', '{线段@MN}']]}}, '69': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@BM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@BM}']]}}, '70': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']], '垂直集合': [['{线段@BC}', '{线段@BM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BM}', '{线段@MN}']]}}, '71': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BC}', '{线段@BM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AM}']]}}, '72': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@AM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@AM}']]}}, '73': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@AM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@BM}']]}}, '74': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@BM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BC}', '{线段@BM}']]}}, '75': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AD}', '{线段@BM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AM}']]}}, '76': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}'], ['{线段@AD}', '{线段@BC}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}']]}}, '77': {'condjson': {'平行集合': [['{线段@BC}', '{线段@AD}'], ['{线段@CD}', '{线段@AB}']]}, 'points': ['@@平行线间平行线等值'], 'outjson': {'等值集合': [['{线段@BC}', '{线段@AD}'], ['{线段@CD}', '{线段@AB}']]}}, '78': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}'], ['{线段@BM}', '{线段@CN}']]}, 'points': ['@@平行线间平行线等值'], 'outjson': {'等值集合': [['{线段@BC}', '{线段@MN}'], ['{线段@BM}', '{线段@CN}']]}}, '79': {'condjson': {'平行集合': [['{线段@AD}', '{线段@MN}'], ['{线段@AM}', '{线段@DN}']]}, 'points': ['@@平行线间平行线等值'], 'outjson': {'等值集合': [['{线段@AD}', '{线段@MN}'], ['{线段@AM}', '{线段@DN}']]}}, '80': {'condjson': {'直线集合': [['{点@D}', '{点@N}', '{点@Q}', '{点@C}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DNM}', '{角@DNP}'], ['{角@MNQ}', '{角@PNQ}', '{角@CNM}', '{角@CNP}'], ['{角@BCD}', '{角@BCN}', '{角@BCQ}']]}}, '81': {'condjson': {'直线集合': [['{点@D}', '{点@N}', '{点@Q}', '{点@C}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ADN}', '{角@ADQ}', '{角@ADC}'], ['{角@BCD}', '{角@BCN}', '{角@BCQ}']]}}, '82': {'condjson': {'直线集合': [['{点@D}', '{点@N}', '{点@Q}', '{点@C}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DNM}', '{角@DNP}'], ['{角@MNQ}', '{角@PNQ}', '{角@CNM}', '{角@CNP}'], ['{角@BCD}', '{角@BCN}', '{角@BCQ}']]}}, '83': {'condjson': {'直线集合': [['{点@M}', '{点@P}', '{点@N}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BMP}', '{角@BMN}'], ['{角@AMP}', '{角@AMN}'], ['{角@DNM}', '{角@DNP}'], ['{角@MNQ}', '{角@CNM}', '{角@PNQ}', '{角@CNP}']]}}, '84': {'condjson': {'直线集合': [['{点@A}', '{点@M}', '{点@B}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BMP}', '{角@BMN}'], ['{角@AMP}', '{角@AMN}'], ['{角@ABC}', '{角@CBM}']]}}, '85': {'condjson': {'直线集合': [['{点@A}', '{点@M}', '{点@B}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAM}', '{角@BAD}'], ['{角@ABC}', '{角@CBM}']]}}, '86': {'condjson': {'直线集合': [['{点@A}', '{点@M}', '{点@B}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BMP}', '{角@BMN}'], ['{角@AMP}', '{角@AMN}'], ['{角@ABC}', '{角@CBM}']]}}, '87': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ACB}', '{角@BCP}']]}}, '88': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '锐角集合': ['{角@ACB}', '{角@BCP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ACB}', '{角@BCP}']]}}, '89': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAP}', '{角@CAD}'], ['{角@ACB}', '{角@BCP}']]}}, '90': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '锐角集合': ['{角@DAP}', '{角@CAD}', '{角@ACB}', '{角@BCP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAP}', '{角@CAD}', '{角@ACB}', '{角@BCP}']]}}, '91': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAP}', '{角@BAP}', '{角@CAM}', '{角@BAC}'], ['{角@ACD}', '{角@ACN}', '{角@ACQ}', '{角@DCP}', '{角@NCP}', '{角@PCQ}']]}}, '92': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '锐角集合': ['{角@MAP}', '{角@BAP}', '{角@CAM}', '{角@BAC}', '{角@ACD}', '{角@ACN}', '{角@ACQ}', '{角@DCP}', '{角@NCP}', '{角@PCQ}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAP}', '{角@BAP}', '{角@CAM}', '{角@BAC}', '{角@ACD}', '{角@ACN}', '{角@ACQ}', '{角@DCP}', '{角@NCP}', '{角@PCQ}']]}}, '93': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@APM}', '{角@CPN}'], ['{角@CPM}', '{角@APN}'], ['{角@ACB}', '{角@BCP}']]}}, '94': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '锐角集合': ['{角@APM}', '{角@APM}', '{角@ACB}', '{角@BCP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@APM}', '{角@APM}', '{角@ACB}', '{角@BCP}']]}}, '95': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '钝角集合': ['{角@APN}', '{角@CPM}', '{角@APN}', '{角@CPM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@APN}', '{角@CPM}', '{角@APN}', '{角@CPM}']]}}, '96': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAP}', '{角@BAP}', '{角@CAM}', '{角@BAC}'], ['{角@ACD}', '{角@ACN}', '{角@ACQ}', '{角@DCP}', '{角@NCP}', '{角@PCQ}']]}}, '97': {'condjson': {'直线集合': [['{点@C}', '{点@B}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BCD}', '{角@BCN}', '{角@BCQ}'], ['{角@ABC}', '{角@CBM}']]}}, '98': {'condjson': {'直线集合': [['{点@C}', '{点@B}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BCD}', '{角@BCN}', '{角@BCQ}'], ['{角@ABC}', '{角@CBM}']]}}, '99': {'condjson': {'直线集合': [['{点@A}', '{点@D}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAM}', '{角@BAD}'], ['{角@ADN}', '{角@ADQ}', '{角@ADC}']]}}, '100': {'condjson': {'直线集合': [['{点@A}', '{点@D}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAM}', '{角@BAD}'], ['{角@ADN}', '{角@ADQ}', '{角@ADC}']]}}, '101': {'condjson': {'直线集合': [['{点@A}', '{点@Q}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAQ}']]}}, '102': {'condjson': {'直线集合': [['{点@A}', '{点@Q}']], '锐角集合': ['{角@DAQ}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAQ}']]}}, '103': {'condjson': {'直线集合': [['{点@A}', '{点@Q}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAQ}', '{角@BAQ}']]}}, '104': {'condjson': {'直线集合': [['{点@A}', '{点@Q}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAQ}', '{角@BAQ}'], ['{角@AQD}', '{角@AQN}'], ['{角@AQC}']]}}, '105': {'condjson': {'直线集合': [['{点@A}', '{点@Q}']], '锐角集合': ['{角@AQD}', '{角@AQN}', '{角@AQD}', '{角@AQN}', '{角@AQD}', '{角@AQN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@AQD}', '{角@AQN}', '{角@AQD}', '{角@AQN}', '{角@AQD}', '{角@AQN}']]}}, '106': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ANM}', '{角@ANP}']]}}, '107': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAN}']]}}, '108': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '锐角集合': ['{角@DAN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAN}']]}}, '109': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAN}', '{角@BAN}']]}}, '110': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ANM}', '{角@ANP}']]}}, '111': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAN}', '{角@BAN}'], ['{角@AND}'], ['{角@ANQ}', '{角@ANC}']]}}, '112': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '锐角集合': ['{角@AND}', '{角@AND}', '{角@AND}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@AND}', '{角@AND}', '{角@AND}']]}}, '113': {'condjson': {'直线集合': [['{点@B}', '{点@P}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBP}']]}}, '114': {'condjson': {'直线集合': [['{点@B}', '{点@P}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBP}']]}}, '115': {'condjson': {'直线集合': [['{点@B}', '{点@P}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABP}', '{角@MBP}']]}}, '116': {'condjson': {'直线集合': [['{点@B}', '{点@P}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBP}'], ['{角@BPM}'], ['{角@BPN}']]}}, '117': {'condjson': {'直线集合': [['{点@B}', '{点@P}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABP}', '{角@MBP}']]}}, '118': {'condjson': {'直线集合': [['{点@B}', '{点@Q}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBQ}']]}}, '119': {'condjson': {'直线集合': [['{点@B}', '{点@Q}']], '锐角集合': ['{角@CBQ}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBQ}']]}}, '120': {'condjson': {'直线集合': [['{点@B}', '{点@Q}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBQ}']]}}, '121': {'condjson': {'直线集合': [['{点@B}', '{点@Q}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABQ}', '{角@MBQ}']]}}, '122': {'condjson': {'直线集合': [['{点@B}', '{点@Q}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBQ}']]}}, '123': {'condjson': {'直线集合': [['{点@B}', '{点@Q}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABQ}', '{角@MBQ}'], ['{角@BQD}', '{角@BQN}'], ['{角@BQC}']]}}, '124': {'condjson': {'直线集合': [['{点@B}', '{点@Q}']], '锐角集合': ['{角@BQC}', '{角@BQC}', '{角@BQC}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BQC}', '{角@BQC}', '{角@BQC}']]}}, '125': {'condjson': {'直线集合': [['{点@B}', '{点@N}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBN}'], ['{角@BNM}', '{角@BNP}']]}}, '126': {'condjson': {'直线集合': [['{点@B}', '{点@N}']], '锐角集合': ['{角@CBN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBN}']]}}, '127': {'condjson': {'直线集合': [['{点@B}', '{点@N}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBN}']]}}, '128': {'condjson': {'直线集合': [['{点@B}', '{点@N}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABN}', '{角@MBN}']]}}, '129': {'condjson': {'直线集合': [['{点@B}', '{点@N}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBN}'], ['{角@BNM}', '{角@BNP}']]}}, '130': {'condjson': {'直线集合': [['{点@B}', '{点@N}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABN}', '{角@MBN}'], ['{角@BND}'], ['{角@BNQ}', '{角@BNC}']]}}, '131': {'condjson': {'直线集合': [['{点@B}', '{点@N}']], '锐角集合': ['{角@BNQ}', '{角@BNC}', '{角@BNQ}', '{角@BNC}', '{角@BNQ}', '{角@BNC}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BNQ}', '{角@BNC}', '{角@BNQ}', '{角@BNC}', '{角@BNQ}', '{角@BNC}']]}}, '132': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBD}']]}}, '133': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '锐角集合': ['{角@CBD}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBD}']]}}, '134': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBD}'], ['{角@ADB}']]}}, '135': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '锐角集合': ['{角@CBD}', '{角@ADB}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBD}', '{角@ADB}']]}}, '136': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABD}', '{角@DBM}'], ['{角@BDN}', '{角@BDQ}', '{角@BDC}']]}}, '137': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '锐角集合': ['{角@ABD}', '{角@DBM}', '{角@BDN}', '{角@BDQ}', '{角@BDC}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABD}', '{角@DBM}', '{角@BDN}', '{角@BDQ}', '{角@BDC}']]}}, '138': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBD}']]}}, '139': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABD}', '{角@DBM}'], ['{角@BDN}', '{角@BDQ}', '{角@BDC}']]}}, '140': {'condjson': {'直线集合': [['{点@P}', '{点@Q}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@NPQ}'], ['{角@MPQ}']]}}, '141': {'condjson': {'直线集合': [['{点@P}', '{点@Q}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DQP}', '{角@NQP}'], ['{角@CQP}']]}}, '142': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ADP}']]}}, '143': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@NDP}', '{角@PDQ}', '{角@CDP}']]}}, '144': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DPN}'], ['{角@DPM}']]}}, '145': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@NDP}', '{角@PDQ}', '{角@CDP}']]}}, '146': {'condjson': {'直线集合': [['{点@Q}', '{点@M}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@PMQ}', '{角@NMQ}']]}}, '147': {'condjson': {'直线集合': [['{点@Q}', '{点@M}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@PMQ}', '{角@NMQ}']]}}, '148': {'condjson': {'直线集合': [['{点@Q}', '{点@M}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CQM}'], ['{角@DQM}', '{角@MQN}'], ['{角@AMQ}'], ['{角@BMQ}']]}}, '149': {'condjson': {'直线集合': [['{点@M}', '{点@C}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CMP}', '{角@CMN}'], ['{角@BCM}']]}}, '150': {'condjson': {'直线集合': [['{点@M}', '{点@C}']], '锐角集合': ['{角@BCM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BCM}']]}}, '151': {'condjson': {'直线集合': [['{点@M}', '{点@C}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BCM}']]}}, '152': {'condjson': {'直线集合': [['{点@M}', '{点@C}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DCM}', '{角@MCN}', '{角@MCQ}']]}}, '153': {'condjson': {'直线集合': [['{点@M}', '{点@C}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CMP}', '{角@CMN}'], ['{角@BCM}']]}}, '154': {'condjson': {'直线集合': [['{点@M}', '{点@C}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BMC}'], ['{角@AMC}'], ['{角@DCM}', '{角@MCN}', '{角@MCQ}']]}}, '155': {'condjson': {'直线集合': [['{点@M}', '{点@C}']], '锐角集合': ['{角@BMC}', '{角@BMC}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BMC}', '{角@BMC}']]}}, '156': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '平行集合': [['{线段@BC}', '{线段@MN}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DMP}', '{角@DMN}']]}}, '157': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '平行集合': [['{线段@AD}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ADM}']]}}, '158': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '锐角集合': ['{角@ADM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ADM}']]}}, '159': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '平行集合': [['{线段@CD}', '{线段@AB}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MDN}', '{角@MDQ}', '{角@CDM}']]}}, '160': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '平行集合': [['{线段@BC}', '{线段@MN}', '{线段@NP}', '{线段@MP}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DMP}', '{角@DMN}']]}}, '161': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BMD}'], ['{角@AMD}'], ['{角@MDN}', '{角@MDQ}', '{角@CDM}']]}}, '162': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '锐角集合': ['{角@AMD}', '{角@AMD}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@AMD}', '{角@AMD}']]}}, '163': {'condjson': {'等值集合': [['{角@APM}', '{角@ACB}', '{角@BCP}'], ['{角@DAP}', '{角@ACB}', '{角@BCP}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAP}', '{角@APM}', '{角@ACB}', '{角@BCP}', '{角@CAD}']]}}, '164': {'condjson': {'等值集合': [['{角@CPN}', '{角@APM}'], ['{角@DAP}', '{角@APM}', '{角@ACB}', '{角@BCP}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAP}', '{角@CPN}', '{角@APM}', '{角@ACB}', '{角@BCP}', '{角@CAD}']]}}, '165': {'condjson': {'等值集合': [['{线段@BC}', '{线段@MN}'], ['{线段@AD}', '{线段@MN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@BC}', '{线段@AD}', '{线段@MN}']]}}, '166': {'condjson': {'等值集合': [['{线段@BC}', '{线段@AD}', '{线段@AB}', '{线段@CD}'], ['{线段@BC}', '{线段@AD}', '{线段@MN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@BC}', '{线段@CD}', '{线段@AB}', '{线段@AD}', '{线段@MN}']]}}, '167': {'condjson': {'默认集合': [0]}, 'points': ['@@平角相等'], 'outjson': {'等值集合': [['{角@APC}', '{角@CQD}', '{角@CQN}', '{角@DNQ}', '{角@MPN}', '{角@AMB}', '{角@CND}']]}}, '168': {'condjson': {'默认集合': [0]}, 'points': ['@@直角相等'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BCN}', '{角@ADN}', '{角@DNP}', '{角@ADQ}', '{角@AMN}', '{角@AMP}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@BPQ}', '{角@DNM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}', '{角@BAD}']]}}, '169': {'condjson': {'默认集合': [0]}, 'points': ['@@自等性质'], 'outjson': {'等值集合': [['{角@BMN}'], ['{角@AMC}'], ['{角@DCP}'], ['{角@CAD}'], ['{线段@CD}'], ['{角@BQD}'], ['{线段@AQ}'], ['{线段@NQ}'], ['{角@MQP}'], ['{角@ABQ}'], ['{线段@CN}'], ['{角@DAN}'], ['{角@DQP}'], ['{角@ANC}'], ['{角@DMP}'], ['{角@DAQ}'], ['{角@ABP}'], ['{角@BPD}'], ['{线段@PQ}'], ['{线段@AC}'], ['{角@DMQ}'], ['{角@AMD}'], ['{角@ACB}'], ['{角@BNC}'], ['{线段@BP}'], ['{角@AQM}'], ['{线段@AN}'], ['{线段@CP}'], ['{线段@BN}'], ['{角@ABC}'], ['{角@BCD}'], ['{角@CBD}'], ['{线段@AB}'], ['{角@NPQ}'], ['{角@BPC}'], ['{角@BPM}'], ['{角@DNM}'], ['{角@AQD}'], ['{角@ADC}'], ['{角@BMD}'], ['{角@ABD}'], ['{线段@DM}'], ['{角@APD}'], ['{角@AND}'], ['{线段@MQ}'], ['{线段@MN}'], ['{角@DCM}'], ['{角@BQN}'], ['{线段@BD}'], ['{角@BQM}'], ['{角@ADB}'], ['{角@DNP}'], ['{线段@BQ}'], ['{角@AMN}'], ['{角@AQN}'], ['{线段@AM}'], ['{线段@BM}'], ['{线段@BC}'], ['{线段@CQ}'], ['{角@BCM}'], ['{线段@AD}'], ['{角@BQC}'], ['{角@CBQ}'], ['{角@ACD}'], ['{角@APQ}'], ['{角@ABN}'], ['{角@CQP}'], ['{角@BND}'], ['{线段@DQ}'], ['{角@BAD}'], ['{角@ADM}'], ['{线段@NP}'], ['{角@AQC}'], ['{角@BDC}'], ['{角@BAC}'], ['{角@CMP}'], ['{线段@CM}'], ['{线段@AP}'], ['{角@NMQ}'], ['{角@APM}'], ['{角@BPQ}'], ['{线段@DN}'], ['{角@NQP}'], ['{线段@MP}'], ['{角@BMC}'], ['{角@BPN}'], ['{角@CNP}'], ['{角@CNM}'], ['{角@CMQ}'], ['{角@APN}'], ['{线段@DP}'], ['{角@CBN}'], ['{角@PMQ}'], ['{角@CPQ}'], ['{角@DPM}'], ['{角@ADP}'], ['{角@BCP}'], ['{角@ACN}'], ['{角@DBN}'], ['{角@MDP}'], ['{角@BMQ}'], ['{角@DQM}'], ['{角@BMP}'], ['{角@DAM}'], ['{角@MAP}'], ['{角@MCQ}'], ['{角@DAP}'], ['{角@BAN}'], ['{角@MPQ}'], ['{角@BDQ}'], ['{角@BNM}'], ['{角@PNQ}'], ['{角@MAN}'], ['{角@PDQ}'], ['{角@MDN}'], ['{角@NAQ}'], ['{角@AMQ}'], ['{角@CAN}'], ['{角@MCN}'], ['{角@BCQ}'], ['{角@MQN}'], ['{角@BQP}'], ['{角@DPN}'], ['{角@MBN}'], ['{角@DBM}'], ['{角@BCN}'], ['{角@MAQ}'], ['{角@ADQ}'], ['{角@BNQ}'], ['{角@BDM}'], ['{角@CPM}'], ['{角@CPN}'], ['{角@AMP}'], ['{角@BAQ}'], ['{角@ANQ}'], ['{角@BDP}'], ['{角@MCP}'], ['{角@CAM}'], ['{角@CDP}'], ['{角@MDQ}'], ['{角@AQB}'], ['{角@ADN}'], ['{角@NDP}'], ['{角@NBP}'], ['{角@DBQ}'], ['{角@MBQ}'], ['{角@CQM}'], ['{角@NCP}'], ['{角@DBP}'], ['{角@PBQ}'], ['{角@PAQ}'], ['{角@MNQ}'], ['{角@MBP}'], ['{角@CDM}'], ['{角@NBQ}'], ['{角@BNP}'], ['{角@APB}'], ['{角@ACQ}'], ['{角@ANB}'], ['{角@BDN}'], ['{角@AQP}'], ['{角@CAQ}'], ['{角@PCQ}'], ['{角@CBM}'], ['{角@CMN}'], ['{角@ANM}'], ['{角@ACM}'], ['{角@NAP}'], ['{角@CBP}'], ['{角@ANP}'], ['{角@CMD}'], ['{角@BAP}'], ['{角@DMN}'], ['{角@DPQ}'], ['{角@CPD}']]}}, '170': {'condjson': {'默认集合': [0]}, 'points': ['@@等值钝角传递'], 'outjson': {'锐角集合': ['{角@ADB}', '{角@CBD}', '{角@DAP}', '{角@CAD}', '{角@ACB}', '{角@BCP}', '{角@DAN}', '{角@BCM}', '{角@BQC}', '{角@DAQ}', '{角@CBQ}', '{角@ACN}', '{角@PCQ}', '{角@DCP}', '{角@ACQ}', '{角@ACD}', '{角@NCP}', '{角@BAC}', '{角@BAP}', '{角@CAM}', '{角@MAP}', '{角@AMD}', '{角@APM}', '{角@BNQ}', '{角@BNC}', '{角@ADM}', '{角@BDC}', '{角@BDN}', '{角@BDQ}', '{角@DBM}', '{角@ABD}', '{角@CPN}', '{角@AQD}', '{角@AQN}', '{角@BMC}', '{角@AND}', '{角@CBN}', '{角@MDN}', '{角@CDM}', '{角@MDQ}', '{角@MCN}', '{角@DCM}', '{角@MCQ}', '{角@ANP}', '{角@ANM}', '{角@CMP}', '{角@CMN}', '{角@DQM}', '{角@MQN}', '{角@DPN}', '{角@MBN}', '{角@ABN}', '{角@DMP}', '{角@DMN}', '{角@NDP}', '{角@PDQ}', '{角@CDP}', '{角@PMQ}', '{角@NMQ}', '{角@NPQ}', '{角@BPM}', '{角@DQP}', '{角@NQP}', '{角@BAN}', '{角@MAN}', '{角@BNM}', '{角@BNP}', '{角@ABP}', '{角@MBP}', '{角@ADP}', '{角@CBP}'], '钝角集合': ['{角@CPM}', '{角@APN}', '{角@ANQ}', '{角@ANC}', '{角@BQN}', '{角@BQD}', '{角@BMD}', '{角@AQC}', '{角@AMC}', '{角@BND}', '{角@BPN}', '{角@MPQ}', '{角@DPM}', '{角@CQM}', '{角@CQP}']}}, '171': {'condjson': {'默认集合': [0]}, 'points': ['@@角度角类型'], 'outjson': {'直角集合': ['{角@BPQ}', '{角@BMN}', '{角@DNP}', '{角@AMN}', '{角@AMP}', '{角@CBM}', '{角@BCQ}', '{角@BAD}', '{角@BCN}', '{角@ADN}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@DNM}', '{角@ADC}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}']}}, '172': {'condjson': {'直角三角形集合': ['{三角形@ADM}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@ADM}', '{角@AMD}']], '锐角集合': ['{角@ADM}', '{角@AMD}']}}, '173': {'condjson': {'直角三角形集合': ['{三角形@ACD}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@ACD}', '{角@CAD}']], '锐角集合': ['{角@ACD}', '{角@CAD}']}}, '174': {'condjson': {'直角三角形集合': ['{三角形@BCM}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@BCM}', '{角@BMC}']], '锐角集合': ['{角@BCM}', '{角@BMC}']}}, '175': {'condjson': {'直角三角形集合': ['{三角形@ABD}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@ABD}', '{角@ADB}']], '锐角集合': ['{角@ABD}', '{角@ADB}']}}, '176': {'condjson': {'直角三角形集合': ['{三角形@BCN}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@BNC}', '{角@CBN}']], '锐角集合': ['{角@BNC}', '{角@CBN}']}}, '177': {'condjson': {'直角三角形集合': ['{三角形@ADQ}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@AQD}', '{角@DAQ}']], '锐角集合': ['{角@AQD}', '{角@DAQ}']}}, '178': {'condjson': {'直角三角形集合': ['{三角形@ADN}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@AND}', '{角@DAN}']], '锐角集合': ['{角@AND}', '{角@DAN}']}}, '179': {'condjson': {'直角三角形集合': ['{三角形@ABC}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@ACB}', '{角@BAC}']], '锐角集合': ['{角@ACB}', '{角@BAC}']}}, '180': {'condjson': {'直角三角形集合': ['{三角形@BCD}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@BDC}', '{角@CBD}']], '锐角集合': ['{角@BDC}', '{角@CBD}']}}, '181': {'condjson': {'直角三角形集合': ['{三角形@BCQ}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@BQC}', '{角@CBQ}']], '锐角集合': ['{角@BQC}', '{角@CBQ}']}}, '182': {'condjson': {'表达式集合': ['1 8 0 ^ { \\circ } = {角@NPQ} + {角@BPQ} + {角@BPM}']}, 'points': ['@@表达式性质'], 'outjson': {'补角集合': [['{角@NPQ}', '{角@NPQ}']], '余角集合': [['{角@NPQ}', '{角@BPM}']]}}, '183': {'condjson': {'等值集合': [['{线段@CD}', '{线段@AB}'], ['{线段@CN}', '{线段@BM}']]}, 'points': ['@@表达式传递'], 'outjson': {'等值集合': [['{线段@AM}', '{线段@DN}']]}}, '184': {'condjson': {'等值集合': [['{线段@CD}', '{线段@AB}'], ['{线段@AM}', '{线段@DN}']]}, 'points': ['@@表达式传递'], 'outjson': {'等值集合': [['{线段@CN}', '{线段@BM}']]}}, '185': {'condjson': {'等值集合': [['{线段@CN}', '{线段@BM}'], ['{线段@AM}', '{线段@DN}']]}, 'points': ['@@表达式传递'], 'outjson': {'等值集合': [['{线段@CD}', '{线段@AB}']]}}, '186': {'condjson': {'补角集合': [['{角@MPQ}', '{角@NPQ}'], ['{角@NPQ}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@MPQ}']]}}, '187': {'condjson': {'补角集合': [['{角@CPM}', '{角@CPN}'], ['{角@APN}', '{角@CPN}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@CPM}', '{角@APN}']]}}, '188': {'condjson': {'补角集合': [['{角@APN}', '{角@APM}'], ['{角@APN}', '{角@CPN}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@CPN}', '{角@APM}']]}}, '189': {'condjson': {'补角集合': [['{角@CPM}', '{角@CPN}'], ['{角@CPM}', '{角@APM}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@CPN}', '{角@APM}']]}}, '190': {'condjson': {'补角集合': [['{角@APN}', '{角@APM}'], ['{角@CPM}', '{角@APM}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@CPM}', '{角@APN}']]}}, '191': {'condjson': {'补角集合': [['{角@DQM}', '{角@CQM}'], ['{角@MQN}', '{角@CQM}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@DQM}', '{角@MQN}']]}}, '192': {'condjson': {'补角集合': [['{角@CQP}', '{角@DQP}'], ['{角@CQP}', '{角@NQP}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@DQP}', '{角@NQP}']]}}, '193': {'condjson': {'补角集合': [['{角@BQD}', '{角@BQC}'], ['{角@BQN}', '{角@BQC}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@BQN}', '{角@BQD}']]}}, '194': {'condjson': {'补角集合': [['{角@AQD}', '{角@AQC}'], ['{角@AQN}', '{角@AQC}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@AQD}', '{角@AQN}']]}}, '195': {'condjson': {'补角集合': [['{角@DNM}', '{角@MNQ}'], ['{角@CNM}', '{角@DNM}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@CNM}', '{角@MNQ}']]}}, '196': {'condjson': {'补角集合': [['{角@PNQ}', '{角@DNP}'], ['{角@CNP}', '{角@DNP}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@PNQ}', '{角@CNP}']]}}, '197': {'condjson': {'补角集合': [['{角@BNQ}', '{角@BND}'], ['{角@BNC}', '{角@BND}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@BNQ}', '{角@BNC}']]}}, '198': {'condjson': {'补角集合': [['{角@ANQ}', '{角@AND}'], ['{角@AND}', '{角@ANC}']]}, 'points': ['@@补角等值反等传递'], 'outjson': {'等值集合': [['{角@ANQ}', '{角@ANC}']]}}, '199': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@DN}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DN}']]}}, '200': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@CN}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@CQ}']]}}, '201': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AB}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@CQ}']]}}, '202': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@NQ}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@CQ}']]}}, '203': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@CD}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CQ}']]}}, '204': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AM}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@CQ}']]}}, '205': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DQ}']]}}, '206': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@CN}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DN}']]}}, '207': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AB}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DN}']]}}, '208': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@NQ}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DN}']]}}, '209': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@CD}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DN}']]}}, '210': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AM}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DN}']]}}, '211': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@DN}', '{线段@DQ}']]}}, '212': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AB}'], ['{线段@AD}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AB}']]}}, '213': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@NQ}'], ['{线段@AD}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@NQ}']]}}, '214': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@CD}'], ['{线段@AD}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CN}']]}}, '215': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AM}'], ['{线段@AD}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AM}']]}}, '216': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DQ}']]}}, '217': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@NQ}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AB}']]}}, '218': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@CD}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AB}']]}}, '219': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AM}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@AM}']]}}, '220': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DQ}']]}}, '221': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@BM}']]}}, '222': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@AB}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@AD}']]}}, '223': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@CD}'], ['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}']]}}, '224': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AM}'], ['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AM}']]}}, '225': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DQ}']]}}, '226': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@AM}'], ['{线段@AD}', '{线段@CD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AM}']]}}, '227': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@CD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DQ}']]}}, '228': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@DQ}'], ['{线段@AD}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DQ}']]}}, '229': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DQ}']]}}, '230': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@BC}']]}}, '231': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@AB}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@BM}']]}}, '232': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@AM}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@AM}']]}}, '233': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CQ}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@CQ}']]}}, '234': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DN}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DN}']]}}, '235': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CN}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@BM}']]}}, '236': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@NQ}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@BM}']]}}, '237': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CD}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@BM}']]}}, '238': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DQ}']]}}, '239': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@MP}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '240': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@NP}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '241': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@MN}'], ['{线段@BC}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '242': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@CQ}']]}}, '243': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DN}']]}}, '244': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@BM}']]}}, '245': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@BM}']]}}, '246': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@CD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@BM}']]}}, '247': {'condjson': {'垂直集合': [['{线段@AD}', '{线段@BM}'], ['{线段@AD}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@BM}']]}}, '248': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@AM}'], ['{线段@AD}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@AD}']]}}, '249': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@AM}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@AM}']]}}, '250': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CQ}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@CQ}']]}}, '251': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DN}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DN}']]}}, '252': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CN}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AB}']]}}, '253': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CQ}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@AD}']]}}, '254': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DN}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@AD}']]}}, '255': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CN}'], ['{线段@AD}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@AD}']]}}, '256': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@NQ}'], ['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@AD}']]}}, '257': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CD}'], ['{线段@AD}', '{线段@CD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@AD}']]}}, '258': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@AD}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@AD}']]}}, '259': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CQ}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@CQ}']]}}, '260': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DN}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DN}']]}}, '261': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CN}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AM}']]}}, '262': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@NQ}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AM}']]}}, '263': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CD}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AM}']]}}, '264': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DN}'], ['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DN}']]}}, '265': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CN}'], ['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@CQ}']]}}, '266': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@NQ}'], ['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@CQ}']]}}, '267': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CD}'], ['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CQ}']]}}, '268': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DQ}']]}}, '269': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CN}'], ['{线段@BC}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DN}']]}}, '270': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@NQ}'], ['{线段@BC}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DN}']]}}, '271': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CD}'], ['{线段@BC}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DN}']]}}, '272': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@DN}', '{线段@DQ}']]}}, '273': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@NQ}'], ['{线段@BC}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@NQ}']]}}, '274': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CD}'], ['{线段@BC}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CN}']]}}, '275': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DQ}']]}}, '276': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@NQ}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AB}']]}}, '277': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CD}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AB}']]}}, '278': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DQ}']]}}, '279': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@MP}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '280': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@NP}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '281': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@MN}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '282': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@CD}'], ['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}']]}}, '283': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DQ}']]}}, '284': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@CD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DQ}']]}}, '285': {'condjson': {'垂直集合': [['{线段@BC}', '{线段@DQ}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DQ}']]}}, '286': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@MP}'], ['{线段@AD}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '287': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@NP}'], ['{线段@AD}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '288': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@MN}'], ['{线段@AD}', '{线段@BM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '289': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@NP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '290': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@MN}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '291': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@BM}']]}}, '292': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@AM}']]}}, '293': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@CQ}']]}}, '294': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@BM}']]}}, '295': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@BM}']]}}, '296': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@BM}']]}}, '297': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DQ}']]}}, '298': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DN}']]}}, '299': {'condjson': {'垂直集合': [['{线段@BM}', '{线段@MN}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '300': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@NP}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@BM}']]}}, '301': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@NP}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@AM}']]}}, '302': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@CQ}']]}}, '303': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@NP}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@BM}']]}}, '304': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@BM}']]}}, '305': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@BM}']]}}, '306': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DQ}']]}}, '307': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@BM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DN}']]}}, '308': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@BM}']]}}, '309': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@AM}']]}}, '310': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@CQ}']]}}, '311': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@BM}']]}}, '312': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@BM}']]}}, '313': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@BM}']]}}, '314': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DQ}']]}}, '315': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BM}', '{线段@DN}']]}}, '316': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MP}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '317': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@NP}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '318': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MN}'], ['{线段@BC}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '319': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@MP}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '320': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@NP}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '321': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@MN}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '322': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@NP}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '323': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@MN}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '324': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MP}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@AM}']]}}, '325': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MP}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@CQ}']]}}, '326': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MP}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AB}']]}}, '327': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MP}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AB}']]}}, '328': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MP}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AB}']]}}, '329': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DQ}']]}}, '330': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@AB}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DN}']]}}, '331': {'condjson': {'垂直集合': [['{线段@AB}', '{线段@MN}'], ['{线段@AB}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '332': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@NP}'], ['{线段@AB}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@AM}']]}}, '333': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}'], ['{线段@AB}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@CQ}']]}}, '334': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@NP}'], ['{线段@AB}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AB}']]}}, '335': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}'], ['{线段@AB}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AB}']]}}, '336': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@AB}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AB}']]}}, '337': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AB}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DQ}']]}}, '338': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@AB}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DN}']]}}, '339': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MN}'], ['{线段@AB}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@AM}']]}}, '340': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MN}'], ['{线段@AB}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@CQ}']]}}, '341': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MN}'], ['{线段@AB}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AB}']]}}, '342': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@AB}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AB}']]}}, '343': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@AB}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AB}']]}}, '344': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AB}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DQ}']]}}, '345': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@AB}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AB}', '{线段@DN}']]}}, '346': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MP}'], ['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '347': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}'], ['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '348': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MN}'], ['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '349': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MP}'], ['{线段@AD}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '350': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@NP}'], ['{线段@AD}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '351': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MN}'], ['{线段@AD}', '{线段@AM}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '352': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@NP}'], ['{线段@AM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '353': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MN}'], ['{线段@AM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '354': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@CQ}']]}}, '355': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AM}']]}}, '356': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AM}']]}}, '357': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AM}']]}}, '358': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DQ}']]}}, '359': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DN}']]}}, '360': {'condjson': {'垂直集合': [['{线段@AM}', '{线段@MN}'], ['{线段@AM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '361': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}'], ['{线段@AM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@CQ}']]}}, '362': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@NP}'], ['{线段@AM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AM}']]}}, '363': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}'], ['{线段@AM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AM}']]}}, '364': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@AM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AM}']]}}, '365': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DQ}']]}}, '366': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@AM}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DN}']]}}, '367': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@CQ}']]}}, '368': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AM}']]}}, '369': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@AM}']]}}, '370': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@AM}']]}}, '371': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DQ}']]}}, '372': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@DN}']]}}, '373': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MP}'], ['{线段@BC}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '374': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@NP}'], ['{线段@BC}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '375': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MN}'], ['{线段@BC}', '{线段@CN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '376': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MP}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '377': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '378': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MN}'], ['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '379': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}'], ['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '380': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MN}'], ['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '381': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MP}'], ['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@CQ}']]}}, '382': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MP}'], ['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@CQ}']]}}, '383': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MP}'], ['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CQ}']]}}, '384': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DQ}']]}}, '385': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DN}']]}}, '386': {'condjson': {'垂直集合': [['{线段@CQ}', '{线段@MN}'], ['{线段@NP}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '387': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@NP}'], ['{线段@NP}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@CQ}']]}}, '388': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}'], ['{线段@NP}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@CQ}']]}}, '389': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@NP}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CQ}']]}}, '390': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@NP}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DQ}']]}}, '391': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@NP}', '{线段@CQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DN}']]}}, '392': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MN}'], ['{线段@CQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@CQ}']]}}, '393': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@CQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@CQ}']]}}, '394': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@CQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CQ}']]}}, '395': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@CQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DQ}']]}}, '396': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@CQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CQ}', '{线段@DN}']]}}, '397': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MP}'], ['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '398': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}'], ['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '399': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '400': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MP}'], ['{线段@CN}', '{线段@AD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '401': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@NP}'], ['{线段@CN}', '{线段@AD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '402': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MN}'], ['{线段@CN}', '{线段@AD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '403': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@NP}'], ['{线段@CN}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '404': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MN}'], ['{线段@CN}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '405': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MP}'], ['{线段@CN}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@NQ}']]}}, '406': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MP}'], ['{线段@CN}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CN}']]}}, '407': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@CN}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DQ}']]}}, '408': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@CN}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DN}']]}}, '409': {'condjson': {'垂直集合': [['{线段@CN}', '{线段@MN}'], ['{线段@CN}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '410': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}'], ['{线段@CN}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@NQ}']]}}, '411': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@CN}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CN}']]}}, '412': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@CN}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DQ}']]}}, '413': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@CN}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DN}']]}}, '414': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@CN}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@NQ}']]}}, '415': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@CN}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CN}']]}}, '416': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@CN}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DQ}']]}}, '417': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@CN}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@DN}']]}}, '418': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MP}'], ['{线段@BC}', '{线段@CD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '419': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@BC}', '{线段@CD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '420': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@BC}', '{线段@CD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '421': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MP}'], ['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '422': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}'], ['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '423': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '424': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}'], ['{线段@NQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '425': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@NQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '426': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MP}'], ['{线段@NQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}']]}}, '427': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@NQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DQ}']]}}, '428': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@NQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DN}']]}}, '429': {'condjson': {'垂直集合': [['{线段@NQ}', '{线段@MN}'], ['{线段@NQ}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '430': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@NQ}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}']]}}, '431': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@NQ}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DQ}']]}}, '432': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@NQ}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DN}']]}}, '433': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@NQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}']]}}, '434': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@NQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DQ}']]}}, '435': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@NQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NQ}', '{线段@DN}']]}}, '436': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@BC}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '437': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@BC}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '438': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@BC}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '439': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MP}'], ['{线段@CD}', '{线段@AD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '440': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@CD}', '{线段@AD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '441': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@CD}', '{线段@AD}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '442': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@NP}'], ['{线段@CD}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '443': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@CD}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '444': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@CD}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DQ}']]}}, '445': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@CD}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DN}']]}}, '446': {'condjson': {'垂直集合': [['{线段@CD}', '{线段@MN}'], ['{线段@CD}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '447': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@CD}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DQ}']]}}, '448': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@CD}', '{线段@NP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DN}']]}}, '449': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@CD}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DQ}']]}}, '450': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@CD}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@DN}']]}}, '451': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@BC}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}']]}}, '452': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@BC}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}']]}}, '453': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@BC}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}']]}}, '454': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AD}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '455': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AD}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '456': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AD}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '457': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@DQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '458': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@DQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '459': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@DQ}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@DQ}', '{线段@DN}']]}}, '460': {'condjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@NP}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '461': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@NP}', '{线段@DQ}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@DQ}', '{线段@DN}']]}}, '462': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@DQ}', '{线段@MN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@DQ}', '{线段@DN}']]}}, '463': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MP}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MP}']]}}, '464': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@NP}']]}}, '465': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@AD}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@MN}']]}}, '466': {'condjson': {'垂直集合': [['{线段@NP}', '{线段@DN}'], ['{线段@DN}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MP}']]}}, '467': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@DN}', '{线段@MP}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@MN}']]}}, '468': {'condjson': {'垂直集合': [['{线段@DN}', '{线段@MN}'], ['{线段@NP}', '{线段@DN}']]}, 'points': ['@@垂直平行反等传递'], 'outjson': {'平行集合': [['{线段@NP}', '{线段@MN}']]}}, '469': {'condjson': {'平行集合': [['{线段@CD}', '{线段@NQ}'], ['{线段@CN}', '{线段@CD}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}', '{线段@CN}']]}}, '470': {'condjson': {'平行集合': [['{线段@CD}', '{线段@CQ}'], ['{线段@CD}', '{线段@NQ}', '{线段@CN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}', '{线段@CN}', '{线段@CQ}']]}}, '471': {'condjson': {'平行集合': [['{线段@CD}', '{线段@DN}'], ['{线段@CD}', '{线段@NQ}', '{线段@CN}', '{线段@CQ}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}', '{线段@CQ}', '{线段@CN}', '{线段@DN}']]}}, '472': {'condjson': {'平行集合': [['{线段@AD}', '{线段@MN}'], ['{线段@BC}', '{线段@AD}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@AD}', '{线段@BC}', '{线段@MN}']]}}, '473': {'condjson': {'平行集合': [['{线段@CD}', '{线段@DQ}'], ['{线段@CD}', '{线段@NQ}', '{线段@CQ}', '{线段@CN}', '{线段@DN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}', '{线段@CQ}', '{线段@CN}', '{线段@DN}', '{线段@DQ}']]}}, '474': {'condjson': {'平行集合': [['{线段@MP}', '{线段@MN}'], ['{线段@NP}', '{线段@MN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@MP}', '{线段@NP}', '{线段@MN}']]}}, '475': {'condjson': {'平行集合': [['{线段@AD}', '{线段@BC}', '{线段@MN}'], ['{线段@MP}', '{线段@NP}', '{线段@MN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}']]}}, '476': {'condjson': {'平行集合': [['{线段@CD}', '{线段@BM}'], ['{线段@CD}', '{线段@NQ}', '{线段@CQ}', '{线段@CN}', '{线段@DN}', '{线段@DQ}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@CD}', '{线段@NQ}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}}, '477': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AM}'], ['{线段@DQ}', '{线段@BM}', '{线段@CD}', '{线段@NQ}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@AM}', '{线段@BM}', '{线段@CD}', '{线段@NQ}', '{线段@CQ}', '{线段@CN}', '{线段@DN}', '{线段@DQ}']]}}, '478': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}'], ['{线段@AM}', '{线段@BM}', '{线段@CD}', '{线段@NQ}', '{线段@CQ}', '{线段@CN}', '{线段@DN}', '{线段@DQ}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@DQ}', '{线段@AM}', '{线段@BM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}}, '479': {'condjson': {'等值集合': [['{角@APM}', '{角@ACB}', '{角@BCP}'], ['{角@CPN}', '{角@APM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@CPN}', '{角@APM}', '{角@ACB}', '{角@BCP}']]}}, '480': {'condjson': {'等值集合': [['{角@CPN}', '{角@APM}', '{角@ACB}', '{角@BCP}'], ['{角@DAP}', '{角@ACB}', '{角@BCP}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAP}', '{角@BCP}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@ACB}']]}}, '481': {'condjson': {'等值集合': [['{角@DAM}', '{角@ADN}'], ['{线段@AD}', '{线段@AD}'], ['{线段@AM}', '{线段@DN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@ADN}']]}}, '482': {'condjson': {'等值集合': [['{线段@AB}', '{线段@CD}'], ['{角@BAC}', '{角@ACD}'], ['{角@ABC}', '{角@ADC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '483': {'condjson': {'等值集合': [['{角@ACB}', '{角@CAD}'], ['{线段@AC}', '{线段@AC}'], ['{线段@BC}', '{线段@AD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '484': {'condjson': {'等值集合': [['{角@ABC}', '{角@ADC}'], ['{线段@AB}', '{线段@AD}'], ['{线段@BC}', '{线段@CD}'], ['{线段@AB}', '{线段@CD}'], ['{线段@BC}', '{线段@AD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '485': {'condjson': {'等值集合': [['{线段@AC}', '{线段@AC}'], ['{角@BAC}', '{角@ACD}'], ['{角@ACB}', '{角@CAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '486': {'condjson': {'等值集合': [['{角@ABC}', '{角@BCD}'], ['{线段@AB}', '{线段@BC}'], ['{线段@BC}', '{线段@CD}'], ['{线段@AB}', '{线段@CD}'], ['{线段@BC}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '487': {'condjson': {'等值集合': [['{角@ABC}', '{角@BAD}'], ['{线段@AB}', '{线段@AB}'], ['{线段@BC}', '{线段@AD}'], ['{线段@AB}', '{线段@AD}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '488': {'condjson': {'等值集合': [['{线段@BC}', '{线段@AD}'], ['{角@ABC}', '{角@ADC}'], ['{角@ACB}', '{角@CAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '489': {'condjson': {'等值集合': [['{角@BAC}', '{角@ACD}'], ['{线段@AB}', '{线段@CD}'], ['{线段@AC}', '{线段@AC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '490': {'condjson': {'等值集合': [['{角@CBM}', '{角@BCN}'], ['{线段@BC}', '{线段@BC}'], ['{线段@BM}', '{线段@CN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BCN}']]}}, '491': {'condjson': {'等值集合': [['{角@ADC}', '{角@BCD}'], ['{线段@AD}', '{线段@BC}'], ['{线段@CD}', '{线段@CD}'], ['{线段@AD}', '{线段@CD}'], ['{线段@CD}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '492': {'condjson': {'等值集合': [['{角@ADC}', '{角@BAD}'], ['{线段@AD}', '{线段@AB}'], ['{线段@CD}', '{线段@AD}'], ['{线段@AD}', '{线段@AD}'], ['{线段@CD}', '{线段@AB}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '493': {'condjson': {'等值集合': [['{线段@BC}', '{线段@AD}'], ['{角@CBD}', '{角@ADB}'], ['{角@BCD}', '{角@BAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '494': {'condjson': {'等值集合': [['{角@BDC}', '{角@ABD}'], ['{线段@BD}', '{线段@BD}'], ['{线段@CD}', '{线段@AB}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '495': {'condjson': {'等值集合': [['{角@BCD}', '{角@BAD}'], ['{线段@BC}', '{线段@AB}'], ['{线段@CD}', '{线段@AD}'], ['{线段@BC}', '{线段@AD}'], ['{线段@CD}', '{线段@AB}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '496': {'condjson': {'等值集合': [['{线段@BD}', '{线段@BD}'], ['{角@CBD}', '{角@ADB}'], ['{角@BDC}', '{角@ABD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '497': {'condjson': {'等值集合': [['{线段@CD}', '{线段@AB}'], ['{角@BCD}', '{角@BAD}'], ['{角@BDC}', '{角@ABD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '498': {'condjson': {'等值集合': [['{角@CBD}', '{角@ADB}'], ['{线段@BC}', '{线段@AD}'], ['{线段@BD}', '{线段@BD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '499': {'condjson': {'全等三角形集合': [['{三角形@ABD}', '{三角形@ACD}'], ['{三角形@BCD}', '{三角形@ABD}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@ABD}', '{三角形@ACD}', '{三角形@BCD}']]}}, '500': {'condjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}'], ['{三角形@ABD}', '{三角形@ACD}', '{三角形@BCD}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@ABD}', '{三角形@ACD}', '{三角形@ABC}', '{三角形@BCD}']]}}, '501': {'condjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@ADM}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@DAN}', '{角@ADM}'], ['{线段@AM}', '{线段@DN}'], ['{角@AMD}', '{角@AND}'], ['{线段@AD}'], ['{线段@AN}', '{线段@DM}'], ['{角@DAM}', '{角@ADN}']]}}, '502': {'condjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@BAC}', '{角@ACB}', '{角@ACD}', '{角@CAD}'], ['{线段@BC}', '{线段@CD}', '{线段@AB}', '{线段@AD}'], ['{线段@AC}'], ['{角@ABC}', '{角@ADC}']]}}, '503': {'condjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@ADB}', '{角@ABD}', '{角@BAC}', '{角@ACB}'], ['{线段@BC}', '{线段@AD}', '{线段@AB}'], ['{线段@BD}', '{线段@AC}'], ['{角@ABC}', '{角@BAD}']]}}, '504': {'condjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@BDC}', '{角@BAC}', '{角@ACB}', '{角@CBD}'], ['{线段@BC}', '{线段@AB}', '{线段@CD}'], ['{线段@BD}', '{线段@AC}'], ['{角@ABC}', '{角@BCD}']]}}, '505': {'condjson': {'全等三角形集合': [['{三角形@ABD}', '{三角形@ACD}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@ADB}', '{角@ABD}', '{角@ACD}', '{角@CAD}'], ['{线段@AD}', '{线段@AB}', '{线段@CD}'], ['{线段@BD}', '{线段@AC}'], ['{角@ADC}', '{角@BAD}']]}}, '506': {'condjson': {'全等三角形集合': [['{三角形@ABD}', '{三角形@BCD}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@ADB}', '{角@ABD}', '{角@CBD}', '{角@BDC}'], ['{线段@CD}', '{线段@BC}', '{线段@AB}', '{线段@AD}'], ['{线段@BD}'], ['{角@BCD}', '{角@BAD}']]}}, '507': {'condjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ACD}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@BDC}', '{角@CBD}', '{角@CAD}', '{角@ACD}'], ['{线段@CD}', '{线段@BC}', '{线段@AD}'], ['{线段@BD}', '{线段@AC}'], ['{角@BCD}', '{角@ADC}']]}}, '508': {'condjson': {'全等三角形集合': [['{三角形@BCN}', '{三角形@BCM}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@BCM}', '{角@CBN}'], ['{线段@CN}', '{线段@BM}'], ['{角@BMC}', '{角@BNC}'], ['{线段@BC}'], ['{线段@CM}', '{线段@BN}'], ['{角@BCN}', '{角@CBM}']]}}, '509': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@CD}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@NP}']]}}, '510': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CD}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@NP}']]}}, '511': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@NP}']]}}, '512': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@AB}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@NP}']]}}, '513': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DN}']]}}, '514': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}']]}}, '515': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@NQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}']]}}, '516': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@CN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@NP}']]}}, '517': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}']]}}, '518': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DN}']]}}, '519': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}']]}}, '520': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@NQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}']]}}, '521': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CN}', '{线段@BC}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@NP}']]}}, '522': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}']]}}, '523': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@AM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@AM}']]}}, '524': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BC}', '{线段@BM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@BM}']]}}, '525': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@AM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@AM}']]}}, '526': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AD}', '{线段@BM}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@BM}']]}}, '527': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CD}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@NP}']]}}, '528': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CD}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}}, '529': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@CD}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@MN}']]}}, '530': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CD}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@MP}']]}}, '531': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CD}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AM}', '{线段@NP}']]}}, '532': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@CD}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@NP}']]}}, '533': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CD}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@NP}']]}}, '534': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CD}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}}, '535': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@CD}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@MP}']]}}, '536': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AB}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@NP}']]}}, '537': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AB}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}}, '538': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@AB}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@MN}']]}}, '539': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AB}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@MP}']]}}, '540': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AB}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AM}', '{线段@NP}']]}}, '541': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@AB}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@NP}']]}}, '542': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AB}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AB}', '{线段@NP}']]}}, '543': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AB}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}}, '544': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}']], '垂直集合': [['{线段@AB}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CD}', '{线段@MP}']]}}, '545': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@DN}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DN}']]}}, '546': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@DN}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}}, '547': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@NP}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MP}', '{线段@DN}']]}}, '548': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@NP}', '{线段@DN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AM}', '{线段@NP}']]}}, '549': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@DN}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DN}']]}}, '550': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@DN}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}}, '551': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@DQ}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}']]}}, '552': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@DQ}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BM}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}}, '553': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@NP}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MP}', '{线段@DQ}']]}}, '554': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@NP}', '{线段@DQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BM}', '{线段@NP}'], ['{线段@AM}', '{线段@NP}']]}}, '555': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@DQ}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}']]}}, '556': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@DQ}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BM}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}}, '557': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@NQ}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}']]}}, '558': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@NQ}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}}, '559': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@NQ}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NQ}', '{线段@MP}']]}}, '560': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@NQ}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AM}', '{线段@NP}']]}}, '561': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@NQ}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NQ}', '{线段@NP}']]}}, '562': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@NQ}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}}, '563': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CN}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@NP}']]}}, '564': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CN}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}}, '565': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CN}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@MP}']]}}, '566': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CN}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AM}', '{线段@NP}']]}}, '567': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CN}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@CN}', '{线段@NP}']]}}, '568': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CN}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}}, '569': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CQ}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}']]}}, '570': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CQ}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}}, '571': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@NP}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@MP}', '{线段@CQ}']]}}, '572': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@NP}', '{线段@CQ}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AM}', '{线段@NP}']]}}, '573': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@CQ}']]}}, '574': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@CQ}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}}, '575': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AM}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@AM}']]}}, '576': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AM}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@BM}', '{线段@MN}']]}}, '577': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AM}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@AM}', '{线段@MP}']]}}, '578': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AM}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@BM}', '{线段@NP}']]}}, '579': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@AM}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@AM}']]}}, '580': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@AM}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@BM}', '{线段@MP}']]}}, '581': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BM}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@BM}']]}}, '582': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BM}', '{线段@MN}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MN}'], ['{线段@AM}', '{线段@MN}']]}}, '583': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BM}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@BM}', '{线段@MP}']]}}, '584': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BM}', '{线段@NP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@DQ}'], ['{线段@AM}', '{线段@NP}']]}}, '585': {'condjson': {'平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']], '垂直集合': [['{线段@BM}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@NP}', '{线段@BM}']]}}, '586': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@BM}', '{线段@AM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']], '垂直集合': [['{线段@BM}', '{线段@MP}']]}, 'points': ['@@平行垂直等反传递'], 'outjson': {'垂直集合': [['{线段@DQ}', '{线段@MP}'], ['{线段@AM}', '{线段@MP}']]}}, '587': {'condjson': {'等值集合': [['{角@ABD}', '{角@BDC}'], ['{角@BDC}', '{角@ACD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ABD}', '{角@ACD}', '{角@BDC}']]}}, '588': {'condjson': {'等值集合': [['{角@ADB}', '{角@CBD}'], ['{角@CBD}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADB}', '{角@CBD}', '{角@CAD}']]}}, '589': {'condjson': {'等值集合': [['{角@BDC}', '{角@ADB}'], ['{角@ADB}', '{角@CBD}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADB}', '{角@BDC}', '{角@CBD}', '{角@CAD}']]}}, '590': {'condjson': {'等值集合': [['{角@ADB}', '{角@BDC}', '{角@CBD}', '{角@CAD}'], ['{角@ABD}', '{角@ACD}', '{角@BDC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@CBD}', '{角@CAD}', '{角@ACD}']]}}, '591': {'condjson': {'等值集合': [['{角@BAC}', '{角@ACD}'], ['{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@CBD}', '{角@CAD}', '{角@ACD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@BAC}', '{角@ACD}', '{角@CBD}', '{角@CAD}']]}}, '592': {'condjson': {'等值集合': [['{角@ACB}', '{角@CAD}'], ['{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@BAC}', '{角@ACD}', '{角@CBD}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@BAC}', '{角@ACD}', '{角@CAD}', '{角@CBD}', '{角@ACB}']]}}, '593': {'condjson': {'等值集合': [['{角@DBM}', '{角@ABD}', '{角@BDC}', '{角@BDQ}', '{角@BDN}'], ['{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@BAC}', '{角@ACD}', '{角@CAD}', '{角@CBD}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@BAC}', '{角@ACD}', '{角@BDQ}', '{角@CAD}', '{角@CBD}', '{角@BDN}', '{角@ACB}']]}}, '594': {'condjson': {'等值集合': [['{角@BNQ}', '{角@BNC}'], ['{角@BMC}', '{角@BNC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMC}', '{角@BNQ}', '{角@BNC}']]}}, '595': {'condjson': {'等值集合': [['{角@ACN}', '{角@PCQ}', '{角@DCP}', '{角@BAC}', '{角@BAP}', '{角@ACQ}', '{角@ACD}', '{角@CAM}', '{角@NCP}', '{角@MAP}'], ['{角@DBM}', '{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@BAC}', '{角@ACD}', '{角@BDQ}', '{角@CAD}', '{角@CBD}', '{角@BDN}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@ADB}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@CBD}', '{角@CAD}', '{角@BDN}', '{角@NCP}', '{角@MAP}', '{角@ACN}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}', '{角@ACB}']]}}, '596': {'condjson': {'等值集合': [['{角@DAP}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}'], ['{角@DBM}', '{角@ADB}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@CBD}', '{角@CAD}', '{角@BDN}', '{角@NCP}', '{角@MAP}', '{角@ACN}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@ADB}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@CBD}', '{角@CAD}', '{角@BDN}', '{角@APM}', '{角@BCP}', '{角@NCP}', '{角@MAP}', '{角@ACN}', '{角@DAP}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}', '{角@CPN}', '{角@ACB}']]}}, '597': {'condjson': {'直线集合': [['{点@D}', '{点@N}', '{点@Q}', '{点@C}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ADN}', '{角@ADQ}', '{角@ADC}'], ['{角@DNM}', '{角@DNP}'], ['{角@MNQ}', '{角@PNQ}', '{角@CNM}', '{角@CNP}'], ['{角@BCD}', '{角@BCN}', '{角@BCQ}']]}}, '598': {'condjson': {'直线集合': [['{点@A}', '{点@M}', '{点@B}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAM}', '{角@BAD}'], ['{角@BMP}', '{角@BMN}'], ['{角@AMP}', '{角@AMN}'], ['{角@ABC}', '{角@CBM}']]}}, '599': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAP}', '{角@CAD}'], ['{角@APM}', '{角@CPN}'], ['{角@CPM}', '{角@APN}'], ['{角@ACB}', '{角@BCP}']]}}, '600': {'condjson': {'直线集合': [['{点@A}', '{点@P}', '{点@C}']], '锐角集合': ['{角@DAP}', '{角@CAD}', '{角@APM}', '{角@CPN}', '{角@APM}', '{角@CPN}', '{角@ACB}', '{角@BCP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAP}', '{角@CAD}', '{角@APM}', '{角@CPN}', '{角@APM}', '{角@CPN}', '{角@ACB}', '{角@BCP}']]}}, '601': {'condjson': {'直线集合': [['{点@A}', '{点@Q}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAQ}']]}}, '602': {'condjson': {'直线集合': [['{点@A}', '{点@Q}']], '钝角集合': ['{角@AQC}', '{角@AQC}', '{角@AQC}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@AQC}', '{角@AQC}', '{角@AQC}']]}}, '603': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAN}'], ['{角@ANM}', '{角@ANP}']]}}, '604': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '锐角集合': ['{角@DAN}', '{角@ANM}', '{角@ANP}', '{角@ANM}', '{角@ANP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DAN}', '{角@ANM}', '{角@ANP}', '{角@ANM}', '{角@ANP}']]}}, '605': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '锐角集合': ['{角@MAN}', '{角@BAN}', '{角@MAN}', '{角@BAN}', '{角@AND}', '{角@AND}', '{角@AND}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAN}', '{角@BAN}', '{角@MAN}', '{角@BAN}', '{角@AND}', '{角@AND}', '{角@AND}']]}}, '606': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '钝角集合': ['{角@ANQ}', '{角@ANC}', '{角@ANQ}', '{角@ANC}', '{角@ANQ}', '{角@ANC}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ANQ}', '{角@ANC}', '{角@ANQ}', '{角@ANC}', '{角@ANQ}', '{角@ANC}']]}}, '607': {'condjson': {'直线集合': [['{点@A}', '{点@N}']], '锐角集合': ['{角@MAN}', '{角@BAN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MAN}', '{角@BAN}']]}}, '608': {'condjson': {'直线集合': [['{点@P}', '{点@B}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BPN}'], ['{角@BPM}'], ['{角@CBP}']]}}, '609': {'condjson': {'直线集合': [['{点@P}', '{点@B}']], '锐角集合': ['{角@BPM}', '{角@BPM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BPM}', '{角@BPM}']]}}, '610': {'condjson': {'直线集合': [['{点@P}', '{点@B}']], '锐角集合': ['{角@ABP}', '{角@MBP}', '{角@ABP}', '{角@MBP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABP}', '{角@MBP}', '{角@ABP}', '{角@MBP}']]}}, '611': {'condjson': {'直线集合': [['{点@Q}', '{点@B}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBQ}']]}}, '612': {'condjson': {'直线集合': [['{点@Q}', '{点@B}']], '钝角集合': ['{角@BQD}', '{角@BQN}', '{角@BQD}', '{角@BQN}', '{角@BQD}', '{角@BQN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BQD}', '{角@BQN}', '{角@BQD}', '{角@BQN}', '{角@BQD}', '{角@BQN}']]}}, '613': {'condjson': {'直线集合': [['{点@N}', '{点@B}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BNM}', '{角@BNP}'], ['{角@CBN}']]}}, '614': {'condjson': {'直线集合': [['{点@N}', '{点@B}']], '锐角集合': ['{角@BNM}', '{角@BNP}', '{角@BNM}', '{角@BNP}', '{角@CBN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BNM}', '{角@BNP}', '{角@BNM}', '{角@BNP}', '{角@CBN}']]}}, '615': {'condjson': {'直线集合': [['{点@N}', '{点@B}']], '锐角集合': ['{角@BNQ}', '{角@BNC}', '{角@BNQ}', '{角@BNC}', '{角@BNQ}', '{角@BNC}', '{角@ABN}', '{角@MBN}', '{角@ABN}', '{角@MBN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BNQ}', '{角@BNC}', '{角@BNQ}', '{角@BNC}', '{角@BNQ}', '{角@BNC}', '{角@ABN}', '{角@MBN}', '{角@ABN}', '{角@MBN}']]}}, '616': {'condjson': {'直线集合': [['{点@N}', '{点@B}']], '钝角集合': ['{角@BND}', '{角@BND}', '{角@BND}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BND}', '{角@BND}', '{角@BND}']]}}, '617': {'condjson': {'直线集合': [['{点@N}', '{点@B}']], '锐角集合': ['{角@ABN}', '{角@MBN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ABN}', '{角@MBN}']]}}, '618': {'condjson': {'直线集合': [['{点@B}', '{点@D}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBD}'], ['{角@ADB}']]}}, '619': {'condjson': {'直线集合': [['{点@Q}', '{点@P}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MPQ}'], ['{角@NPQ}']]}}, '620': {'condjson': {'直线集合': [['{点@Q}', '{点@P}']], '锐角集合': ['{角@NPQ}', '{角@NPQ}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@NPQ}', '{角@NPQ}']]}}, '621': {'condjson': {'直线集合': [['{点@Q}', '{点@P}']], '锐角集合': ['{角@DQP}', '{角@NQP}', '{角@DQP}', '{角@NQP}', '{角@DQP}', '{角@NQP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DQP}', '{角@NQP}', '{角@DQP}', '{角@NQP}', '{角@DQP}', '{角@NQP}']]}}, '622': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DPN}'], ['{角@DPM}'], ['{角@ADP}']]}}, '623': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '锐角集合': ['{角@DPN}', '{角@DPN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DPN}', '{角@DPN}']]}}, '624': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '锐角集合': ['{角@NDP}', '{角@PDQ}', '{角@CDP}', '{角@NDP}', '{角@PDQ}', '{角@CDP}', '{角@NDP}', '{角@PDQ}', '{角@CDP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@NDP}', '{角@PDQ}', '{角@CDP}', '{角@NDP}', '{角@PDQ}', '{角@CDP}', '{角@NDP}', '{角@PDQ}', '{角@CDP}']]}}, '625': {'condjson': {'直线集合': [['{点@Q}', '{点@M}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@PMQ}', '{角@NMQ}']]}}, '626': {'condjson': {'直线集合': [['{点@Q}', '{点@M}']], '锐角集合': ['{角@PMQ}', '{角@NMQ}', '{角@PMQ}', '{角@NMQ}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@PMQ}', '{角@NMQ}', '{角@PMQ}', '{角@NMQ}']]}}, '627': {'condjson': {'直线集合': [['{点@Q}', '{点@M}']], '锐角集合': ['{角@DQM}', '{角@MQN}', '{角@DQM}', '{角@MQN}', '{角@DQM}', '{角@MQN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DQM}', '{角@MQN}', '{角@DQM}', '{角@MQN}', '{角@DQM}', '{角@MQN}']]}}, '628': {'condjson': {'直线集合': [['{点@C}', '{点@M}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BCM}'], ['{角@CMP}', '{角@CMN}']]}}, '629': {'condjson': {'直线集合': [['{点@C}', '{点@M}']], '锐角集合': ['{角@BCM}', '{角@CMP}', '{角@CMN}', '{角@CMP}', '{角@CMN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BCM}', '{角@CMP}', '{角@CMN}', '{角@CMP}', '{角@CMN}']]}}, '630': {'condjson': {'直线集合': [['{点@C}', '{点@M}']], '锐角集合': ['{角@DCM}', '{角@MCN}', '{角@MCQ}', '{角@DCM}', '{角@MCN}', '{角@MCQ}', '{角@DCM}', '{角@MCN}', '{角@MCQ}', '{角@BMC}', '{角@BMC}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DCM}', '{角@MCN}', '{角@MCQ}', '{角@DCM}', '{角@MCN}', '{角@MCQ}', '{角@DCM}', '{角@MCN}', '{角@MCQ}', '{角@BMC}', '{角@BMC}']]}}, '631': {'condjson': {'直线集合': [['{点@C}', '{点@M}']], '钝角集合': ['{角@AMC}', '{角@AMC}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@AMC}', '{角@AMC}']]}}, '632': {'condjson': {'直线集合': [['{点@C}', '{点@M}']], '锐角集合': ['{角@DCM}', '{角@MCN}', '{角@MCQ}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DCM}', '{角@MCN}', '{角@MCQ}']]}}, '633': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '平行集合': [['{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}', '{线段@BC}']]}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DMP}', '{角@DMN}'], ['{角@ADM}']]}}, '634': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '锐角集合': ['{角@DMP}', '{角@DMN}', '{角@DMP}', '{角@DMN}', '{角@ADM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DMP}', '{角@DMN}', '{角@DMP}', '{角@DMN}', '{角@ADM}']]}}, '635': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '锐角集合': ['{角@AMD}', '{角@AMD}', '{角@MDN}', '{角@MDQ}', '{角@CDM}', '{角@MDN}', '{角@MDQ}', '{角@CDM}', '{角@MDN}', '{角@MDQ}', '{角@CDM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@AMD}', '{角@AMD}', '{角@MDN}', '{角@MDQ}', '{角@CDM}', '{角@MDN}', '{角@MDQ}', '{角@CDM}', '{角@MDN}', '{角@MDQ}', '{角@CDM}']]}}, '636': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '钝角集合': ['{角@BMD}', '{角@BMD}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BMD}', '{角@BMD}']]}}, '637': {'condjson': {'直线集合': [['{点@M}', '{点@D}']], '锐角集合': ['{角@MDN}', '{角@MDQ}', '{角@CDM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MDN}', '{角@MDQ}', '{角@CDM}']]}}, '638': {'condjson': {'等值集合': [['{角@AMD}', '{角@AND}'], ['{角@MDN}', '{角@AMD}', '{角@CDM}', '{角@MDQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MDN}', '{角@CDM}', '{角@AMD}', '{角@AND}', '{角@MDQ}']]}}, '639': {'condjson': {'等值集合': [['{角@DAN}', '{角@ADM}'], ['{角@ADM}', '{角@DMP}', '{角@DMN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAN}', '{角@ADM}', '{角@DMN}', '{角@DMP}']]}}, '640': {'condjson': {'等值集合': [['{角@BMC}', '{角@BNC}'], ['{角@MCN}', '{角@DCM}', '{角@BMC}', '{角@MCQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MCN}', '{角@DCM}', '{角@MCQ}', '{角@BMC}', '{角@BNC}']]}}, '641': {'condjson': {'等值集合': [['{角@BCM}', '{角@CBN}'], ['{角@CMP}', '{角@BCM}', '{角@CMN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@CMP}', '{角@BCM}', '{角@CMN}', '{角@CBN}']]}}, '642': {'condjson': {'等值集合': [['{角@BDC}', '{角@ACD}'], ['{角@DBM}', '{角@ABD}', '{角@BDC}', '{角@BDQ}', '{角@BDN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@ABD}', '{角@BDC}', '{角@BDN}', '{角@ACD}', '{角@BDQ}']]}}, '643': {'condjson': {'等值集合': [['{角@MCN}', '{角@DCM}', '{角@MCQ}', '{角@BMC}', '{角@BNC}'], ['{角@BNQ}', '{角@MBN}', '{角@ABN}', '{角@BNC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MCN}', '{角@MCQ}', '{角@MBN}', '{角@BMC}', '{角@DCM}', '{角@BNQ}', '{角@ABN}', '{角@BNC}']]}}, '644': {'condjson': {'等值集合': [['{角@CMP}', '{角@BCM}', '{角@CMN}', '{角@CBN}'], ['{角@BNM}', '{角@BNP}', '{角@CBN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@CMP}', '{角@BNM}', '{角@BCM}', '{角@BNP}', '{角@CMN}', '{角@CBN}']]}}, '645': {'condjson': {'等值集合': [['{角@MDN}', '{角@CDM}', '{角@AMD}', '{角@AND}', '{角@MDQ}'], ['{角@BAN}', '{角@AND}', '{角@MAN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MDN}', '{角@CDM}', '{角@MAN}', '{角@BAN}', '{角@AMD}', '{角@AND}', '{角@MDQ}']]}}, '646': {'condjson': {'等值集合': [['{角@DAN}', '{角@ADM}', '{角@DMN}', '{角@DMP}'], ['{角@DAN}', '{角@ANP}', '{角@ANM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAN}', '{角@ADM}', '{角@ANP}', '{角@DMN}', '{角@DMP}', '{角@ANM}']]}}, '647': {'condjson': {'等值集合': [['{角@DBM}', '{角@ABD}', '{角@BDC}', '{角@BDN}', '{角@ACD}', '{角@BDQ}'], ['{角@ACN}', '{角@PCQ}', '{角@DCP}', '{角@BAC}', '{角@BAP}', '{角@ACQ}', '{角@ACD}', '{角@CAM}', '{角@NCP}', '{角@MAP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@BDN}', '{角@NCP}', '{角@MAP}', '{角@ACN}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}']]}}, '648': {'condjson': {'等值集合': [['{角@ADB}', '{角@CBD}', '{角@CAD}'], ['{角@DAP}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAP}', '{角@ADB}', '{角@CBD}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}']]}}, '649': {'condjson': {'等值集合': [['{角@ABC}', '{角@ADC}'], ['{角@ABC}', '{角@CBM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ABC}', '{角@ADC}', '{角@CBM}']]}}, '650': {'condjson': {'等值集合': [['{角@BCD}', '{角@BAD}'], ['{角@DAM}', '{角@BAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAM}', '{角@BCD}', '{角@BAD}']]}}, '651': {'condjson': {'等值集合': [['{角@DAM}', '{角@BCD}', '{角@BAD}'], ['{角@BCQ}', '{角@BCN}', '{角@BCD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BCN}', '{角@BCQ}', '{角@DAM}', '{角@BCD}', '{角@BAD}']]}}, '652': {'condjson': {'等值集合': [['{角@ABC}', '{角@ADC}', '{角@CBM}'], ['{角@ADQ}', '{角@ADN}', '{角@ADC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADN}', '{角@ADC}', '{角@CBM}', '{角@ADQ}', '{角@ABC}']]}}, '653': {'condjson': {'等值集合': [['{线段@AD}', '{线段@BC}'], ['{线段@BC}', '{线段@MN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@AD}', '{线段@BC}', '{线段@MN}']]}}, '654': {'condjson': {'等值集合': [['{角@BDC}', '{角@ADB}'], ['{角@DAP}', '{角@ADB}', '{角@CBD}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAP}', '{角@ADB}', '{角@BDC}', '{角@CBD}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}']]}}, '655': {'condjson': {'等值集合': [['{角@DAP}', '{角@ADB}', '{角@BDC}', '{角@CBD}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}'], ['{角@DBM}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@BDN}', '{角@NCP}', '{角@MAP}', '{角@ACN}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@ADB}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@CBD}', '{角@CAD}', '{角@BDN}', '{角@APM}', '{角@BCP}', '{角@NCP}', '{角@MAP}', '{角@ACN}', '{角@DAP}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}', '{角@CPN}', '{角@ACB}']]}}, '656': {'condjson': {'直角三角形集合': ['{三角形@AMN}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@ANM}', '{角@MAN}']], '锐角集合': ['{角@ANM}', '{角@MAN}']}}, '657': {'condjson': {'直角三角形集合': ['{三角形@MNQ}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@MQN}', '{角@NMQ}']], '锐角集合': ['{角@MQN}', '{角@NMQ}']}}, '658': {'condjson': {'直角三角形集合': ['{三角形@AMP}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@APM}', '{角@MAP}']], '锐角集合': ['{角@APM}', '{角@MAP}']}}, '659': {'condjson': {'直角三角形集合': ['{三角形@BMN}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@BNM}', '{角@MBN}']], '锐角集合': ['{角@BNM}', '{角@MBN}']}}, '660': {'condjson': {'直角三角形集合': ['{三角形@DNP}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@DPN}', '{角@NDP}']], '锐角集合': ['{角@DPN}', '{角@NDP}']}}, '661': {'condjson': {'直角三角形集合': ['{三角形@DMN}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@DMN}', '{角@MDN}']], '锐角集合': ['{角@DMN}', '{角@MDN}']}}, '662': {'condjson': {'直角三角形集合': ['{三角形@CNP}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@CPN}', '{角@NCP}']], '锐角集合': ['{角@CPN}', '{角@NCP}']}}, '663': {'condjson': {'直角三角形集合': ['{三角形@CMN}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@CMN}', '{角@MCN}']], '锐角集合': ['{角@CMN}', '{角@MCN}']}}, '664': {'condjson': {'直角三角形集合': ['{三角形@BMP}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@BPM}', '{角@MBP}']], '锐角集合': ['{角@BPM}', '{角@MBP}']}}, '665': {'condjson': {'直角三角形集合': ['{三角形@NPQ}']}, 'points': ['@@直角三角形属性必要条件'], 'outjson': {'余角集合': [['{角@NPQ}', '{角@NQP}']], '锐角集合': ['{角@NPQ}', '{角@NQP}']}}, '666': {'condjson': {'等值集合': [['{角@BPQ}', '9 0 ^ { \\circ }'], ['{角@BMN}', '{角@DNP}', '{角@AMN}', '{角@AMP}', '{角@CBM}', '{角@BCQ}', '{角@BAD}', '{角@BCN}', '{角@ADN}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@BPQ}', '{角@DNM}', '{角@ADC}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BCN}', '{角@ADN}', '{角@DNP}', '{角@ADQ}', '{角@AMN}', '{角@AMP}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@BPQ}', '9 0 ^ { \\circ }', '{角@DNM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}', '{角@BAD}']]}}, '667': {'condjson': {'余角集合': [['{角@NPQ}', '{角@NQP}'], ['{角@BPM}', '{角@NPQ}']]}, 'points': ['@@余角等值反等传递'], 'outjson': {'等值集合': [['{角@BPM}', '{角@NQP}']]}}, '668': {'condjson': {'余角集合': [['{角@BPM}', '{角@MBP}'], ['{角@BPM}', '{角@NPQ}']]}, 'points': ['@@余角等值反等传递'], 'outjson': {'等值集合': [['{角@NPQ}', '{角@MBP}']]}}, '669': {'condjson': {'平行集合': [['{线段@DQ}', '{线段@CQ}'], ['{线段@DQ}', '{线段@DN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@DQ}', '{线段@DN}', '{线段@CQ}']]}}, '670': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MP}'], ['{线段@MN}', '{线段@MP}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@MP}']]}}, '671': {'condjson': {'平行集合': [['{线段@BC}', '{线段@MN}', '{线段@MP}'], ['{线段@BC}', '{线段@AD}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@MP}', '{线段@AD}', '{线段@MN}']]}}, '672': {'condjson': {'平行集合': [['{线段@CD}', '{线段@DN}'], ['{线段@DQ}', '{线段@DN}', '{线段@CQ}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@CQ}', '{线段@DQ}', '{线段@DN}']]}}, '673': {'condjson': {'平行集合': [['{线段@AD}', '{线段@NP}'], ['{线段@BC}', '{线段@MP}', '{线段@AD}', '{线段@MN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@BC}', '{线段@NP}', '{线段@MP}', '{线段@AD}', '{线段@MN}']]}}, '674': {'condjson': {'平行集合': [['{线段@CN}', '{线段@BM}'], ['{线段@BM}', '{线段@AM}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@CN}', '{线段@AM}', '{线段@BM}']]}}, '675': {'condjson': {'平行集合': [['{线段@CD}', '{线段@AB}'], ['{线段@CD}', '{线段@NQ}', '{线段@CN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}']]}}, '676': {'condjson': {'平行集合': [['{线段@CD}', '{线段@CQ}', '{线段@DQ}', '{线段@DN}'], ['{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@DQ}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CQ}', '{线段@CN}', '{线段@DN}']]}}, '677': {'condjson': {'平行集合': [['{线段@CN}', '{线段@AM}', '{线段@BM}'], ['{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@CQ}', '{线段@DQ}', '{线段@DN}']]}, 'points': ['@@平行间传递'], 'outjson': {'平行集合': [['{线段@DQ}', '{线段@AM}', '{线段@BM}', '{线段@CD}', '{线段@NQ}', '{线段@AB}', '{线段@CN}', '{线段@DN}', '{线段@CQ}']]}}, '678': {'condjson': {'等值集合': [['{角@BPM}', '{角@NQP}'], ['{角@DQP}', '{角@NQP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BPM}', '{角@DQP}', '{角@NQP}']]}}, '679': {'condjson': {'等值集合': [['{角@ABP}', '{角@MBP}'], ['{角@NPQ}', '{角@MBP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ABP}', '{角@NPQ}', '{角@MBP}']]}}, '680': {'condjson': {'等值集合': [['{线段@CD}', '{线段@AB}'], ['{角@DCM}', '{角@ABN}'], ['{角@CDM}', '{角@BAN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CDM}', '{三角形@ABN}']]}}, '681': {'condjson': {'等值集合': [['{角@CDM}', '{角@BAN}'], ['{线段@CD}', '{线段@AB}'], ['{线段@DM}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CDM}', '{三角形@ABN}']]}}, '682': {'condjson': {'等值集合': [['{角@DCM}', '{角@ABN}'], ['{线段@CD}', '{线段@AB}'], ['{线段@CM}', '{线段@BN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CDM}', '{三角形@ABN}']]}}, '683': {'condjson': {'等值集合': [['{角@BDN}', '{角@CAM}'], ['{线段@BD}', '{线段@AC}'], ['{线段@DN}', '{线段@AM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BDN}', '{三角形@ACM}']]}}, '684': {'condjson': {'等值集合': [['{线段@AD}', '{线段@MN}'], ['{角@DAM}', '{角@DNM}'], ['{角@ADM}', '{角@DMN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@DMN}']]}}, '685': {'condjson': {'等值集合': [['{角@AMD}', '{角@MDN}'], ['{线段@AM}', '{线段@DN}'], ['{线段@DM}', '{线段@DM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@DMN}']]}}, '686': {'condjson': {'等值集合': [['{线段@AD}', '{线段@AD}'], ['{角@DAM}', '{角@ADN}'], ['{角@ADM}', '{角@DAN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@ADN}']]}}, '687': {'condjson': {'等值集合': [['{角@AMD}', '{角@AND}'], ['{线段@AM}', '{线段@DN}'], ['{线段@DM}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@ADN}']]}}, '688': {'condjson': {'等值集合': [['{线段@AD}', '{线段@MN}'], ['{角@DAM}', '{角@AMN}'], ['{角@ADM}', '{角@ANM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@AMN}']]}}, '689': {'condjson': {'等值集合': [['{角@AMD}', '{角@MAN}'], ['{线段@AM}', '{线段@AM}'], ['{线段@DM}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@AMN}']]}}, '690': {'condjson': {'等值集合': [['{线段@AM}', '{线段@DN}'], ['{角@DAM}', '{角@DNM}'], ['{角@AMD}', '{角@MDN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@DMN}']]}}, '691': {'condjson': {'等值集合': [['{角@ADM}', '{角@DMN}'], ['{线段@AD}', '{线段@MN}'], ['{线段@DM}', '{线段@DM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@DMN}']]}}, '692': {'condjson': {'等值集合': [['{线段@AM}', '{线段@DN}'], ['{角@DAM}', '{角@ADN}'], ['{角@AMD}', '{角@AND}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@ADN}']]}}, '693': {'condjson': {'等值集合': [['{角@ADM}', '{角@DAN}'], ['{线段@AD}', '{线段@AD}'], ['{线段@DM}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@ADN}']]}}, '694': {'condjson': {'等值集合': [['{线段@AM}', '{线段@AM}'], ['{角@DAM}', '{角@AMN}'], ['{角@AMD}', '{角@MAN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@AMN}']]}}, '695': {'condjson': {'等值集合': [['{角@ADM}', '{角@ANM}'], ['{线段@AD}', '{线段@MN}'], ['{线段@DM}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@AMN}']]}}, '696': {'condjson': {'等值集合': [['{角@DAM}', '{角@DNM}'], ['{线段@AD}', '{线段@MN}'], ['{线段@AM}', '{线段@DN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@DMN}']]}}, '697': {'condjson': {'等值集合': [['{线段@DM}', '{线段@DM}'], ['{角@ADM}', '{角@DMN}'], ['{角@AMD}', '{角@MDN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@DMN}']]}}, '698': {'condjson': {'等值集合': [['{线段@DM}', '{线段@AN}'], ['{角@ADM}', '{角@DAN}'], ['{角@AMD}', '{角@AND}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@ADN}']]}}, '699': {'condjson': {'等值集合': [['{角@DAM}', '{角@AMN}'], ['{线段@AD}', '{线段@MN}'], ['{线段@AM}', '{线段@AM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@AMN}']]}}, '700': {'condjson': {'等值集合': [['{线段@DM}', '{线段@AN}'], ['{角@ADM}', '{角@ANM}'], ['{角@AMD}', '{角@MAN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@AMN}']]}}, '701': {'condjson': {'等值集合': [['{角@CNM}', '{角@CBM}'], ['{线段@CN}', '{线段@BM}'], ['{线段@MN}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCM}']]}}, '702': {'condjson': {'等值集合': [['{线段@CM}', '{线段@CM}'], ['{角@MCN}', '{角@BMC}'], ['{角@CMN}', '{角@BCM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCM}']]}}, '703': {'condjson': {'等值集合': [['{角@CNM}', '{角@BMN}'], ['{线段@CN}', '{线段@BM}'], ['{线段@MN}', '{线段@MN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BMN}']]}}, '704': {'condjson': {'等值集合': [['{线段@CM}', '{线段@BN}'], ['{角@MCN}', '{角@MBN}'], ['{角@CMN}', '{角@BNM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BMN}']]}}, '705': {'condjson': {'等值集合': [['{角@CNM}', '{角@BCN}'], ['{线段@CN}', '{线段@CN}'], ['{线段@MN}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCN}']]}}, '706': {'condjson': {'等值集合': [['{线段@CM}', '{线段@BN}'], ['{角@MCN}', '{角@BNC}'], ['{角@CMN}', '{角@CBN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCN}']]}}, '707': {'condjson': {'等值集合': [['{线段@CN}', '{线段@BM}'], ['{角@MCN}', '{角@BMC}'], ['{角@CNM}', '{角@CBM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCM}']]}}, '708': {'condjson': {'等值集合': [['{角@CMN}', '{角@BCM}'], ['{线段@CM}', '{线段@CM}'], ['{线段@MN}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCM}']]}}, '709': {'condjson': {'等值集合': [['{线段@CN}', '{线段@BM}'], ['{角@MCN}', '{角@MBN}'], ['{角@CNM}', '{角@BMN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BMN}']]}}, '710': {'condjson': {'等值集合': [['{角@CMN}', '{角@BNM}'], ['{线段@CM}', '{线段@BN}'], ['{线段@MN}', '{线段@MN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BMN}']]}}, '711': {'condjson': {'等值集合': [['{线段@CN}', '{线段@CN}'], ['{角@MCN}', '{角@BNC}'], ['{角@CNM}', '{角@BCN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCN}']]}}, '712': {'condjson': {'等值集合': [['{角@CMN}', '{角@CBN}'], ['{线段@CM}', '{线段@BN}'], ['{线段@MN}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCN}']]}}, '713': {'condjson': {'等值集合': [['{线段@MN}', '{线段@BC}'], ['{角@CMN}', '{角@BCM}'], ['{角@CNM}', '{角@CBM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCM}']]}}, '714': {'condjson': {'等值集合': [['{角@MCN}', '{角@BMC}'], ['{线段@CM}', '{线段@CM}'], ['{线段@CN}', '{线段@BM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCM}']]}}, '715': {'condjson': {'等值集合': [['{线段@MN}', '{线段@MN}'], ['{角@CMN}', '{角@BNM}'], ['{角@CNM}', '{角@BMN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BMN}']]}}, '716': {'condjson': {'等值集合': [['{角@MCN}', '{角@MBN}'], ['{线段@CM}', '{线段@BN}'], ['{线段@CN}', '{线段@BM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BMN}']]}}, '717': {'condjson': {'等值集合': [['{线段@MN}', '{线段@BC}'], ['{角@CMN}', '{角@CBN}'], ['{角@CNM}', '{角@BCN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCN}']]}}, '718': {'condjson': {'等值集合': [['{角@MCN}', '{角@BNC}'], ['{线段@CM}', '{线段@BN}'], ['{线段@CN}', '{线段@CN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCN}']]}}, '719': {'condjson': {'等值集合': [['{线段@AB}', '{线段@AD}'], ['{角@BAC}', '{角@CAD}'], ['{角@ABC}', '{角@ADC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '720': {'condjson': {'等值集合': [['{角@ACB}', '{角@ACD}'], ['{线段@AC}', '{线段@AC}'], ['{线段@BC}', '{线段@CD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '721': {'condjson': {'等值集合': [['{线段@AB}', '{线段@BC}'], ['{角@BAC}', '{角@CBD}'], ['{角@ABC}', '{角@BCD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '722': {'condjson': {'等值集合': [['{角@ACB}', '{角@BDC}'], ['{线段@AC}', '{线段@BD}'], ['{线段@BC}', '{线段@CD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '723': {'condjson': {'等值集合': [['{线段@AB}', '{线段@CD}'], ['{角@BAC}', '{角@BDC}'], ['{角@ABC}', '{角@BCD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '724': {'condjson': {'等值集合': [['{角@ACB}', '{角@CBD}'], ['{线段@AC}', '{线段@BD}'], ['{线段@BC}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '725': {'condjson': {'等值集合': [['{线段@AB}', '{线段@AB}'], ['{角@BAC}', '{角@ABD}'], ['{角@ABC}', '{角@BAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '726': {'condjson': {'等值集合': [['{角@ACB}', '{角@ADB}'], ['{线段@AC}', '{线段@BD}'], ['{线段@BC}', '{线段@AD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '727': {'condjson': {'等值集合': [['{线段@AB}', '{线段@AD}'], ['{角@BAC}', '{角@ADB}'], ['{角@ABC}', '{角@BAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '728': {'condjson': {'等值集合': [['{角@ACB}', '{角@ABD}'], ['{线段@AC}', '{线段@BD}'], ['{线段@BC}', '{线段@AB}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '729': {'condjson': {'等值集合': [['{线段@AC}', '{线段@AC}'], ['{角@BAC}', '{角@CAD}'], ['{角@ACB}', '{角@ACD}'], ['{角@BAC}', '{角@ACD}'], ['{角@ACB}', '{角@CAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '730': {'condjson': {'等值集合': [['{线段@AC}', '{线段@BD}'], ['{角@BAC}', '{角@CBD}'], ['{角@ACB}', '{角@BDC}'], ['{角@BAC}', '{角@BDC}'], ['{角@ACB}', '{角@CBD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '731': {'condjson': {'等值集合': [['{线段@AC}', '{线段@BD}'], ['{角@BAC}', '{角@ABD}'], ['{角@ACB}', '{角@ADB}'], ['{角@BAC}', '{角@ADB}'], ['{角@ACB}', '{角@ABD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '732': {'condjson': {'等值集合': [['{线段@BC}', '{线段@CD}'], ['{角@ABC}', '{角@ADC}'], ['{角@ACB}', '{角@ACD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '733': {'condjson': {'等值集合': [['{角@BAC}', '{角@CAD}'], ['{线段@AB}', '{线段@AD}'], ['{线段@AC}', '{线段@AC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ACD}']]}}, '734': {'condjson': {'等值集合': [['{线段@BC}', '{线段@BC}'], ['{角@ABC}', '{角@BCD}'], ['{角@ACB}', '{角@CBD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '735': {'condjson': {'等值集合': [['{角@BAC}', '{角@BDC}'], ['{线段@AB}', '{线段@CD}'], ['{线段@AC}', '{线段@BD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '736': {'condjson': {'等值集合': [['{线段@BC}', '{线段@CD}'], ['{角@ABC}', '{角@BCD}'], ['{角@ACB}', '{角@BDC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '737': {'condjson': {'等值集合': [['{角@BAC}', '{角@CBD}'], ['{线段@AB}', '{线段@BC}'], ['{线段@AC}', '{线段@BD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@BCD}']]}}, '738': {'condjson': {'等值集合': [['{线段@BC}', '{线段@AB}'], ['{角@ABC}', '{角@BAD}'], ['{角@ACB}', '{角@ABD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '739': {'condjson': {'等值集合': [['{角@BAC}', '{角@ADB}'], ['{线段@AB}', '{线段@AD}'], ['{线段@AC}', '{线段@BD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '740': {'condjson': {'等值集合': [['{线段@BC}', '{线段@AD}'], ['{角@ABC}', '{角@BAD}'], ['{角@ACB}', '{角@ADB}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '741': {'condjson': {'等值集合': [['{角@BAC}', '{角@ABD}'], ['{线段@AB}', '{线段@AB}'], ['{线段@AC}', '{线段@BD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}']]}}, '742': {'condjson': {'等值集合': [['{角@DBM}', '{角@ACN}'], ['{线段@BD}', '{线段@AC}'], ['{线段@BM}', '{线段@CN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BDM}', '{三角形@ACN}']]}}, '743': {'condjson': {'等值集合': [['{角@DAP}', '{角@BAP}'], ['{线段@AD}', '{线段@AB}'], ['{线段@AP}', '{线段@AP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADP}', '{三角形@ABP}']]}}, '744': {'condjson': {'等值集合': [['{角@DNM}', '{角@ADN}'], ['{线段@DN}', '{线段@DN}'], ['{线段@MN}', '{线段@AD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@ADN}']]}}, '745': {'condjson': {'等值集合': [['{线段@DM}', '{线段@AN}'], ['{角@MDN}', '{角@AND}'], ['{角@DMN}', '{角@DAN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@ADN}']]}}, '746': {'condjson': {'等值集合': [['{角@DNM}', '{角@AMN}'], ['{线段@DN}', '{线段@AM}'], ['{线段@MN}', '{线段@MN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}']]}}, '747': {'condjson': {'等值集合': [['{线段@DM}', '{线段@AN}'], ['{角@MDN}', '{角@MAN}'], ['{角@DMN}', '{角@ANM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}']]}}, '748': {'condjson': {'等值集合': [['{线段@DN}', '{线段@DN}'], ['{角@MDN}', '{角@AND}'], ['{角@DNM}', '{角@ADN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@ADN}']]}}, '749': {'condjson': {'等值集合': [['{角@DMN}', '{角@DAN}'], ['{线段@DM}', '{线段@AN}'], ['{线段@MN}', '{线段@AD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@ADN}']]}}, '750': {'condjson': {'等值集合': [['{线段@DN}', '{线段@AM}'], ['{角@MDN}', '{角@MAN}'], ['{角@DNM}', '{角@AMN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}']]}}, '751': {'condjson': {'等值集合': [['{角@DMN}', '{角@ANM}'], ['{线段@DM}', '{线段@AN}'], ['{线段@MN}', '{线段@MN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}']]}}, '752': {'condjson': {'等值集合': [['{线段@MN}', '{线段@AD}'], ['{角@DMN}', '{角@DAN}'], ['{角@DNM}', '{角@ADN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@ADN}']]}}, '753': {'condjson': {'等值集合': [['{角@MDN}', '{角@AND}'], ['{线段@DM}', '{线段@AN}'], ['{线段@DN}', '{线段@DN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@ADN}']]}}, '754': {'condjson': {'等值集合': [['{线段@MN}', '{线段@MN}'], ['{角@DMN}', '{角@ANM}'], ['{角@DNM}', '{角@AMN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}']]}}, '755': {'condjson': {'等值集合': [['{角@MDN}', '{角@MAN}'], ['{线段@DM}', '{线段@AN}'], ['{线段@DN}', '{线段@AM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}']]}}, '756': {'condjson': {'等值集合': [['{线段@BC}', '{线段@MN}'], ['{角@CBM}', '{角@BMN}'], ['{角@BCM}', '{角@BNM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BMN}']]}}, '757': {'condjson': {'等值集合': [['{角@BMC}', '{角@MBN}'], ['{线段@BM}', '{线段@BM}'], ['{线段@CM}', '{线段@BN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BMN}']]}}, '758': {'condjson': {'等值集合': [['{线段@BC}', '{线段@BC}'], ['{角@CBM}', '{角@BCN}'], ['{角@BCM}', '{角@CBN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BCN}']]}}, '759': {'condjson': {'等值集合': [['{角@BMC}', '{角@BNC}'], ['{线段@BM}', '{线段@CN}'], ['{线段@CM}', '{线段@BN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BCN}']]}}, '760': {'condjson': {'等值集合': [['{线段@BM}', '{线段@BM}'], ['{角@CBM}', '{角@BMN}'], ['{角@BMC}', '{角@MBN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BMN}']]}}, '761': {'condjson': {'等值集合': [['{角@BCM}', '{角@BNM}'], ['{线段@BC}', '{线段@MN}'], ['{线段@CM}', '{线段@BN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BMN}']]}}, '762': {'condjson': {'等值集合': [['{线段@BM}', '{线段@CN}'], ['{角@CBM}', '{角@BCN}'], ['{角@BMC}', '{角@BNC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BCN}']]}}, '763': {'condjson': {'等值集合': [['{角@BCM}', '{角@CBN}'], ['{线段@BC}', '{线段@BC}'], ['{线段@CM}', '{线段@BN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BCN}']]}}, '764': {'condjson': {'等值集合': [['{角@CBM}', '{角@BMN}'], ['{线段@BC}', '{线段@MN}'], ['{线段@BM}', '{线段@BM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BMN}']]}}, '765': {'condjson': {'等值集合': [['{线段@CM}', '{线段@BN}'], ['{角@BCM}', '{角@BNM}'], ['{角@BMC}', '{角@MBN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BMN}']]}}, '766': {'condjson': {'等值集合': [['{线段@CM}', '{线段@BN}'], ['{角@BCM}', '{角@CBN}'], ['{角@BMC}', '{角@BNC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCM}', '{三角形@BCN}']]}}, '767': {'condjson': {'等值集合': [['{线段@AC}', '{线段@BD}'], ['{角@CAD}', '{角@CBD}'], ['{角@ACD}', '{角@BDC}'], ['{角@CAD}', '{角@BDC}'], ['{角@ACD}', '{角@CBD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '768': {'condjson': {'等值集合': [['{线段@AC}', '{线段@BD}'], ['{角@CAD}', '{角@ABD}'], ['{角@ACD}', '{角@ADB}'], ['{角@CAD}', '{角@ADB}'], ['{角@ACD}', '{角@ABD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '769': {'condjson': {'等值集合': [['{线段@AD}', '{线段@BC}'], ['{角@CAD}', '{角@CBD}'], ['{角@ADC}', '{角@BCD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '770': {'condjson': {'等值集合': [['{角@ACD}', '{角@BDC}'], ['{线段@AC}', '{线段@BD}'], ['{线段@CD}', '{线段@CD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '771': {'condjson': {'等值集合': [['{线段@AD}', '{线段@CD}'], ['{角@CAD}', '{角@BDC}'], ['{角@ADC}', '{角@BCD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '772': {'condjson': {'等值集合': [['{角@ACD}', '{角@CBD}'], ['{线段@AC}', '{线段@BD}'], ['{线段@CD}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '773': {'condjson': {'等值集合': [['{线段@AD}', '{线段@AB}'], ['{角@CAD}', '{角@ABD}'], ['{角@ADC}', '{角@BAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '774': {'condjson': {'等值集合': [['{角@ACD}', '{角@ADB}'], ['{线段@AC}', '{线段@BD}'], ['{线段@CD}', '{线段@AD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '775': {'condjson': {'等值集合': [['{线段@AD}', '{线段@AD}'], ['{角@CAD}', '{角@ADB}'], ['{角@ADC}', '{角@BAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '776': {'condjson': {'等值集合': [['{角@ACD}', '{角@ABD}'], ['{线段@AC}', '{线段@BD}'], ['{线段@CD}', '{线段@AB}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '777': {'condjson': {'等值集合': [['{线段@CD}', '{线段@BC}'], ['{角@ACD}', '{角@CBD}'], ['{角@ADC}', '{角@BCD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '778': {'condjson': {'等值集合': [['{角@CAD}', '{角@BDC}'], ['{线段@AC}', '{线段@BD}'], ['{线段@AD}', '{线段@CD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '779': {'condjson': {'等值集合': [['{线段@CD}', '{线段@CD}'], ['{角@ACD}', '{角@BDC}'], ['{角@ADC}', '{角@BCD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '780': {'condjson': {'等值集合': [['{角@CAD}', '{角@CBD}'], ['{线段@AC}', '{线段@BD}'], ['{线段@AD}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@BCD}']]}}, '781': {'condjson': {'等值集合': [['{线段@CD}', '{线段@AB}'], ['{角@ACD}', '{角@ABD}'], ['{角@ADC}', '{角@BAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '782': {'condjson': {'等值集合': [['{角@CAD}', '{角@ADB}'], ['{线段@AC}', '{线段@BD}'], ['{线段@AD}', '{线段@AD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '783': {'condjson': {'等值集合': [['{线段@CD}', '{线段@AD}'], ['{角@ACD}', '{角@ADB}'], ['{角@ADC}', '{角@BAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '784': {'condjson': {'等值集合': [['{角@CAD}', '{角@ABD}'], ['{线段@AC}', '{线段@BD}'], ['{线段@AD}', '{线段@AB}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ACD}', '{三角形@ABD}']]}}, '785': {'condjson': {'等值集合': [['{线段@BM}', '{线段@CN}'], ['{角@MBN}', '{角@BNC}'], ['{角@BMN}', '{角@BCN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}']]}}, '786': {'condjson': {'等值集合': [['{角@BNM}', '{角@CBN}'], ['{线段@BN}', '{线段@BN}'], ['{线段@MN}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}']]}}, '787': {'condjson': {'等值集合': [['{角@BMN}', '{角@BCN}'], ['{线段@BM}', '{线段@CN}'], ['{线段@MN}', '{线段@BC}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}']]}}, '788': {'condjson': {'等值集合': [['{线段@BN}', '{线段@BN}'], ['{角@MBN}', '{角@BNC}'], ['{角@BNM}', '{角@CBN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}']]}}, '789': {'condjson': {'等值集合': [['{线段@MN}', '{线段@BC}'], ['{角@BMN}', '{角@BCN}'], ['{角@BNM}', '{角@CBN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}']]}}, '790': {'condjson': {'等值集合': [['{角@MBN}', '{角@BNC}'], ['{线段@BM}', '{线段@CN}'], ['{线段@BN}', '{线段@BN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}']]}}, '791': {'condjson': {'等值集合': [['{线段@AD}', '{线段@MN}'], ['{角@DAN}', '{角@ANM}'], ['{角@ADN}', '{角@AMN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@AMN}']]}}, '792': {'condjson': {'等值集合': [['{角@AND}', '{角@MAN}'], ['{线段@AN}', '{线段@AN}'], ['{线段@DN}', '{线段@AM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@AMN}']]}}, '793': {'condjson': {'等值集合': [['{角@ADN}', '{角@AMN}'], ['{线段@AD}', '{线段@MN}'], ['{线段@DN}', '{线段@AM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@AMN}']]}}, '794': {'condjson': {'等值集合': [['{线段@AN}', '{线段@AN}'], ['{角@DAN}', '{角@ANM}'], ['{角@AND}', '{角@MAN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@AMN}']]}}, '795': {'condjson': {'等值集合': [['{线段@DN}', '{线段@AM}'], ['{角@ADN}', '{角@AMN}'], ['{角@AND}', '{角@MAN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@AMN}']]}}, '796': {'condjson': {'等值集合': [['{角@DAN}', '{角@ANM}'], ['{线段@AD}', '{线段@MN}'], ['{线段@AN}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@AMN}']]}}, '797': {'condjson': {'等值集合': [['{角@BCP}', '{角@DCP}'], ['{线段@BC}', '{线段@CD}'], ['{线段@CP}', '{线段@CP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCP}', '{三角形@CDP}']]}}, '798': {'condjson': {'等值集合': [['{线段@BC}', '{线段@AB}'], ['{角@CBD}', '{角@ABD}'], ['{角@BCD}', '{角@BAD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '799': {'condjson': {'等值集合': [['{角@BDC}', '{角@ADB}'], ['{线段@BD}', '{线段@BD}'], ['{线段@CD}', '{线段@AD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '800': {'condjson': {'等值集合': [['{线段@BD}', '{线段@BD}'], ['{角@CBD}', '{角@ABD}'], ['{角@BDC}', '{角@ADB}'], ['{角@CBD}', '{角@ADB}'], ['{角@BDC}', '{角@ABD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '801': {'condjson': {'等值集合': [['{线段@CD}', '{线段@AD}'], ['{角@BCD}', '{角@BAD}'], ['{角@BDC}', '{角@ADB}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '802': {'condjson': {'等值集合': [['{角@CBD}', '{角@ABD}'], ['{线段@BC}', '{线段@AB}'], ['{线段@BD}', '{线段@BD}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCD}', '{三角形@ABD}']]}}, '803': {'condjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}'], ['{三角形@ADN}', '{三角形@AMN}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}', '{三角形@ADN}']]}}, '804': {'condjson': {'全等三角形集合': [['{三角形@BCN}', '{三角形@BCM}'], ['{三角形@BMN}', '{三角形@BCN}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}', '{三角形@BCM}']]}}, '805': {'condjson': {'全等三角形集合': [['{三角形@ABC}', '{三角形@ABD}'], ['{三角形@ABD}', '{三角形@ACD}', '{三角形@BCD}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@ABD}', '{三角形@ACD}', '{三角形@ABC}', '{三角形@BCD}']]}}, '806': {'condjson': {'全等三角形集合': [['{三角形@BCN}', '{三角形@CMN}'], ['{三角形@BMN}', '{三角形@BCN}', '{三角形@BCM}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}', '{三角形@BCM}', '{三角形@CMN}']]}}, '807': {'condjson': {'全等三角形集合': [['{三角形@AMN}', '{三角形@ADM}'], ['{三角形@DMN}', '{三角形@AMN}', '{三角形@ADN}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@AMN}', '{三角形@DMN}', '{三角形@ADN}']]}}, '808': {'condjson': {'全等三角形集合': [['{三角形@AMN}', '{三角形@DMN}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@MDN}', '{角@MAN}'], ['{线段@MN}'], ['{线段@AN}', '{线段@DM}'], ['{角@AMN}', '{角@DNM}'], ['{线段@AM}', '{线段@DN}'], ['{角@DMN}', '{角@ANM}']]}}, '809': {'condjson': {'全等三角形集合': [['{三角形@AMN}', '{三角形@ADM}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@AMD}', '{角@MAN}'], ['{线段@AD}', '{线段@MN}'], ['{角@AMN}', '{角@DAM}'], ['{线段@AN}', '{线段@DM}'], ['{线段@AM}'], ['{角@ADM}', '{角@ANM}']]}}, '810': {'condjson': {'全等三角形集合': [['{三角形@AMN}', '{三角形@ADN}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@AND}', '{角@MAN}'], ['{线段@AD}', '{线段@MN}'], ['{线段@AN}'], ['{角@AMN}', '{角@ADN}'], ['{线段@AM}', '{线段@DN}'], ['{角@DAN}', '{角@ANM}']]}}, '811': {'condjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@ADM}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@MDN}', '{角@AMD}'], ['{线段@AD}', '{线段@MN}'], ['{线段@AM}', '{线段@DN}'], ['{角@ADM}', '{角@DMN}'], ['{线段@DM}'], ['{角@DAM}', '{角@DNM}']]}}, '812': {'condjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@ADN}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@MDN}', '{角@AND}'], ['{线段@AD}', '{线段@MN}'], ['{角@DAN}', '{角@DMN}'], ['{线段@DN}'], ['{线段@AN}', '{线段@DM}'], ['{角@ADN}', '{角@DNM}']]}}, '813': {'condjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCN}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@BNC}', '{角@MBN}'], ['{线段@BC}', '{线段@MN}'], ['{线段@BN}'], ['{角@BMN}', '{角@BCN}'], ['{线段@CN}', '{线段@BM}'], ['{角@BNM}', '{角@CBN}']]}}, '814': {'condjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@CMN}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@MCN}', '{角@MBN}'], ['{线段@MN}'], ['{线段@CM}', '{线段@BN}'], ['{角@BMN}', '{角@CNM}'], ['{线段@CN}', '{线段@BM}'], ['{角@BNM}', '{角@CMN}']]}}, '815': {'condjson': {'全等三角形集合': [['{三角形@BMN}', '{三角形@BCM}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@BMC}', '{角@MBN}'], ['{线段@BC}', '{线段@MN}'], ['{角@BMN}', '{角@CBM}'], ['{线段@CM}', '{线段@BN}'], ['{线段@BM}'], ['{角@BNM}', '{角@BCM}']]}}, '816': {'condjson': {'全等三角形集合': [['{三角形@BCN}', '{三角形@CMN}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@CMN}', '{角@CBN}'], ['{线段@CN}'], ['{线段@CM}', '{线段@BN}'], ['{角@BCN}', '{角@CNM}'], ['{线段@BC}', '{线段@MN}'], ['{角@MCN}', '{角@BNC}']]}}, '817': {'condjson': {'全等三角形集合': [['{三角形@CMN}', '{三角形@BCM}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@MCN}', '{角@BMC}'], ['{线段@BC}', '{线段@MN}'], ['{线段@CN}', '{线段@BM}'], ['{角@CMN}', '{角@BCM}'], ['{线段@CM}'], ['{角@CNM}', '{角@CBM}']]}}, '818': {'condjson': {'全等三角形集合': [['{三角形@CDM}', '{三角形@ABN}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@DCM}', '{角@ABN}'], ['{线段@AN}', '{线段@DM}'], ['{角@ANB}', '{角@CMD}'], ['{线段@CD}', '{线段@AB}'], ['{线段@CM}', '{线段@BN}'], ['{角@BAN}', '{角@CDM}']]}}, '819': {'condjson': {'全等三角形集合': [['{三角形@ACM}', '{三角形@BDN}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@DBN}', '{角@ACM}'], ['{线段@AM}', '{线段@DN}'], ['{角@AMC}', '{角@BND}'], ['{线段@BD}', '{线段@AC}'], ['{线段@CM}', '{线段@BN}'], ['{角@BDN}', '{角@CAM}']]}}, '820': {'condjson': {'全等三角形集合': [['{三角形@ACN}', '{三角形@BDM}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@CAN}', '{角@BDM}'], ['{线段@CN}', '{线段@BM}'], ['{角@BMD}', '{角@ANC}'], ['{线段@BD}', '{线段@AC}'], ['{线段@AN}', '{线段@DM}'], ['{角@DBM}', '{角@ACN}']]}}, '821': {'condjson': {'全等三角形集合': [['{三角形@ADP}', '{三角形@ABP}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@ABP}', '{角@ADP}'], ['{线段@AP}'], ['{角@APD}', '{角@APB}'], ['{线段@AD}', '{线段@AB}'], ['{线段@DP}', '{线段@BP}'], ['{角@DAP}', '{角@BAP}']]}}, '822': {'condjson': {'全等三角形集合': [['{三角形@BCP}', '{三角形@CDP}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@CBP}', '{角@CDP}'], ['{线段@CP}'], ['{角@BPC}', '{角@CPD}'], ['{线段@BC}', '{线段@CD}'], ['{线段@DP}', '{线段@BP}'], ['{角@DCP}', '{角@BCP}']]}}, '823': {'condjson': {'等值集合': [['{线段@BC}', '{线段@MN}'], ['{线段@BC}', '{线段@AD}', '{线段@AB}', '{线段@CD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@BC}', '{线段@CD}', '{线段@AB}', '{线段@AD}', '{线段@MN}']]}}, '824': {'condjson': {'等值集合': [['{角@NDP}', '{角@PDQ}', '{角@CDP}'], ['{角@CBP}', '{角@CDP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@CBP}', '{角@PDQ}', '{角@NDP}', '{角@CDP}']]}}, '825': {'condjson': {'等值集合': [['{角@BCN}', '{角@CBM}'], ['{角@CNM}', '{角@CBM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BCN}', '{角@CNM}', '{角@CBM}']]}}, '826': {'condjson': {'等值集合': [['{角@BCM}', '{角@CBN}'], ['{角@CMN}', '{角@BCM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@CMN}', '{角@BCM}', '{角@CBN}']]}}, '827': {'condjson': {'等值集合': [['{角@BMC}', '{角@BNC}'], ['{角@MCN}', '{角@BMC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMC}', '{角@BNC}', '{角@MCN}']]}}, '828': {'condjson': {'等值集合': [['{角@BMC}', '{角@MBN}'], ['{角@BMC}', '{角@BNC}', '{角@MCN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MCN}', '{角@BNC}', '{角@BMC}', '{角@MBN}']]}}, '829': {'condjson': {'等值集合': [['{角@BMN}', '{角@CBM}'], ['{角@BCN}', '{角@CNM}', '{角@CBM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BCN}', '{角@CBM}', '{角@CNM}']]}}, '830': {'condjson': {'等值集合': [['{角@BNM}', '{角@BCM}'], ['{角@CMN}', '{角@BCM}', '{角@CBN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BNM}', '{角@BCM}', '{角@CMN}', '{角@CBN}']]}}, '831': {'condjson': {'等值集合': [['{角@BDC}', '{角@CAD}'], ['{角@CBD}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BDC}', '{角@CBD}', '{角@CAD}']]}}, '832': {'condjson': {'等值集合': [['{角@BDC}', '{角@CBD}', '{角@CAD}'], ['{角@BDC}', '{角@ACD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BDC}', '{角@CBD}', '{角@CAD}', '{角@ACD}']]}}, '833': {'condjson': {'等值集合': [['{角@BCD}', '{角@BAD}'], ['{角@BCD}', '{角@ADC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BCD}', '{角@ADC}', '{角@BAD}']]}}, '834': {'condjson': {'等值集合': [['{角@ABD}', '{角@CBD}'], ['{角@BDC}', '{角@CBD}', '{角@CAD}', '{角@ACD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ACD}', '{角@ABD}', '{角@CBD}', '{角@CAD}', '{角@BDC}']]}}, '835': {'condjson': {'等值集合': [['{角@BDC}', '{角@ADB}'], ['{角@ACD}', '{角@ABD}', '{角@CBD}', '{角@CAD}', '{角@BDC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADB}', '{角@ABD}', '{角@CBD}', '{角@BDC}', '{角@ACD}', '{角@CAD}']]}}, '836': {'condjson': {'等值集合': [['{角@ABC}', '{角@BCD}'], ['{角@BCD}', '{角@ADC}', '{角@BAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADC}', '{角@BCD}', '{角@ABC}', '{角@BAD}']]}}, '837': {'condjson': {'等值集合': [['{角@BAC}', '{角@CBD}'], ['{角@ADB}', '{角@ABD}', '{角@CBD}', '{角@BDC}', '{角@ACD}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@BAC}', '{角@CBD}', '{角@ACD}', '{角@CAD}']]}}, '838': {'condjson': {'等值集合': [['{角@BDC}', '{角@ACB}'], ['{角@ADB}', '{角@ABD}', '{角@BDC}', '{角@BAC}', '{角@CBD}', '{角@ACD}', '{角@CAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADB}', '{角@BDC}', '{角@ABD}', '{角@BAC}', '{角@CBD}', '{角@ACD}', '{角@CAD}', '{角@ACB}']]}}, '839': {'condjson': {'等值集合': [['{角@ADN}', '{角@DNM}'], ['{角@DAM}', '{角@ADN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAM}', '{角@ADN}', '{角@DNM}']]}}, '840': {'condjson': {'等值集合': [['{角@MDN}', '{角@AND}'], ['{角@AMD}', '{角@AND}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MDN}', '{角@AMD}', '{角@AND}']]}}, '841': {'condjson': {'等值集合': [['{角@DAN}', '{角@DMN}'], ['{角@DAN}', '{角@ADM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAN}', '{角@ADM}', '{角@DMN}']]}}, '842': {'condjson': {'等值集合': [['{角@AMN}', '{角@ADN}'], ['{角@DAM}', '{角@ADN}', '{角@DNM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADN}', '{角@DAM}', '{角@AMN}', '{角@DNM}']]}}, '843': {'condjson': {'等值集合': [['{角@DAN}', '{角@ANM}'], ['{角@DAN}', '{角@ADM}', '{角@DMN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAN}', '{角@ADM}', '{角@DMN}', '{角@ANM}']]}}, '844': {'condjson': {'等值集合': [['{角@AND}', '{角@MAN}'], ['{角@MDN}', '{角@AMD}', '{角@AND}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MDN}', '{角@MAN}', '{角@AMD}', '{角@AND}']]}}, '845': {'condjson': {'等值集合': [['{角@ANQ}', '{角@ANC}'], ['{角@BMD}', '{角@ANC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ANQ}', '{角@BMD}', '{角@ANC}']]}}, '846': {'condjson': {'等值集合': [['{角@ABP}', '{角@NPQ}', '{角@MBP}'], ['{角@ABP}', '{角@ADP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADP}', '{角@NPQ}', '{角@ABP}', '{角@MBP}']]}}, '847': {'condjson': {'直线集合': [['{点@P}', '{点@B}']], '钝角集合': ['{角@BPN}', '{角@BPN}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BPN}', '{角@BPN}']]}}, '848': {'condjson': {'直线集合': [['{点@Q}', '{点@P}']], '钝角集合': ['{角@MPQ}', '{角@MPQ}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@MPQ}', '{角@MPQ}']]}}, '849': {'condjson': {'直线集合': [['{点@Q}', '{点@P}']], '钝角集合': ['{角@CQP}', '{角@CQP}', '{角@CQP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CQP}', '{角@CQP}', '{角@CQP}']]}}, '850': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '钝角集合': ['{角@DPM}', '{角@DPM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DPM}', '{角@DPM}']]}}, '851': {'condjson': {'直线集合': [['{点@Q}', '{点@M}']], '钝角集合': ['{角@CQM}', '{角@CQM}', '{角@CQM}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CQM}', '{角@CQM}', '{角@CQM}']]}}, '852': {'condjson': {'等值集合': [['{角@BAN}', '{角@CDM}'], ['{角@MDN}', '{角@AMD}', '{角@CDM}', '{角@MDQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MDN}', '{角@CDM}', '{角@BAN}', '{角@AMD}', '{角@MDQ}']]}}, '853': {'condjson': {'等值集合': [['{角@DCM}', '{角@ABN}'], ['{角@MCN}', '{角@DCM}', '{角@BMC}', '{角@MCQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MCN}', '{角@DCM}', '{角@MCQ}', '{角@ABN}', '{角@BMC}']]}}, '854': {'condjson': {'等值集合': [['{角@MCN}', '{角@DCM}', '{角@MCQ}', '{角@ABN}', '{角@BMC}'], ['{角@BNQ}', '{角@MBN}', '{角@ABN}', '{角@BNC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MCN}', '{角@MCQ}', '{角@BNC}', '{角@BMC}', '{角@DCM}', '{角@BNQ}', '{角@ABN}', '{角@MBN}']]}}, '855': {'condjson': {'等值集合': [['{角@BCM}', '{角@CBN}'], ['{角@BNM}', '{角@BNP}', '{角@CBN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BNM}', '{角@BCM}', '{角@BNP}', '{角@CBN}']]}}, '856': {'condjson': {'等值集合': [['{角@ABP}', '{角@ADP}'], ['{角@ABP}', '{角@MBP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ABP}', '{角@MBP}', '{角@ADP}']]}}, '857': {'condjson': {'等值集合': [['{角@MDN}', '{角@CDM}', '{角@BAN}', '{角@AMD}', '{角@MDQ}'], ['{角@BAN}', '{角@AND}', '{角@MAN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@MDN}', '{角@CDM}', '{角@MAN}', '{角@BAN}', '{角@AMD}', '{角@AND}', '{角@MDQ}']]}}, '858': {'condjson': {'等值集合': [['{角@CNM}', '{角@CBM}'], ['{角@ABC}', '{角@CBM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ABC}', '{角@CNM}', '{角@CBM}']]}}, '859': {'condjson': {'等值集合': [['{角@AMN}', '{角@ADN}'], ['{角@AMN}', '{角@AMP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@AMN}', '{角@AMP}', '{角@ADN}']]}}, '860': {'condjson': {'等值集合': [['{角@BMN}', '{角@CBM}'], ['{角@BMN}', '{角@BMP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BMP}', '{角@CBM}']]}}, '861': {'condjson': {'等值集合': [['{角@BCN}', '{角@CBM}'], ['{角@BCQ}', '{角@BCN}', '{角@BCD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BCN}', '{角@CBM}', '{角@BCQ}', '{角@BCD}']]}}, '862': {'condjson': {'等值集合': [['{角@ABC}', '{角@CNM}', '{角@CBM}'], ['{角@PNQ}', '{角@CNP}', '{角@CNM}', '{角@MNQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@PNQ}', '{角@MNQ}', '{角@CBM}', '{角@CNP}', '{角@ABC}', '{角@CNM}']]}}, '863': {'condjson': {'等值集合': [['{角@ADN}', '{角@DNM}'], ['{角@DNM}', '{角@DNP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADN}', '{角@DNM}', '{角@DNP}']]}}, '864': {'condjson': {'等值集合': [['{角@BCD}', '{角@ADC}'], ['{角@ADQ}', '{角@ADN}', '{角@ADC}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADN}', '{角@ADC}', '{角@ADQ}', '{角@BCD}']]}}, '865': {'condjson': {'等值集合': [['{角@BCN}', '{角@CBM}', '{角@BCQ}', '{角@BCD}'], ['{角@PNQ}', '{角@MNQ}', '{角@CBM}', '{角@CNP}', '{角@ABC}', '{角@CNM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BCN}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}']]}}, '866': {'condjson': {'等值集合': [['{角@BNM}', '{角@BCM}', '{角@BNP}', '{角@CBN}'], ['{角@CMP}', '{角@BCM}', '{角@CMN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BNM}', '{角@CMP}', '{角@BCM}', '{角@CMN}', '{角@BNP}', '{角@CBN}']]}}, '867': {'condjson': {'等值集合': [['{角@BMN}', '{角@BMP}', '{角@CBM}'], ['{角@BCN}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BCN}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@BMP}', '{角@MNQ}']]}}, '868': {'condjson': {'等值集合': [['{角@ADN}', '{角@ADC}', '{角@ADQ}', '{角@BCD}'], ['{角@BMN}', '{角@BCN}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@BMP}', '{角@MNQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@ADC}', '{角@CBM}', '{角@ADQ}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@BMP}', '{角@MNQ}']]}}, '869': {'condjson': {'等值集合': [['{角@BDC}', '{角@CAD}'], ['{角@DAP}', '{角@ADB}', '{角@CBD}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DAP}', '{角@ADB}', '{角@BDC}', '{角@CBD}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}']]}}, '870': {'condjson': {'等值集合': [['{角@DAM}', '{角@BCD}', '{角@BAD}'], ['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@ADC}', '{角@CBM}', '{角@ADQ}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@BMP}', '{角@MNQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}', '{角@BAD}']]}}, '871': {'condjson': {'等值集合': [['{角@ADN}', '{角@DNM}', '{角@DNP}'], ['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}', '{角@BAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@DNP}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@DNM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}', '{角@BAD}']]}}, '872': {'condjson': {'等值集合': [['{角@AMN}', '{角@AMP}', '{角@ADN}'], ['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@DNP}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@DNM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}', '{角@BAD}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@DNP}', '{角@ADQ}', '{角@AMN}', '{角@AMP}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@DNM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}', '{角@BAD}']]}}, '873': {'condjson': {'等值集合': [['{角@ABP}', '{角@NPQ}', '{角@MBP}'], ['{角@ABP}', '{角@MBP}', '{角@ADP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADP}', '{角@NPQ}', '{角@ABP}', '{角@MBP}']]}}, '874': {'condjson': {'等值集合': [['{角@CMD}', '{角@ANB}'], ['{线段@CM}', '{线段@BN}'], ['{线段@DM}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@CDM}', '{三角形@ABN}']]}}, '875': {'condjson': {'等值集合': [['{线段@CM}', '{线段@BN}'], ['{角@DCM}', '{角@ABN}'], ['{角@CMD}', '{角@ANB}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CDM}', '{三角形@ABN}']]}}, '876': {'condjson': {'等值集合': [['{线段@DM}', '{线段@AN}'], ['{角@CDM}', '{角@BAN}'], ['{角@CMD}', '{角@ANB}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@CDM}', '{三角形@ABN}']]}}, '877': {'condjson': {'等值集合': [['{角@BND}', '{角@AMC}'], ['{线段@BN}', '{线段@CM}'], ['{线段@DN}', '{线段@AM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BDN}', '{三角形@ACM}']]}}, '878': {'condjson': {'等值集合': [['{线段@BD}', '{线段@AC}'], ['{角@DBN}', '{角@ACM}'], ['{角@BDN}', '{角@CAM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BDN}', '{三角形@ACM}']]}}, '879': {'condjson': {'等值集合': [['{线段@BN}', '{线段@CM}'], ['{角@DBN}', '{角@ACM}'], ['{角@BND}', '{角@AMC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BDN}', '{三角形@ACM}']]}}, '880': {'condjson': {'等值集合': [['{线段@DN}', '{线段@AM}'], ['{角@BDN}', '{角@CAM}'], ['{角@BND}', '{角@AMC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BDN}', '{三角形@ACM}']]}}, '881': {'condjson': {'等值集合': [['{角@DBN}', '{角@ACM}'], ['{线段@BD}', '{线段@AC}'], ['{线段@BN}', '{线段@CM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BDN}', '{三角形@ACM}']]}}, '882': {'condjson': {'等值集合': [['{角@BMD}', '{角@ANC}'], ['{线段@BM}', '{线段@CN}'], ['{线段@DM}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BDM}', '{三角形@ACN}']]}}, '883': {'condjson': {'等值集合': [['{线段@BD}', '{线段@AC}'], ['{角@DBM}', '{角@ACN}'], ['{角@BDM}', '{角@CAN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BDM}', '{三角形@ACN}']]}}, '884': {'condjson': {'等值集合': [['{线段@BM}', '{线段@CN}'], ['{角@DBM}', '{角@ACN}'], ['{角@BMD}', '{角@ANC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BDM}', '{三角形@ACN}']]}}, '885': {'condjson': {'等值集合': [['{角@BDM}', '{角@CAN}'], ['{线段@BD}', '{线段@AC}'], ['{线段@DM}', '{线段@AN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BDM}', '{三角形@ACN}']]}}, '886': {'condjson': {'等值集合': [['{线段@DM}', '{线段@AN}'], ['{角@BDM}', '{角@CAN}'], ['{角@BMD}', '{角@ANC}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BDM}', '{三角形@ACN}']]}}, '887': {'condjson': {'等值集合': [['{线段@AD}', '{线段@AB}'], ['{角@DAP}', '{角@BAP}'], ['{角@ADP}', '{角@ABP}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADP}', '{三角形@ABP}']]}}, '888': {'condjson': {'等值集合': [['{角@APD}', '{角@APB}'], ['{线段@AP}', '{线段@AP}'], ['{线段@DP}', '{线段@BP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADP}', '{三角形@ABP}']]}}, '889': {'condjson': {'等值集合': [['{角@ADP}', '{角@ABP}'], ['{线段@AD}', '{线段@AB}'], ['{线段@DP}', '{线段@BP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADP}', '{三角形@ABP}']]}}, '890': {'condjson': {'等值集合': [['{线段@AP}', '{线段@AP}'], ['{角@DAP}', '{角@BAP}'], ['{角@APD}', '{角@APB}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADP}', '{三角形@ABP}']]}}, '891': {'condjson': {'等值集合': [['{线段@DP}', '{线段@BP}'], ['{角@ADP}', '{角@ABP}'], ['{角@APD}', '{角@APB}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@ADP}', '{三角形@ABP}']]}}, '892': {'condjson': {'等值集合': [['{线段@BC}', '{线段@CD}'], ['{角@CBP}', '{角@CDP}'], ['{角@BCP}', '{角@DCP}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCP}', '{三角形@CDP}']]}}, '893': {'condjson': {'等值集合': [['{角@BPC}', '{角@CPD}'], ['{线段@BP}', '{线段@DP}'], ['{线段@CP}', '{线段@CP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCP}', '{三角形@CDP}']]}}, '894': {'condjson': {'等值集合': [['{线段@BP}', '{线段@DP}'], ['{角@CBP}', '{角@CDP}'], ['{角@BPC}', '{角@CPD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCP}', '{三角形@CDP}']]}}, '895': {'condjson': {'等值集合': [['{角@CBP}', '{角@CDP}'], ['{线段@BC}', '{线段@CD}'], ['{线段@BP}', '{线段@DP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BCP}', '{三角形@CDP}']]}}, '896': {'condjson': {'等值集合': [['{线段@CP}', '{线段@CP}'], ['{角@BCP}', '{角@DCP}'], ['{角@BPC}', '{角@CPD}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BCP}', '{三角形@CDP}']]}}, '897': {'condjson': {'直线集合': [['{点@P}', '{点@B}']], '锐角集合': ['{角@BPM}', '{角@BPM}', '{角@CBP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@BPM}', '{角@BPM}', '{角@CBP}']]}}, '898': {'condjson': {'直线集合': [['{点@P}', '{点@B}']], '锐角集合': ['{角@CBP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@CBP}']]}}, '899': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '锐角集合': ['{角@DPN}', '{角@DPN}', '{角@ADP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@DPN}', '{角@DPN}', '{角@ADP}']]}}, '900': {'condjson': {'直线集合': [['{点@P}', '{点@D}']], '锐角集合': ['{角@ADP}']}, 'points': ['@@同位角对顶角内错角属性'], 'outjson': {'等值集合': [['{角@ADP}']]}}, '901': {'condjson': {'等值集合': [['{角@ABP}', '{角@ADP}'], ['{角@DPN}', '{角@ADP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DPN}', '{角@ABP}', '{角@ADP}']]}}, '902': {'condjson': {'等值集合': [['{角@DBM}', '{角@ACN}'], ['{角@DBM}', '{角@ABD}', '{角@BDC}', '{角@BDQ}', '{角@BDN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@ACN}', '{角@ABD}', '{角@BDN}', '{角@BDQ}', '{角@BDC}']]}}, '903': {'condjson': {'等值集合': [['{角@DPN}', '{角@ABP}', '{角@ADP}'], ['{角@ABP}', '{角@MBP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ADP}', '{角@DPN}', '{角@ABP}', '{角@MBP}']]}}, '904': {'condjson': {'等值集合': [['{角@NDP}', '{角@PDQ}', '{角@CBP}', '{角@CDP}'], ['{角@CBP}', '{角@BPM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@PDQ}', '{角@CBP}', '{角@CDP}', '{角@BPM}', '{角@NDP}']]}}, '905': {'condjson': {'等值集合': [['{角@DCP}', '{角@BCP}'], ['{角@ACN}', '{角@PCQ}', '{角@DCP}', '{角@BAC}', '{角@BAP}', '{角@ACQ}', '{角@ACD}', '{角@CAM}', '{角@NCP}', '{角@MAP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ACN}', '{角@PCQ}', '{角@DCP}', '{角@BAC}', '{角@BAP}', '{角@ACQ}', '{角@ACD}', '{角@CAM}', '{角@NCP}', '{角@BCP}', '{角@MAP}']]}}, '906': {'condjson': {'等值集合': [['{角@ACN}', '{角@PCQ}', '{角@DCP}', '{角@BAC}', '{角@BAP}', '{角@ACQ}', '{角@ACD}', '{角@CAM}', '{角@NCP}', '{角@BCP}', '{角@MAP}'], ['{角@DAP}', '{角@CAD}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ACN}', '{角@DAP}', '{角@PCQ}', '{角@DCP}', '{角@BAC}', '{角@BAP}', '{角@ACQ}', '{角@ACD}', '{角@CAD}', '{角@CAM}', '{角@NCP}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@MAP}', '{角@ACB}']]}}, '907': {'condjson': {'等值集合': [['{角@BPM}', '{角@DQP}', '{角@NQP}'], ['{角@NDP}', '{角@PDQ}', '{角@CBP}', '{角@BPM}', '{角@CDP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@PDQ}', '{角@DQP}', '{角@NDP}', '{角@CBP}', '{角@CDP}', '{角@BPM}', '{角@NQP}']]}}, '908': {'condjson': {'等值集合': [['{角@DBM}', '{角@ACN}', '{角@ABD}', '{角@BDN}', '{角@BDQ}', '{角@BDC}'], ['{角@ACN}', '{角@DAP}', '{角@PCQ}', '{角@DCP}', '{角@BAC}', '{角@BAP}', '{角@ACQ}', '{角@ACD}', '{角@CAD}', '{角@CAM}', '{角@NCP}', '{角@CPN}', '{角@APM}', '{角@BCP}', '{角@MAP}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@CAD}', '{角@BDN}', '{角@APM}', '{角@NCP}', '{角@BCP}', '{角@MAP}', '{角@ACN}', '{角@DAP}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}', '{角@CPN}', '{角@ACB}']]}}, '909': {'condjson': {'等值集合': [['{角@ADB}', '{角@CBD}', '{角@CAD}'], ['{角@DBM}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@CAD}', '{角@BDN}', '{角@APM}', '{角@NCP}', '{角@BCP}', '{角@MAP}', '{角@ACN}', '{角@DAP}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}', '{角@CPN}', '{角@ACB}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DBM}', '{角@ADB}', '{角@BDC}', '{角@DCP}', '{角@BAC}', '{角@ACQ}', '{角@CBD}', '{角@CAD}', '{角@BDN}', '{角@APM}', '{角@NCP}', '{角@BCP}', '{角@MAP}', '{角@ACN}', '{角@DAP}', '{角@PCQ}', '{角@ABD}', '{角@BAP}', '{角@ACD}', '{角@BDQ}', '{角@CAM}', '{角@CPN}', '{角@ACB}']]}}, '910': {'condjson': {'等值集合': [['{角@NPQ}', '{角@ABP}', '{角@MBP}', '{角@ADP}'], ['{角@ADP}', '{角@DPN}', '{角@ABP}', '{角@MBP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@ABP}', '{角@MBP}', '{角@ADP}', '{角@NPQ}', '{角@DPN}']]}}, '911': {'condjson': {'等值集合': [['{角@CBP}', '{角@BPM}'], ['{角@BPM}', '{角@DQP}', '{角@NQP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@DQP}', '{角@CBP}', '{角@BPM}', '{角@NQP}']]}}, '912': {'condjson': {'等值集合': [['{角@DPN}', '{角@ABP}', '{角@ADP}'], ['{角@ABP}', '{角@NPQ}', '{角@MBP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@NPQ}', '{角@ADP}', '{角@DPN}', '{角@ABP}', '{角@MBP}']]}}, '913': {'condjson': {'等值集合': [['{角@NDP}', '{角@PDQ}', '{角@CBP}', '{角@CDP}'], ['{角@DQP}', '{角@CBP}', '{角@BPM}', '{角@NQP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@PDQ}', '{角@DQP}', '{角@NDP}', '{角@CBP}', '{角@BPM}', '{角@CDP}', '{角@NQP}']]}}, '914': {'condjson': {'等值集合': [['{线段@NP}', '{线段@NP}'], ['{角@PNQ}', '{角@DNP}'], ['{角@NPQ}', '{角@DPN}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@DNP}']]}}, '915': {'condjson': {'等值集合': [['{线段@BP}', '{线段@DP}'], ['{角@MBP}', '{角@DPN}'], ['{角@BPM}', '{角@NDP}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BMP}', '{三角形@DNP}']]}}, '916': {'condjson': {'全等三角形集合': [['{三角形@DNP}', '{三角形@NPQ}'], ['{三角形@BMP}', '{三角形@DNP}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@BMP}', '{三角形@DNP}', '{三角形@NPQ}']]}}, '917': {'condjson': {'全等三角形集合': [['{三角形@BMP}', '{三角形@DNP}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@BMP}', '{角@DNP}'], ['{线段@DP}', '{线段@BP}'], ['{线段@DN}', '{线段@MP}'], ['{角@DPN}', '{角@MBP}'], ['{线段@NP}', '{线段@BM}'], ['{角@NDP}', '{角@BPM}']]}}, '918': {'condjson': {'全等三角形集合': [['{三角形@DNP}', '{三角形@NPQ}']]}, 'points': ['@@全等三角形必要条件'], 'outjson': {'等值集合': [['{角@NDP}', '{角@NQP}'], ['{线段@NP}'], ['{角@PNQ}', '{角@DNP}'], ['{线段@DP}', '{线段@PQ}'], ['{线段@NQ}', '{线段@DN}'], ['{角@DPN}', '{角@NPQ}']]}}, '919': {'condjson': {'等值集合': [['{线段@DN}', '{线段@MP}'], ['{线段@NQ}', '{线段@DN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@DN}', '{线段@NQ}', '{线段@MP}']]}}, '920': {'condjson': {'等值集合': [['{线段@DP}', '{线段@BP}'], ['{线段@DP}', '{线段@PQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@DP}', '{线段@BP}', '{线段@PQ}']]}}, '921': {'condjson': {'等值集合': [['{线段@CN}', '{线段@BM}'], ['{线段@NP}', '{线段@BM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@CN}', '{线段@NP}', '{线段@BM}']]}}, '922': {'condjson': {'等值集合': [['{线段@AM}', '{线段@DN}'], ['{线段@DN}', '{线段@NQ}', '{线段@MP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@NQ}', '{线段@MP}', '{线段@AM}', '{线段@DN}']]}}, '923': {'condjson': {'等值集合': [['{角@NDP}', '{角@NQP}'], ['{角@NDP}', '{角@PDQ}', '{角@CDP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@PDQ}', '{角@NDP}', '{角@CDP}', '{角@NQP}']]}}, '924': {'condjson': {'等值集合': [['{角@PDQ}', '{角@NDP}', '{角@CDP}', '{角@NQP}'], ['{角@DQP}', '{角@NQP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@PDQ}', '{角@DQP}', '{角@NDP}', '{角@CDP}', '{角@NQP}']]}}, '925': {'condjson': {'等值集合': [['{角@CBP}', '{角@CDP}'], ['{角@CBP}', '{角@BPM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@CBP}', '{角@BPM}', '{角@CDP}']]}}, '926': {'condjson': {'等值集合': [['{角@BMP}', '{角@DNP}'], ['{角@BMN}', '{角@BMP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BMP}', '{角@DNP}']]}}, '927': {'condjson': {'等值集合': [['{角@BMN}', '{角@BMP}', '{角@DNP}'], ['{角@DNM}', '{角@DNP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BMP}', '{角@DNP}', '{角@DNM}']]}}, '928': {'condjson': {'等值集合': [['{线段@NQ}', '{线段@DN}'], ['{线段@AM}', '{线段@DN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@NQ}', '{线段@AM}', '{线段@DN}']]}}, '929': {'condjson': {'等值集合': [['{线段@DN}', '{线段@MP}'], ['{线段@NQ}', '{线段@AM}', '{线段@DN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@NQ}', '{线段@MP}', '{线段@AM}', '{线段@DN}']]}}, '930': {'condjson': {'等值集合': [['{角@CBP}', '{角@BPM}', '{角@CDP}'], ['{角@PDQ}', '{角@DQP}', '{角@NDP}', '{角@CDP}', '{角@NQP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@PDQ}', '{角@DQP}', '{角@NDP}', '{角@CBP}', '{角@CDP}', '{角@BPM}', '{角@NQP}']]}}, '931': {'condjson': {'等值集合': [['{角@BMN}', '{角@CBM}'], ['{角@BMN}', '{角@BMP}', '{角@DNP}', '{角@DNM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BMP}', '{角@DNP}', '{角@CBM}', '{角@DNM}']]}}, '932': {'condjson': {'等值集合': [['{角@BMN}', '{角@BMP}', '{角@DNP}', '{角@CBM}', '{角@DNM}'], ['{角@BCN}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@BCN}', '{角@DNP}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@BMP}', '{角@MNQ}', '{角@DNM}']]}}, '933': {'condjson': {'等值集合': [['{角@ADN}', '{角@ADC}', '{角@ADQ}', '{角@BCD}'], ['{角@BMN}', '{角@BCN}', '{角@DNP}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@BCD}', '{角@ABC}', '{角@CNM}', '{角@PNQ}', '{角@BMP}', '{角@MNQ}', '{角@DNM}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@DNP}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DNM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}']]}}, '934': {'condjson': {'等值集合': [['{角@DAM}', '{角@BCD}', '{角@BAD}'], ['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@DNP}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DNM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@BMN}', '{角@ADN}', '{角@BCN}', '{角@DNP}', '{角@ADQ}', '{角@BCD}', '{角@ABC}', '{角@BMP}', '{角@DAM}', '{角@DNM}', '{角@ADC}', '{角@CBM}', '{角@BCQ}', '{角@CNP}', '{角@CNM}', '{角@PNQ}', '{角@MNQ}', '{角@BAD}']]}}, '935': {'condjson': {'等值集合': [['{线段@AB}', '{线段@MN}'], ['{线段@NP}', '{线段@BM}']]}, 'points': ['@@表达式传递'], 'outjson': {'等值集合': [['{线段@AM}', '{线段@MP}']]}}, '936': {'condjson': {'等值集合': [['{线段@CD}', '{线段@MN}'], ['{线段@DN}', '{线段@MP}']]}, 'points': ['@@表达式传递'], 'outjson': {'等值集合': [['{线段@CN}', '{线段@NP}']]}}, '937': {'condjson': {'等值集合': [['{线段@CN}', '{线段@BM}'], ['{线段@CN}', '{线段@NP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@CN}', '{线段@NP}', '{线段@BM}']]}}, '938': {'condjson': {'等值集合': [['{线段@AM}', '{线段@MP}'], ['{线段@AM}', '{线段@DN}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@DN}', '{线段@AM}', '{线段@MP}']]}}, '939': {'condjson': {'等值集合': [['{角@PDQ}', '{角@NDP}', '{角@CDP}', '{角@NQP}'], ['{角@DQP}', '{角@CBP}', '{角@BPM}', '{角@NQP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{角@PDQ}', '{角@DQP}', '{角@NDP}', '{角@CBP}', '{角@BPM}', '{角@CDP}', '{角@NQP}']]}}, '940': {'condjson': {'等值集合': [['{线段@NQ}', '{线段@DN}'], ['{线段@DN}', '{线段@AM}', '{线段@MP}']]}, 'points': ['@@等值间传递'], 'outjson': {'等值集合': [['{线段@NQ}', '{线段@MP}', '{线段@AM}', '{线段@DN}']]}}, '941': {'condjson': {'等值集合': [['{线段@NP}', '{线段@BM}'], ['{角@PNQ}', '{角@BMP}'], ['{角@NPQ}', '{角@MBP}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@BMP}']]}}, '942': {'condjson': {'等值集合': [['{角@NQP}', '{角@BPM}'], ['{线段@NQ}', '{线段@MP}'], ['{线段@PQ}', '{线段@BP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@BMP}']]}}, '943': {'condjson': {'等值集合': [['{角@NQP}', '{角@NDP}'], ['{线段@NQ}', '{线段@DN}'], ['{线段@PQ}', '{线段@DP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@DNP}']]}}, '944': {'condjson': {'等值集合': [['{线段@NQ}', '{线段@MP}'], ['{角@PNQ}', '{角@BMP}'], ['{角@NQP}', '{角@BPM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@BMP}']]}}, '945': {'condjson': {'等值集合': [['{角@NPQ}', '{角@MBP}'], ['{线段@NP}', '{线段@BM}'], ['{线段@PQ}', '{线段@BP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@BMP}']]}}, '946': {'condjson': {'等值集合': [['{线段@NQ}', '{线段@DN}'], ['{角@PNQ}', '{角@DNP}'], ['{角@NQP}', '{角@NDP}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@DNP}']]}}, '947': {'condjson': {'等值集合': [['{角@NPQ}', '{角@DPN}'], ['{线段@NP}', '{线段@NP}'], ['{线段@PQ}', '{线段@DP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@DNP}']]}}, '948': {'condjson': {'等值集合': [['{角@PNQ}', '{角@BMP}'], ['{线段@NP}', '{线段@BM}'], ['{线段@NQ}', '{线段@MP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@BMP}']]}}, '949': {'condjson': {'等值集合': [['{线段@PQ}', '{线段@BP}'], ['{角@NPQ}', '{角@MBP}'], ['{角@NQP}', '{角@BPM}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@BMP}']]}}, '950': {'condjson': {'等值集合': [['{角@PNQ}', '{角@DNP}'], ['{线段@NP}', '{线段@NP}'], ['{线段@NQ}', '{线段@DN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@DNP}']]}}, '951': {'condjson': {'等值集合': [['{线段@PQ}', '{线段@DP}'], ['{角@NPQ}', '{角@DPN}'], ['{角@NQP}', '{角@NDP}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@NPQ}', '{三角形@DNP}']]}}, '952': {'condjson': {'等值集合': [['{角@DAM}', '{角@MNQ}'], ['{线段@AD}', '{线段@MN}'], ['{线段@AM}', '{线段@NQ}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADM}', '{三角形@MNQ}']]}}, '953': {'condjson': {'等值集合': [['{线段@BM}', '{线段@NP}'], ['{角@MBP}', '{角@DPN}'], ['{角@BMP}', '{角@DNP}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BMP}', '{三角形@DNP}']]}}, '954': {'condjson': {'等值集合': [['{角@BPM}', '{角@NDP}'], ['{线段@BP}', '{线段@DP}'], ['{线段@MP}', '{线段@DN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BMP}', '{三角形@DNP}']]}}, '955': {'condjson': {'等值集合': [['{角@BMP}', '{角@DNP}'], ['{线段@BM}', '{线段@NP}'], ['{线段@MP}', '{线段@DN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BMP}', '{三角形@DNP}']]}}, '956': {'condjson': {'等值集合': [['{线段@MP}', '{线段@DN}'], ['{角@BMP}', '{角@DNP}'], ['{角@BPM}', '{角@NDP}']]}, 'points': ['@@全等三角形充分条件角边角'], 'outjson': {'全等三角形集合': [['{三角形@BMP}', '{三角形@DNP}']]}}, '957': {'condjson': {'等值集合': [['{角@MBP}', '{角@DPN}'], ['{线段@BM}', '{线段@NP}'], ['{线段@BP}', '{线段@DP}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@BMP}', '{三角形@DNP}']]}}, '958': {'condjson': {'等值集合': [['{角@DNM}', '{角@MNQ}'], ['{线段@DN}', '{线段@NQ}'], ['{线段@MN}', '{线段@MN}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@MNQ}']]}}, '959': {'condjson': {'等值集合': [['{角@ADN}', '{角@MNQ}'], ['{线段@AD}', '{线段@MN}'], ['{线段@DN}', '{线段@NQ}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@MNQ}']]}}, '960': {'condjson': {'等值集合': [['{角@MNQ}', '{角@AMN}'], ['{线段@MN}', '{线段@MN}'], ['{线段@NQ}', '{线段@AM}']]}, 'points': ['@@全等三角形充分条件边角边'], 'outjson': {'全等三角形集合': [['{三角形@MNQ}', '{三角形@AMN}']]}}, '961': {'condjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@AMN}'], ['{三角形@AMN}', '{三角形@MNQ}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@ADN}', '{三角形@AMN}', '{三角形@MNQ}']]}}, '962': {'condjson': {'全等三角形集合': [['{三角形@DMN}', '{三角形@AMN}'], ['{三角形@ADN}', '{三角形@AMN}', '{三角形@MNQ}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@AMN}', '{三角形@DMN}', '{三角形@MNQ}', '{三角形@ADN}']]}}, '963': {'condjson': {'全等三角形集合': [['{三角形@AMN}', '{三角形@ADM}'], ['{三角形@AMN}', '{三角形@DMN}', '{三角形@MNQ}', '{三角形@ADN}']]}, 'points': ['@@全等三角形间传递'], 'outjson': {'全等三角形集合': [['{三角形@MNQ}', '{三角形@ADM}', '{三角形@AMN}', '{三角形@DMN}', '{三角形@ADN}']]}}, '已知': {'condjson': {}, 'points': ['@@已知'], 'outjson': {'锐角集合': ['{角@APM}', '{角@ACB}'], '钝角集合': [], '等价集合': [], '全等集合': [], '全等三角形集合': [], '垂直集合': [], '平行集合': [['{线段@BC}', '{线段@MN}']], '直角集合': [], '平角集合': [], '直角三角形集合': [], '余角集合': [], '补角集合': [], '表达式集合': ['1 8 0 ^ { \\circ } = {角@NPQ} + {角@BPQ} + {角@BPM}'], '点集合': ['{点@A}', '{点@B}', '{点@P}', '{点@Q}', '{点@M}', '{点@N}', '{点@C}', '{点@D}'], '直线集合': [['{点@C}', '{点@Q}', '{点@N}', '{点@D}'], ['{点@M}', '{点@P}', '{点@N}'], ['{点@A}', '{点@M}', '{点@B}'], ['{点@A}', '{点@P}', '{点@C}']], '正方形集合': ['{正方形@ABCD}'], '角集合': ['{角@BPQ}', '{角@NPQ}', '{角@APM}', '{角@ACB}', '{角@BPM}'], '线段集合': ['{线段@BC}', '{线段@MN}', '{线段@BP}', '{线段@PQ}'], '等值集合': [['{角@BPQ}', '9 0 ^ { \\circ }']], '默认集合': [0]}}, '求证': {'condjson': {'等值集合': [['{线段@BP}', '{线段@PQ}']]}, 'points': ['@@求证'], 'outjson': {}}}"""
        instra = instra.replace("'", '"').replace("\\", "\\\\")
        print(instra)
        space_ins._step_node = json.loads(instra, encoding="utf-8")
        nodejson = space_ins._step_node
        logger1.info("原始节点数:{}, {}".format(len(nodejson), nodejson))
        # 4. 根据已知节点向下生成树
        nodename = "已知"
        G1 = self.gene_downtree_from(nodejson, nodename)
        solvelast = [1 for i1 in G1.edges() if i1[1] == "求证"]
        if sum(solvelast) > 0:
            logger1.info("思维树成功生成！")
            logger1.info("精简树...")
            delcounter = 0
            stopcounter = 9999
            logger1.info("ori_node num: {}. ori_edge num: {}.".format(len(G1.nodes()), len(G1.edges())))
            G1 = self.deltree_layer(G1, delcounter, stopcounter)
            logger1.info("final_node num: {}. final_edge num: {}.".format(len(G1.nodes()), len(G1.edges())))
            pos = nx.kamada_kawai_layout(G1)
            nx.draw(G1, pos, font_size=10, with_labels=True)
            plt.axis('on')
            plt.xticks([])
            plt.yticks([])
            plt.show()
            edglist = [[*edg] for edg in G1.edges()]
            return json.dumps(nodejson, ensure_ascii=False), json.dumps(edglist, ensure_ascii=False)
        logger1.info("没有发现解答路径！")
        # 查找下行G1末 输出等值节点. 查找上行G2末 输入等值节点 {线段@AD}', '{线段@MN}
        nodename = "807"
        # G2 = self.gene_uptree_from(nodejson, nodename)
        G3 = self.gene_fulltree_from(nodejson)
        # mislist1 = [node for node in G2.nodes() if node not in G1.nodes()]
        # mislist2 = [node for node in G2.nodes() if node not in G3.nodes()]
        # commlist = [node for node in G2.nodes() if node in G1.nodes()]
        # print(mislist1)
        # print(mislist2)
        # print(commlist)
        print("下行树节点{}，边{}".format(len(G1.nodes()), len(G1.edges())))
        # print("上行树节点{}，边{}".format(len(G2.nodes()), len(G2.edges())))
        print("全量树节点{}，边{}".format(len(G3.nodes()), len(G3.edges())))
        # print(G2.nodes())
        # print(G2.edges())
        print(555)
        for node in G1.nodes():
            if "等值集合" in nodejson[node]["outjson"]:
                # print(node, nodejson[node]["outjson"]["等值集合"])
                for tlist in nodejson[node]["outjson"]["等值集合"]:
                    # sigt1 = 1 if len(set(["{角@BPM}", "{角@NQP}"]).intersection(set(tlist))) == 2 else 0
                    # sigt2 = 1 if len(set(["{角@MBP}", "{角@NPQ}"]).intersection(set(tlist))) == 2 else 0
                    # sigt3 = 1 if len(set(["{角@BMP}", "{角@PNQ}"]).intersection(set(tlist))) == 2 else 0
                    sigt4 = 1 if len(set(["{线段@MP}", "{线段@NQ}"]).intersection(set(tlist))) == 2 else 0
                    sigt5 = 1 if len(set(["{线段@BP}", "{线段@PQ}"]).intersection(set(tlist))) == 2 else 0
                    sigt6 = 1 if len(set(["{线段@BM}", "{线段@NP}"]).intersection(set(tlist))) == 2 else 0
                    # sigt7 = 1 if len(set(["{线段@AD}", "{线段@MN}"]).intersection(set(tlist))) == 2 else 0
                    # sigta = sum([sigt7])
                    sigta = sum([sigt4, sigt5, sigt6])
                    # sigta = sum([sigt1, sigt2, sigt3, sigt4, sigt5, sigt6])
                    if sigta > 0:
                        print(node, nodejson[node])
                        break
        print(556)
        for node in G1.nodes():
            if nodejson[node]["points"][0].startswith("@@全等三角形充分"):
                for tlist in nodejson[node]["outjson"]["全等三角形集合"]:
                    if "{三角形@ADM}" in tlist and "{三角形@DMN}" in tlist:
                        # if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
                        print(node, nodejson[node])
        print(557)
        for node in G1.nodes():
            if "@@全等三角形必要条件" in nodejson[node]["points"]:
                for tlist in nodejson[node]["condjson"]["全等三角形集合"]:
                    if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
                        print(node, nodejson[node])
        print(558)
        targst = set(['{角@BMP}', '{角@PNQ}', '{角@BPM}', '{角@NPQ}', '{角@NQP}', '{角@MBP}', '{线段@BM}', '{线段@MP}', '{线段@NP}', '{线段@NQ}'])
        for node in G1.nodes():
            if "等值集合" in nodejson[node]["outjson"]:
                for onsett in nodejson[node]["outjson"]["等值集合"]:
                    if len(targst.intersection(set(onsett))) > 0:
                        print(node, onsett)
        # print(558)
        # for node in G2.nodes():
        #     if "等值集合" in nodejson[node]["condjson"]:
        #         # print(node, nodejson[node]["outjson"]["等值集合"])
        #         for tlist in nodejson[node]["condjson"]["等值集合"]:
        #             # sigt1 = 1 if len(set(["{角@BPM}", "{角@NQP}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt2 = 1 if len(set(["{角@MBP}", "{角@NPQ}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt3 = 1 if len(set(["{角@BMP}", "{角@PNQ}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt4 = 1 if len(set(["{线段@AB}", "{线段@MN}"]).intersection(set(tlist))) == 2 else 0
        #             # sigt5 = 1 if len(set(["{线段@AM}", "{线段@MP}"]).intersection(set(tlist))) == 2 else 0
        #             sigt7 = 1 if len(set(["{线段@AD}", "{线段@MN}"]).intersection(set(tlist))) == 2 else 0
        #             sigta = sum([sigt7])
        #             # sigta = sum([sigt1, sigt2, sigt3, sigt4, sigt5, sigt6])
        #             if sigta > 0:
        #                 print(node, nodejson[node])
        #                 break
        # print(559)
        # for node in G2.nodes():
        #     if nodejson[node]["points"][0].startswith("@@全等三角形充分"):
        #         for tlist in nodejson[node]["outjson"]["全等三角形集合"]:
        #             if "{三角形@ADM}" in tlist and "{三角形@DMN}" in tlist:
        #                 # if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
        #                 print(node, nodejson[node])
        # print(560)
        # for node in G2.nodes():
        #     if "@@全等三角形必要条件" in nodejson[node]["points"]:
        #         for tlist in nodejson[node]["condjson"]["全等三角形集合"]:
        #             if "{三角形@ADM}" in tlist and "{三角形@AMN}" in tlist:
        #                 print(node, nodejson[node])
        print(561)
        targst = set(['{角@BMP}', '{角@PNQ}', '{角@BPM}', '{角@NPQ}', '{角@NQP}', '{角@MBP}', '{线段@BM}', '{线段@MP}', '{线段@NP}', '{线段@NQ}'])
        for node in G3.nodes():
            if "等值集合" in nodejson[node]["outjson"]:
                # print(node, nodejson[node]["outjson"]["等值集合"])
                for tlist in nodejson[node]["outjson"]["等值集合"]:
                    # sigt1 = 1 if len(set(["{角@BPM}", "{角@NQP}"]).intersection(set(tlist))) == 2 else 0
                    # sigt2 = 1 if len(set(["{角@MBP}", "{角@NPQ}"]).intersection(set(tlist))) == 2 else 0
                    # sigt3 = 1 if len(set(["{角@BMP}", "{角@PNQ}"]).intersection(set(tlist))) == 2 else 0
                    # sigt4 = 1 if len(set(["{线段@MP}", "{线段@NQ}"]).intersection(set(tlist))) == 2 else 0
                    # sigt5 = 1 if len(set(["{线段@BP}", "{线段@PQ}"]).intersection(set(tlist))) == 2 else 0
                    # sigt6 = 1 if len(set(["{线段@BM}", "{线段@NP}"]).intersection(set(tlist))) == 2 else 0
                    sigt7 = 1 if len(targst.intersection(set(tlist))) > 0 else 0
                    sigta = sum([sigt7])
                    # sigta = sum([sigt1, sigt2, sigt3, sigt4, sigt5, sigt6])
                    if sigta > 0:
                        print(node, nodejson[node])
                        break
        print(571)
        for node in G3.nodes():
            if "等值集合" in nodejson[node]["condjson"]:
                for tlist in nodejson[node]["condjson"]["等值集合"]:
                    sigt7 = 1 if len(set(["{线段@AD}", "{线段@MN}"]).intersection(set(tlist))) == 2 else 0
                    sigta = sum([sigt7])
                    # sigta = sum([sigt1, sigt2, sigt3, sigt4, sigt5, sigt6])
                    if sigta > 0:
                        print(node, nodejson[node])
                        break
        print(562)
        for node in G3.nodes():
            if nodejson[node]["points"][0].startswith("@@全等三角形充分"):
                for tlist in nodejson[node]["outjson"]["全等三角形集合"]:
                    if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
                        print(node, nodejson[node])
        print(563)
        for node in G3.nodes():
            if "@@全等三角形必要条件" in nodejson[node]["points"]:
                for tlist in nodejson[node]["condjson"]["全等三角形集合"]:
                    if "{三角形@ADM}" in tlist and "{三角形@AMN}" in tlist:
                        # if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
                        print(node, nodejson[node])
        print(564)
        for node in G3.nodes():
            if "@@全等三角形充分" in nodejson[node]["points"][0]:
                for tlist in nodejson[node]["outjson"]["全等三角形集合"]:
                    if "{三角形@ADM}" in tlist and "{三角形@AMN}" in tlist:
                        # if "{三角形@BMP}" in tlist and "{三角形@NPQ}" in tlist:
                        print(node, nodejson[node])
        print(565)
        for node in G3.nodes():
            if "等值集合" in nodejson[node]["outjson"]:
                for onsett in nodejson[node]["outjson"]["等值集合"]:
                    if len(targst.intersection(set(onsett))) > 0:
                        print(node, onsett)
        raise 456
        # 删除非答案的末节点
        nodename = "求证"
        parentnodes = self.checknode_parent(G1, nodename, steps=2)
        print("{} 的父节点有 {} {}".format(nodename, len(parentnodes), parentnodes))
        nodename = "已知"
        childnodes = self.checknode_child(G1, nodename, steps=1)
        print("{} 的子节点有 {} {}".format(nodename, len(childnodes), childnodes))
        return None, None

    def gene_uptree_from(self, nodejson, nodename):
        # 3. 函数定义： 子集合并判断充分条件的超集，子集判断共性，是否可连接
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        listset_obj = [objset for objset in basic_space_ins._setobj if
                       basic_space_ins._setobj[objset]["结构形式"].startswith("一级列表二级")]
        set_obj = [objset for objset in basic_space_ins._setobj if
                   basic_space_ins._setobj[objset]["结构形式"].startswith("一级集合")]

        def a_commonset_b(a, b):
            commenset = {}
            for obja in a:
                for objb in b:
                    if objb == obja:
                        if obja in listset_obj:
                            commenset[obja] = []
                            for onea in a[obja]:
                                for oneb in b[objb]:
                                    comelem = set(onea).intersection(set(oneb))
                                    if len(comelem) > 0:
                                        commenset[obja].append(list(comelem))
                            if len(commenset[obja]) == 0:
                                del commenset[obja]
                        elif obja in set_obj:
                            comelem = set(a[obja]).intersection(set(b[objb]))
                            if len(comelem) > 0:
                                commenset[obja] = list(comelem)
            if len(commenset) == 0:
                return False
            else:
                return commenset

        def genesuperset(listsobj):
            superset = {}
            for nodejso in listsobj:
                for obj in nodejso:
                    if obj not in superset:
                        superset[obj] = []
                    if obj in listset_obj:
                        for onelist in nodejso[obj]:
                            superset[obj].append(onelist)
                    elif obj in set_obj:
                        superset[obj] += nodejso[obj]
                        superset[obj] = list(set(superset[obj]))
            superset = self.listlist_deliverall(superset)
            return superset

        def a_supset_b(a, b):
            "判断a是b的超集"
            for objb in b:
                if objb not in a:
                    return False
                if objb in listset_obj:
                    for oneb in b[objb]:
                        setsig = 0
                        for onea in a[objb]:
                            if set(onea).issuperset(set(oneb)):
                                setsig = 1
                                break
                        if setsig == 0:
                            return False
                elif objb in set_obj:
                    if not set(a[objb]).issuperset(set(b[objb])):
                        return False
                else:
                    raise Exception("unknow error")
            # 没有发现异常，最后输出相同
            return True

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
                        common_elem = a_commonset_b(nodejson[waitenode]["outjson"], nodejson[knownode]["condjson"])
                        if common_elem:
                            tcondilist[waitenode] = common_elem
                # 生成超集
                supersets = genesuperset(tcondilist.values())
                if a_supset_b(supersets, nodejson[knownode]["condjson"]):
                    knownodes |= set(tcondilist.keys())
                    for condnode in tcondilist:
                        if "_".join([condnode, knownode]) not in edgepairs:
                            G.add_edge(condnode, knownode, weight=1)
                            edgepairs.append("_".join([condnode, knownode]))
        return G

    def gene_downtree_from(self, nodejson, nodename):
        # 3. 函数定义： 子集合并判断充分条件的超集，子集判断共性，是否可连接
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        listset_obj = [objset for objset in basic_space_ins._setobj if
                       basic_space_ins._setobj[objset]["结构形式"].startswith("一级列表二级")]
        set_obj = [objset for objset in basic_space_ins._setobj if
                   basic_space_ins._setobj[objset]["结构形式"].startswith("一级集合")]

        def a_commonset_b(a, b):
            commenset = {}
            for obja in a:
                for objb in b:
                    if objb == obja:
                        if obja in listset_obj:
                            commenset[obja] = []
                            for onea in a[obja]:
                                for oneb in b[objb]:
                                    comelem = set(onea).intersection(set(oneb))
                                    if len(comelem) > 0:
                                        commenset[obja].append(list(comelem))
                            if len(commenset[obja]) == 0:
                                del commenset[obja]
                        elif obja in set_obj:
                            comelem = set(a[obja]).intersection(set(b[objb]))
                            if len(comelem) > 0:
                                commenset[obja] = list(comelem)
            if len(commenset) == 0:
                return False
            else:
                return commenset

        def genesuperset(listsobj):
            superset = {}
            for nodejso in listsobj:
                for obj in nodejso:
                    if obj not in superset:
                        superset[obj] = []
                    if obj in listset_obj:
                        for onelist in nodejso[obj]:
                            superset[obj].append(onelist)
                    elif obj in set_obj:
                        superset[obj] += nodejso[obj]
                        superset[obj] = list(set(superset[obj]))
            superset = self.listlist_deliverall(superset)
            return superset

        def a_supset_b(a, b):
            "判断a是b的超集"
            for objb in b:
                if objb not in a:
                    return False
                if objb in listset_obj:
                    for oneb in b[objb]:
                        setsig = 0
                        for onea in a[objb]:
                            if set(onea).issuperset(set(oneb)):
                                setsig = 1
                                break
                        if setsig == 0:
                            return False
                elif objb in set_obj:
                    if not set(a[objb]).issuperset(set(b[objb])):
                        return False
                else:
                    raise Exception("unknow error")
            # 没有发现异常，最后输出相同
            return True

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
                        common_elem = a_commonset_b(nodejson[waitenode]["condjson"], nodejson[knownode]["outjson"])
                        if common_elem:
                            tcondilist[knownode] = common_elem
                # 生成超集
                supersets = genesuperset(tcondilist.values())
                if a_supset_b(supersets, nodejson[waitenode]["condjson"]):
                    knownodes.add(waitenode)
                    for condnode in tcondilist:
                        if "_".join([condnode, waitenode]) not in edgepairs:
                            G.add_edge(condnode, waitenode, weight=1)
                            edgepairs.append("_".join([condnode, waitenode]))
        return G

    def gene_fulltree_from(self, nodejson):
        # 3. 函数定义： 子集合并判断充分条件的超集，子集判断共性，是否可连接
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        listset_obj = [objset for objset in basic_space_ins._setobj if
                       basic_space_ins._setobj[objset]["结构形式"].startswith("一级列表二级")]
        set_obj = [objset for objset in basic_space_ins._setobj if
                   basic_space_ins._setobj[objset]["结构形式"].startswith("一级集合")]

        def a_commonset_b(a, b):
            commenset = {}
            for obja in a:
                for objb in b:
                    if objb == obja:
                        if obja in listset_obj:
                            commenset[obja] = []
                            for onea in a[obja]:
                                for oneb in b[objb]:
                                    comelem = set(onea).intersection(set(oneb))
                                    if len(comelem) > 0:
                                        commenset[obja].append(list(comelem))
                            if len(commenset[obja]) == 0:
                                del commenset[obja]
                        elif obja in set_obj:
                            comelem = set(a[obja]).intersection(set(b[objb]))
                            if len(comelem) > 0:
                                commenset[obja] = list(comelem)
            if len(commenset) == 0:
                return False
            else:
                return commenset

        def genesuperset(listsobj):
            superset = {}
            for nodejso in listsobj:
                for obj in nodejso:
                    if obj not in superset:
                        superset[obj] = []
                    if obj in listset_obj:
                        for onelist in nodejso[obj]:
                            superset[obj].append(onelist)
                    elif obj in set_obj:
                        superset[obj] += nodejso[obj]
                        superset[obj] = list(set(superset[obj]))
            superset = self.listlist_deliverall(superset)
            return superset

        def a_supset_b(a, b):
            "判断a是b的超集"
            for objb in b:
                if objb not in a:
                    return False
                if objb in listset_obj:
                    for oneb in b[objb]:
                        setsig = 0
                        for onea in a[objb]:
                            if set(onea).issuperset(set(oneb)):
                                setsig = 1
                                break
                        if setsig == 0:
                            return False
                elif objb in set_obj:
                    if not set(a[objb]).issuperset(set(b[objb])):
                        return False
                else:
                    raise Exception("unknow error")
            # 没有发现异常，最后输出相同
            return True

        G = nx.DiGraph()
        # 生成全树
        for targnode in nodejson:
            tcondilist = {}
            # 提取公共元素
            for condnode in nodejson:
                if condnode != targnode:
                    common_elem = a_commonset_b(nodejson[condnode]["outjson"], nodejson[targnode]["condjson"])
                    if common_elem:
                        tcondilist[condnode] = common_elem
            # 生成超集
            supersets = genesuperset(tcondilist.values())
            if a_supset_b(supersets, nodejson[targnode]["condjson"]):
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
        logger1.info("use time: {}s".format(time.time() - startt))
        return G

    def deriv_basicelement(self, analist):
        # 衍生一级元素
        # 1. 生成汉语法列表
        # print("deriv_basicelement")
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
            if not isinstance(i1[0], list):
                outlist, outjson = self.language.latex_default_property(i1)
                write_json += outjson
                newoutlist.append(outlist)
            else:
                newoutlist.append(i1)
        return newoutlist, write_json

    def deriv_relationelement(self, analist):
        # 提取所有 已知或求证 的关系
        # 输入: [[['已知'], ['v']], ['{线段@PQ}', '=', '{线段@BP}'], ['{线段@MN}', '\\parallel', '{线段@BC}'], ['{角@BPQ}', '=', '9', '0', '^', '{ \\circ }'], [['求证'], ['v']], ['{线段@BP}', '=', '{线段@PQ}']]
        # 输出:
        purpose_json = []
        # 0. 文本latex标记
        length = len(analist)
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
                if analist[i1][i2] in ['\\cong', '全等']:
                    sigmatch = 1
                    if setstr == "全等" or setstr == "":
                        setstr = "全等"
                    else:
                        raise Exception("全等!={}".format(setstr))
                if analist[i1][i2] in ["相似"]:
                    sigmatch = 1
                    if setstr == "相似" or setstr == "":
                        setstr = "相似"
                    else:
                        raise Exception("相似!={}".format(setstr))
                if analist[i1][i2] in ["等腰三角形"]:
                    sigmatch = 1
                    if setstr == "等腰三角形" or setstr == "":
                        setstr = "等腰三角形"
                    else:
                        raise Exception("等腰三角形!={}".format(setstr))
                if analist[i1][i2] in ["等边三角形"]:
                    sigmatch = 1
                    if setstr == "等边三角形" or setstr == "":
                        setstr = "等边三角形"
                    else:
                        raise Exception("等边三角形!={}".format(setstr))
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
                # print("latex")
                # print(contlist[idn])
                latexlist, propertyjson = self.language.latex_extract_property(contlist[idn])
                outlatex += latexlist
                propertyjson = [{"因为": i1} for i1 in propertyjson]
                self.language.json2space(propertyjson, basic_space_ins, space_ins)
            else:
                # print("text")
                # print(contlist[idn])
                # sentence = HanLP.parseDependency(contlist[idn])
                # word_array =[]
                # for word in sentence.iterator():
                #     word_array.append([word.LEMMA, word.DEPREL, word.HEAD.LEMMA])
                #     # print("%s --(%s)--> %s" % (word.LEMMA, word.DEPREL, word.HEAD.LEMMA))
                # word_array = HanLP.parseDependency(contlist[idn]).getWordArray()
                # outlatex += list(word_array)
                # print(list(jieba.cut(contlist[idn])))
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

    def sentence2normal(self, analist):
        """text latex 句子间合并 按句意合并, 结果全部写入内存。"""
        # 1. 展成 同级 list
        # print("sentence2normal")
        analist = list(itertools.chain(*analist))
        # 2. 去掉空的
        # analist = [{list(sentence.keys())[0]: latex2space(list(sentence.values())[0]).strip(",，。 \t")} for
        analist = [{list(sentence.keys())[0]: list(sentence.values())[0].strip(",，。 \t")} for
                   sentence in analist if list(sentence.values())[0].strip(",，。 \t") != ""]
        # print(analist)
        # 3. 合并 临近相同的
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [list(sentence.values())[0].strip() for sentence in analist]
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
        keylist = [list(sentence.keys())[0] for sentence in analist]
        contlist = [sentence[list(sentence.keys())[0]].strip() for sentence in analist]
        olenth = len(analist)
        if olenth < 2:
            print("latex text 转化 olenth < 2")
            analist = [[{keylist[i1]: contlist[i1]} for i1 in range(olenth)]]
            anastr = self.get_allkeyproperty(analist)
            return anastr
        for i1 in range(olenth - 1, 0, -1):
            # 目前 仅支持两种模式  1. 如： 正方形 ABCD 2. 如：A B C D 在一条直线上
            if keylist[i1] == "latex" and keylist[i1 - 1] == "text":
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
                                    ins_json.append([tnewstr, "是", "点"])
                                    tconcept_list.append(tnewstr)
                                tnewstr = "{ " + jsonkey + "@" + " ".join(tstrstr[0:posind + 1]) + " }"
                                tnewstr = self.language.name_normal(tnewstr)
                                ins_json.append([tnewstr, "是", jsonkey])
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
            if keylist[i1] == "text" and keylist[i1 - 1] == "latex":
                ttypelist = ["在一条直线上", "是锐角"]  # 目前仅支持一种模式: \\angle {xxx}
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
                        if jsonkey == "在一条直线上":
                            posind = -1
                            for i2 in range(siglenth - 1, -1, -1):
                                if siglist[i2] == 1:
                                    posind = i2
                                else:
                                    break
                        elif jsonkey == "是锐角":
                            posind = 0 if siglist[0] > 0 else -1
                        if posind == -1:
                            raise Exception("在一条直线上 或 是锐角 前面不应为空")
                        else:
                            tstrstr = [" ".join(i2) for i2 in tstrli]
                            tconcept_list = []
                            if jsonkey == "在一条直线上":
                                for i2 in range(posind, siglenth):
                                    tnewstr = "{ 点@" + tstrstr[i2] + " }"
                                    tnewstr = self.language.name_normal(tnewstr)
                                    ins_json.append([tnewstr, "是", "点"])
                                    tconcept_list.append(tnewstr)
                                ins_json.append([tconcept_list, "是", "直线"])
                                if 0 != posind:
                                    # 是否删除latex部分
                                    contlist[i1 - 1] = " , ".join(tstrstr[0:posind])
                                else:
                                    del keylist[i1 - 1]
                                    del contlist[i1 - 1]
                            elif jsonkey == "是锐角":
                                tstrli = [i2 for i2 in tstrli[-1] if i2 != "\\angle"]
                                for i2 in tstrli:
                                    tnewstr = "{ 角@" + i2.strip("{}") + " }"
                                    tnewstr = self.language.name_normal(tnewstr)
                                    ins_json.append([tnewstr, "是", "锐角"])
                            else:
                                print(jsonkey)
                                raise Exception("在一条直线上")
        # 5. 写入句间的实例
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        ins_json = [{"因为": i1} for i1 in ins_json]
        self.language.json2space(ins_json, basic_space_ins, space_ins)
        # 6. 提取所有 抽象类。对应实例，改变字符。属性
        olenth = len(contlist)
        analist = [[{keylist[i1]: contlist[i1]}] for i1 in range(olenth)]
        anastr = self.get_allkeyproperty(analist)
        return anastr

    def analyize_strs(self, instr_list):
        """解析字符串到空间: 考虑之前的话语"""
        self.language.nature2space(instr_list, self.gstack)

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
            space_ins._step_node = gene_cond_outjson(tripleobj, space_ins._step_node)
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
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        old_space_setobj = copy.deepcopy(space_ins._setobj)
        # 1. 查找 具体 属性值
        # old_space_setobj = self.prepare_clean_set(old_space_setobj)
        logger1.info("in step {}: {}".format(0, old_space_setobj))
        step_counter = 0
        steplist = {"0": old_space_setobj}
        starttime = time.time()
        while True:
            # 推演步骤 打印出 用到的集合元素属性 和 集合元素属性导出的结果。
            # 根据最终结论，倒寻相关的属性概念。根据年级，忽略非考点的属性，即评判的结果。
            step_counter += 1
            logger1.info("in step {}: {}".format(step_counter, old_space_setobj))
            new_space_setobj = self.step_infere(old_space_setobj)
            steplist[str(step_counter)] = copy.deepcopy(new_space_setobj)
            # 5. 判断终止
            judgeres = self.judge_stop(steplist[str(step_counter - 1)], steplist[str(step_counter)], space_ins._stopobj,
                                       basic_space_ins)
            if step_counter == 90:
                logger1.info("步数超长@{}，停止。".format(step_counter))
                raise Exception("步数超长@{}，停止。".format(step_counter))
            logger1.info("stop inference:{}".format(judgeres[0]))
            if judgeres[1]:
                # 1. 停止操作 写记录
                logger1.info("final step: {}".format(new_space_setobj))
                # 2. 停止操作 变更树写入标记
                logger1.info("writing tree info")
                # self.treesig = True
                # self.step_infere(new_space_setobj)
                break
            old_space_setobj = steplist[str(step_counter)]
        logger1.info("use time:{}hours".format((time.time() - starttime) / 3600))
        # 6. 生成思维树
        return self.get_condition_tree()

    def list_set_deliver(self, inlistset, key):
        "一级列表二级集合，集合传递缩并。如平行 等值 全等"
        # keyset = key
        key = key.replace("集合", "")
        tripleobjlist = []
        inlistset = [setins for setins in inlistset if setins != set()]
        lenth_paralist = len(inlistset)
        dictest = {}
        for indmain in range(lenth_paralist - 1, 0, -1):
            for indcli in range(indmain - 1, -1, -1):
                if len(set(inlistset[indcli]).intersection(set(inlistset[indmain]))) > 0:
                    if operator.eq(set(inlistset[indcli]), set(inlistset[indmain])):
                        # print("delete same")
                        # print(indcli, indmain)
                        # print(inlistset[indcli], inlistset[indmain])
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
                    outjson.append([outsame, "是", purposekey])
                    outsame = list(set(outsame))
                    if self.treesig:
                        tkstr = "".join(set(["".join(objset[indcli]), "".join(objset[indmain])]))
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join(outsame)
                        if tvstr not in dictest[tkstr] and outsame != []:
                            dictest[tkstr].append(tvstr)
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
                                # tripleobjlist.append([[[[0], "是", "默认"]], ["@@{}{}等反传递".format(tarkey, purposekey)],
                                #                       [[list(set([fkey, onout]))], "是", purposekey]])
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
        # print(res)
        return newsetobj

    def points_relations(self, oldsetobj):
        "根据所有点 和 直线，得到 线段 角 和 三角形"
        pointslist = [point.rstrip("}").lstrip("{点@") for point in oldsetobj["点集合"]]
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        outjson = []
        for line in lineslist:
            polist = []
            for point in line:
                tname = self.language.name_symmetric(" ".join(point)).replace(" ", "")
                polist.append("{点@" + tname + "}")
            outjson.append([polist, "是", "直线"])
        # 2. 线段
        for c in combinations(pointslist, 2):
            tname = self.language.name_symmetric(" ".join(c)).replace(" ", "")
            tname = "{线段@" + tname + "}"
            outjson.append([tname, "是", "线段"])
            tpname1 = self.language.name_symmetric(" ".join(c[0:1])).replace(" ", "")
            tpname1 = "{点@" + tpname1 + "}"
            tpname2 = self.language.name_symmetric(" ".join(c[1:])).replace(" ", "")
            tpname2 = "{点@" + tpname2 + "}"
            outjson.append([[tpname1, tpname2], "是", "直线"])
        # 3. 角 三角形
        for c in combinations(pointslist, 3):
            insig = 0
            for oneline in lineslist:
                if set(c).issubset(set(oneline)):
                    insig = 1
                    break
            if insig != 1:
                tname = self.language.name_symmetric(" ".join([c[0], c[1], c[2]])).replace(" ", "")
                outjson.append(["{角@" + tname + "}", "是", "角"])
                tname = self.language.name_symmetric(" ".join([c[0], c[1], c[2]])).replace(" ", "")
                outjson.append(["{角@" + tname + "}", "是", "角"])
                tname = self.language.name_symmetric(" ".join([c[0], c[1], c[2]])).replace(" ", "")
                outjson.append(["{角@" + tname + "}", "是", "角"])
                tname = self.language.name_cyc_one(" ".join(c)).replace(" ", "")
                outjson.append(["{三角形@" + tname + "}", "是", "三角形"])
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

    def vert2Rt_relations(self, oldsetobj):
        " 遍历垂直，得到直角，直角三角形 "
        # 如果 线段的点 全在一条直线上，两条线上的任意一对都垂直。如果垂直的有 共同点，改组为直角。改代表角为直角三角形
        outjson = []
        tripleobjlist = []
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        vertlist = [[segm.rstrip("}").lstrip("{线段@") for segm in segms] for segms in oldsetobj["垂直集合"]]
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
                outjson.append([[tname1, tname2], "是", "垂直"])
                if self.treesig:
                    tkstr = "".join(["默认"])
                    if tkstr not in dictest:
                        dictest[tkstr] = []
                    tvstr = "".join(set([tname1, tname2, "垂直"]))
                    if tvstr not in dictest[tkstr]:
                        dictest[tkstr].append(tvstr)
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
                    outjson.append([tname, "是", "直角"])
                    if self.treesig:
                        tkstr = "".join(["默认"])
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join(set([tname, "直角"]))
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@垂直直角的属性"], [[tname], "是", "直角"]])
                            # tripleobjlist.append([[[[[tseedname1, tseedname2]], "是", "垂直"]], ["@@垂直直角的属性"],
                            #                       [[tname], "是", "直角"]])
                    tname = self.language.name_symmetric(" ".join(insetlist + vertsegm1 + vertsegm2)).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "角"])
                    outjson.append([tname, "是", "锐角"])
                    if self.treesig:
                        tkstr = "".join(["默认"])
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join(set([tname, "锐角"]))
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@余角性质"], [[tname], "是", "锐角"]])
                            # tripleobjlist.append(
                            #     [[[[[tseedname1, tseedname2]], "是", "垂直"]], ["@@余角性质"], [[tname], "是", "锐角"]])
                    tname = self.language.name_symmetric(" ".join(vertsegm1 + vertsegm2 + insetlist)).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "角"])
                    outjson.append([tname, "是", "锐角"])
                    if self.treesig:
                        tkstr = "".join(["默认"])
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join(set([tname, "锐角"]))
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@余角性质"], [[tname], "是", "锐角"]])
                            # tripleobjlist.append(
                            #     [[[[[tseedname1, tseedname2]], "是", "垂直"]], ["@@余角性质"], [[tname], "是", "锐角"]])
                    tname = self.language.name_cyc_one(" ".join(tanlgelist)).replace(" ", "")
                    tname = "{三角形@" + tname + "}"
                    outjson.append([tname, "是", "直角三角形"])
                    if self.treesig:
                        tkstr = "".join(["默认"])
                        if tkstr not in dictest:
                            dictest[tkstr] = []
                        tvstr = "".join(set([tname, "直角三角形"]))
                        if tvstr not in dictest[tkstr]:
                            dictest[tkstr].append(tvstr)
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
            outjson.append([strlist, "是", "平行"])
            if self.treesig:
                tkstr = "".join(["默认"])
                if tkstr not in dictest:
                    dictest[tkstr] = []
                tvstr = "".join(set(strlist))
                if tvstr not in dictest[tkstr]:
                    dictest[tkstr].append(tvstr)
                    # tripleobjlist.append([[[[strlist[0]], "是", "平行"], [idlinegroup, "是", "直线"]], ["@@平行属性"],
                    #                       [[list(set(strlist))], "是", "平行"]])
                    tripleobjlist.append([[[[0], "是", "默认"]], ["@@平行属性"], [[strlist], "是", "平行"]])
        if self.treesig:
            if self.debugsig:
                print("parali2segm_relations")
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
        for ruijiao in oldsetobj["锐角集合"]:
            for equals in oldsetobj["等值集合"]:
                if ruijiao in equals:
                    for elem in equals:
                        outjson.append([elem, "是", "锐角"])
                        if self.treesig:
                            # tripleobjlist.append(
                            #     [[[[[elem, ruijiao]], "是", "等值"], [[ruijiao], "是", "锐角"]], ["@@等值钝角传递"],
                            #      [[elem], "是", "锐角"]])
                            tripleobjlist.append([[[[0], "是", "默认"]], ["@@等值钝角传递"], [[elem], "是", "锐角"]])
        for dunjiao in oldsetobj["钝角集合"]:
            for equals in oldsetobj["等值集合"]:
                if dunjiao in equals:
                    for elem in equals:
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
                vastr = solve_latex_equation(newoneset, varlist=anglelems, const_dic={"\\pi": "3.14"})
                tmpkeys = []
                for angobj in vastr:
                    tkey = list(angobj.keys())[0]
                    tmpkeys.append(tkey)
                # print(tmpkeys)
                for angobj in vastr:
                    tkey = list(angobj.keys())[0]
                    tvalue = str(angobj[tkey])
                    tangl = tvalue.replace("1.57 - 1.0*", "")
                    # print(tkey)
                    # print(tvalue)
                    # print(tangl)
                    if tangl in tmpkeys and tangl != tkey:
                        outjson.append([[tangl, tkey], "是", "余角"])
                        if self.treesig:
                            tripleobjlist.append([[[[oneset], "是", "表达式"]], ["@@表达式性质"], [[[tangl, tkey]], "是", "余角"]])
                            # tripleobjlist.append([[[[0], "是", "默认"]], ["@@表达式性质"], [[[tangl, tkey]], "是", "余角"]])
                    tangl = tvalue.replace("3.14 - 1.0*", "")
                    if tangl in tmpkeys:
                        outjson.append([[tangl, tkey], "是", "补角"])
                        if self.treesig:
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
        # space_name = "basic"
        # basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
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
        # 4. 平行间传递平行。线段是元素，多点直线作为多个元素处理。不同组间有重复的元素，则合并。
        space_ins._setobj = self.parali2segm_relations(oldsetobj)
        # 5. 平行传递垂直。
        oldsetobj = self.list_set_equalanti(space_ins._setobj, tarkey="平行", purposekey="垂直")
        oldsetobj = self.listset_deliverall(oldsetobj)
        space_ins._setobj = oldsetobj
        # 集合缩并
        space_ins._setobj["平行集合"] = self.list_set_deliver(space_ins._setobj["平行集合"], "平行集合")
        oldsetobj = space_ins._setobj
        # 5. 遍历平行，对顶角, 同位角
        oldsetobj["直线集合"] = lines_deliver(oldsetobj["直线集合"])
        oldsetobj = self.corresangles2relations(oldsetobj)
        oldsetobj = self.listset_deliverall(oldsetobj)
        # 所有直角 平角 导入 等值集合
        oldsetobj["等值集合"] += [set([elem]) for elem in oldsetobj["线段集合"]]
        oldsetobj["等值集合"] += [set([elem]) for elem in oldsetobj["角集合"]]
        oldsetobj["等值集合"].append(copy.deepcopy(oldsetobj["直角集合"]))
        oldsetobj["等值集合"].append(copy.deepcopy(oldsetobj["平角集合"]))
        oldsetobj = self.listset_deliverall(oldsetobj)
        space_ins._setobj["等值集合"] = oldsetobj["等值集合"]
        # 6. 等值传递。不同组间有重复的元素，则合并。余角后面和直角三角形一起做
        # 钝角锐角 根据等值传递
        oldsetobj = self.equall2dunrui_relations(oldsetobj)
        oldsetobj = self.degree2angle_relations(oldsetobj)
        # print(sys.getsizeof(space_ins._setobj))
        # 7. 遍历直角三角形 垂直已导出角可以略过，得到余角
        oldsetobj = self.rttriang2remain_relations(oldsetobj)
        oldsetobj = self.listset_deliverall(oldsetobj)
        # 8. 表达式得出补角余角集合
        oldsetobj = self.express2compleremain_relations(oldsetobj)
        oldsetobj = self.express2relations(oldsetobj)
        # 9. 余角 反等传递
        oldsetobj = self.list_set_antiequal(oldsetobj, tarkey="余角", purposekey="等值")
        # 10. 补角 反等传递
        oldsetobj = self.list_set_antiequal(oldsetobj, tarkey="补角", purposekey="等值")
        # 11. 垂直 反等传递
        oldsetobj = self.list_set_antiequal(oldsetobj, tarkey="垂直", purposekey="平行")
        oldsetobj = self.listset_deliverall(oldsetobj)
        space_ins._setobj = oldsetobj
        # 10. 删除空的集合
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
                tname = self.language.name_symmetric(" ".join([tanglist[idangle], tanglist[idangle + 2],
                                                               tanglist[idangle + 1]])).replace(" ", "")
                tname = "{角@" + tname + "}"
                outjson.append([tname, "是", "角"])
                outjson.append([tname, "是", "锐角"])
                tname = self.language.name_cyc_one(" ".join(tanglist[idangle:idangle + 3])).replace(" ", "")
                tname = "{三角形@" + tname + "}"
                outjson.append([tname, "是", "三角形"])
                outjson.append([tname, "是", "直角三角形"])
                tripleobjlist.append([[[[obj], "是", "正方形"]], ["@@正方形直角属性"], [[tname], "是", "直角三角形"]])
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
                                    tkstr = "".join(set([eae_list[idmain][-1], eae_list[idcli][-1]]))
                                    if tkstr not in dictest:
                                        dictest[tkstr] = []
                                    tvstr = "".join(set([eae_list[idmain][2], eae_list[idcli][2]]))
                                    if tvstr not in dictest[tkstr]:
                                        dictest[tkstr].append(tvstr)
                                        tripleobjlist.append(
                                            [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                             ["@@全等三角形必要条件"],
                                             [[[eae_list[idmain][2], eae_list[idcli][2]]], "是", "等值"]])
                                    tvstr = "".join(set([aea_list[idmain][0], aea_list[idcli][0]]))
                                    if tvstr not in dictest[tkstr]:
                                        dictest[tkstr].append(tvstr)
                                        tripleobjlist.append(
                                            [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                             ["@@全等三角形必要条件"],
                                             [[[aea_list[idmain][0], aea_list[idcli][0]]], "是", "等值"]])
                                if aea_sig[1] == 1 and aea_sig[4] == 1:
                                    outjson.append([[eae_list[idmain][1], eae_list[idcli][1]], "是", "等值"])
                                    outjson.append([[eae_list[idmain][0], eae_list[idcli][0]], "是", "等值"])
                                    if self.treesig:
                                        tkstr = "".join(set([eae_list[idmain][-1], eae_list[idcli][-1]]))
                                        if tkstr not in dictest:
                                            dictest[tkstr] = []
                                        tvstr = "".join(set([aea_list[idmain][1], aea_list[idcli][1]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][1], aea_list[idcli][1]]], "是", "等值"]])
                                        tvstr = "".join(set([aea_list[idmain][1], aea_list[idcli][1]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][1], aea_list[idcli][1]]], "是", "等值"]])
                                        tvstr = "".join(set([eae_list[idmain][0], eae_list[idcli][0]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[eae_list[idmain][0], eae_list[idcli][0]]], "是", "等值"]])
                                        tvstr = "".join(set([aea_list[idmain][2], aea_list[idcli][2]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][2], aea_list[idcli][2]]], "是", "等值"]])
                                elif aea_sig[2] == 1 and aea_sig[3] == 1:
                                    outjson.append([[eae_list[idmain][1], eae_list[idcli][0]], "是", "等值"])
                                    outjson.append([[eae_list[idmain][0], eae_list[idcli][1]], "是", "等值"])
                                    if self.treesig:
                                        tkstr = "".join(set([eae_list[idmain][-1], eae_list[idcli][-1]]))
                                        if tkstr not in dictest:
                                            dictest[tkstr] = []
                                        tvstr = "".join(set([eae_list[idmain][1], eae_list[idcli][0]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[eae_list[idmain][1], eae_list[idcli][0]]], "是", "等值"]])
                                        tvstr = "".join(set([aea_list[idmain][1], aea_list[idcli][2]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][1], aea_list[idcli][2]]], "是", "等值"]])
                                        tvstr = "".join(set([eae_list[idmain][0], eae_list[idcli][1]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[eae_list[idmain][0], eae_list[idcli][1]]], "是", "等值"]])
                                        tvstr = "".join(set([aea_list[idmain][2], aea_list[idcli][1]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][2], aea_list[idcli][1]]], "是", "等值"]])
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
                                        tkstr = "".join(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                        if tkstr not in dictest:
                                            dictest[tkstr] = []
                                        tvstr = "".join(set([aea_list[idmain][1], aea_list[idcli][1]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][1], aea_list[idcli][1]]], "是", "等值"]])
                                        tvstr = "".join(set([eae_list[idmain][1], eae_list[idcli][1]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[eae_list[idmain][1], eae_list[idcli][1]]], "是", "等值"]])
                                        tvstr = "".join(set([aea_list[idmain][2], aea_list[idcli][2]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][2], aea_list[idcli][2]]], "是", "等值"]])
                                        tvstr = "".join(set([eae_list[idmain][0], eae_list[idcli][0]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[eae_list[idmain][0], eae_list[idcli][0]]], "是", "等值"]])
                                elif eae_sig[1] == 1 and eae_sig[2] == 1:
                                    outjson.append([[aea_list[idmain][1], aea_list[idcli][2]], "是", "等值"])
                                    outjson.append([[aea_list[idmain][2], aea_list[idcli][1]], "是", "等值"])
                                    if self.treesig:
                                        tkstr = "".join(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                        if tkstr not in dictest:
                                            dictest[tkstr] = []
                                        tvstr = "".join(set([aea_list[idmain][1], aea_list[idcli][2]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][1], aea_list[idcli][2]]], "是", "等值"]])
                                        tvstr = "".join(set([eae_list[idmain][1], eae_list[idcli][0]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[eae_list[idmain][1], eae_list[idcli][0]]], "是", "等值"]])
                                        tvstr = "".join(set([aea_list[idmain][2], aea_list[idcli][1]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[aea_list[idmain][2], aea_list[idcli][1]]], "是", "等值"]])
                                        tvstr = "".join(set([eae_list[idmain][0], eae_list[idcli][1]]))
                                        if tvstr not in dictest[tkstr]:
                                            dictest[tkstr].append(tvstr)
                                            tripleobjlist.append(
                                                [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                                 ["@@全等三角形必要条件"],
                                                 [[[eae_list[idmain][0], eae_list[idcli][1]]], "是", "等值"]])
                                outjson.append([[aea_list[idmain][0], aea_list[idcli][0]], "是", "等值"])
                                if self.treesig:
                                    tkstr = "".join(set([aea_list[idmain][-1], aea_list[idcli][-1]]))
                                    if tkstr not in dictest:
                                        dictest[tkstr] = []
                                    tvstr = "".join(set([aea_list[idmain][0], aea_list[idcli][0]]))
                                    if tvstr not in dictest[tkstr]:
                                        dictest[tkstr].append(tvstr)
                                        tripleobjlist.append(
                                            [[[[[aea_list[idmain][-1], aea_list[idcli][-1]]], "是", "全等三角形"]],
                                             ["@@全等三角形必要条件"],
                                             [[[aea_list[idmain][0], aea_list[idcli][0]]], "是", "等值"]])
                                    tvstr = "".join(set([eae_list[idmain][2], eae_list[idcli][2]]))
                                    if tvstr not in dictest[tkstr]:
                                        dictest[tkstr].append(tvstr)
                                        tripleobjlist.append(
                                            [[[[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是", "全等三角形"]],
                                             ["@@全等三角形必要条件"],
                                             [[[eae_list[idmain][2], eae_list[idcli][2]]], "是", "等值"]])
        if self.treesig:
            if self.debugsig:
                print("congruent_triangle2elements")
            self.step_node_write(tripleobjlist)
        return self.math_solver_write(outjson)

    def conception2element(self, oldsetobj):
        " 根据概念属性 衍生，点 线段 角 三角形，去掉顺序差异，再根据直线 衍生等值角"
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
        return space_ins._setobj

    def element2conception(self, oldsetobj):
        "元素衍生概念"
        logger1.info("in element2conception")
        # 1. 得出 全等三角形
        triang_pointlist = [elems.rstrip("}").lstrip("{三角形@") for elems in oldsetobj["三角形集合"]]
        triang_pointlist = [latex_fenci(latex2space(angli)) for angli in triang_pointlist]
        # 1.1 边角边
        tripleobjlist = []
        outjson = []
        eae_list = []
        aea_list = []
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
        comb_lenth = len(aea_list)
        for idmain in range(comb_lenth - 1, 0, -1):
            for idcli in range(idmain - 1, -1, -1):
                # 判断边
                aea_sig = [0, 0, 0, 0, 0]
                eae_sig = [0, 0, 0, 0, 0]
                for equset in oldsetobj["等值集合"]:
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
                        # if sum(aea_sig) > 2:
                        #     print("aea_sig", aea_sig)
                        if aea_sig[0] == 1 and (aea_sig[1] + aea_sig[4] == 2 or aea_sig[2] + aea_sig[3] == 2):
                            # if aea_list[idmain][-1] in ["{三角形@BMP}", "{三角形@NPQ}"] and aea_list[idcli][-1] in ["{三角形@BMP}",
                            #                                                                                 "{三角形@NPQ}"]:
                            #     print("{三角形@BMP}", "{三角形@NPQ}")
                            #     print(aea_list[idmain][0], aea_list[idcli][0])
                            #     print(aea_list[idmain][1], aea_list[idcli][1])
                            #     print(aea_list[idmain][2], aea_list[idcli][2])
                            #     print(aea_list[idmain][1], aea_list[idcli][2])
                            #     print(aea_list[idmain][2], aea_list[idcli][1])
                            outjson.append([[aea_list[idmain][-1], aea_list[idcli][-1]], "是", "全等三角形"])
                            if self.treesig:
                                keyelem = [aea_list[idmain][0], aea_list[idcli][0]]
                                taea_equal = []
                                taea_equal.append([[[aea_list[idmain][0], aea_list[idcli][0]]], "是", "等值"])
                                if aea_sig[1] + aea_sig[4] == 2:
                                    taea_equal.append([[[aea_list[idmain][1], aea_list[idcli][1]]], "是", "等值"])
                                    taea_equal.append([[[aea_list[idmain][2], aea_list[idcli][2]]], "是", "等值"])
                                    keyelem += [aea_list[idmain][1], aea_list[idcli][1]]
                                    keyelem += [aea_list[idmain][2], aea_list[idcli][2]]
                                if aea_sig[2] + aea_sig[3] == 2:
                                    taea_equal.append([[[aea_list[idmain][1], aea_list[idcli][2]]], "是", "等值"])
                                    taea_equal.append([[[aea_list[idmain][2], aea_list[idcli][1]]], "是", "等值"])
                                    keyelem += [aea_list[idmain][1], aea_list[idcli][2]]
                                    keyelem += [aea_list[idmain][2], aea_list[idcli][1]]
                                tkstr = "".join(keyelem)
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
                        # if sum(eae_sig) > 2:
                        #     print("eae_sig", eae_sig)
                        #     print(eae_list[idmain][-1], eae_list[idcli][-1])
                        if eae_sig[4] == 1 and (eae_sig[0] + eae_sig[3] == 2 or eae_sig[1] + eae_sig[2] == 2):
                            outjson.append([[eae_list[idmain][-1], eae_list[idcli][-1]], "是", "全等三角形"])
                            if self.treesig:
                                keyelem = [eae_list[idmain][2], eae_list[idcli][2]]
                                teae_equal = []
                                teae_equal.append([[[eae_list[idmain][2], eae_list[idcli][2]]], "是", "等值"])
                                if eae_sig[0] + eae_sig[3] == 2:
                                    teae_equal.append([[[eae_list[idmain][0], eae_list[idcli][0]]], "是", "等值"])
                                    teae_equal.append([[[eae_list[idmain][1], eae_list[idcli][1]]], "是", "等值"])
                                    keyelem += [eae_list[idmain][1], eae_list[idcli][1]]
                                    keyelem += [eae_list[idmain][0], eae_list[idcli][0]]
                                if eae_sig[1] + eae_sig[2] == 2:
                                    teae_equal.append([[[eae_list[idmain][0], eae_list[idcli][1]]], "是", "等值"])
                                    teae_equal.append([[[eae_list[idmain][1], eae_list[idcli][0]]], "是", "等值"])
                                    keyelem += [eae_list[idmain][1], eae_list[idcli][0]]
                                    keyelem += [eae_list[idmain][0], eae_list[idcli][1]]
                                tkstr = "".join(keyelem)
                                if tkstr not in dictest:
                                    dictest[tkstr] = []
                                tvstr = "".join([aea_list[idmain][-1], aea_list[idcli][-1]])
                                if tvstr not in dictest[tkstr]:
                                    dictest[tkstr].append(tvstr)
                                    tripleobjlist.append([teae_equal, ["@@全等三角形充分条件边角边"],
                                                          [[[eae_list[idmain][-1], eae_list[idcli][-1]]], "是",
                                                           "全等三角形"]])
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        if self.treesig:
            if self.debugsig:
                print("element2conception")
            self.step_node_write(tripleobjlist)
        space_ins._setobj = self.math_solver_write(outjson)
        space_ins._setobj = self.listset_deliverall(space_ins._setobj)
        return space_ins._setobj

    def judge_stop(self, oldsetobj, newsetobj, stopobj, basic_space_ins):
        "每步推理的具体操作 true为应该结束"
        if operator.eq(oldsetobj, newsetobj):
            message = "已知条件无法进一步推理"
            return message, True
        for key in stopobj.keys():
            if setobj[key]["结构形式"] == ["一级集合", "一级列表"]:
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
        # return message, False
        return message, True


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


def answer_latex_prove(instr, inconditon, intree):
    "输入：解答字符串，输出：序列化要素，相关知识点报告"
    # 1. 答案字符标准化
    ans_inlist = recog_str2list(instr)
    # 2. 分解答案
    li_ins = LogicalInference()
    anastr = li_ins.sentence2normal(ans_inlist)
    anastr = li_ins.analysis_tree(anastr, inconditon, intree)
    print(anastr)
    outelem = "res"
    outreport = "res"
    return outelem, outreport


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
    # printstr3 = "已知：四边形 $ABCD$ 中 ， $AD\\parallel BC , Ac=bf $，$AC=BD$ ，\n 是不是容易求证 ：$AB=DC$"
    # printstr3 = "某村计划建造如图所示的矩形蔬菜温室，要求长与宽的比为$2:1$．在温室内，沿前侧内墙保留$3m$宽的空地，其他三侧内墙各保留$1m$宽的通道．当矩形温室的长与宽各为多少米时，蔬菜种植区域的面积是$288m^{2}$？"
    # printstr3 = "证明：\\\n 联结 $CE$ \\\n $\\because \\angle{ACB}=90^{\\circ}\\qquad AE=BE$ \\\n $\\therefore CE=AE=\\frac{1}{2}AB$ \\\n 又 $\\because CD=\\frac{1}{2}AB$ \\\n $\\therefore CD=CE$ \\\n $\\therefore \\angle{CED}=\\angle{CDE}$ \\\n 又 $\\because A 、C、 D$ 成一直线 \\\n $\\therefore \\angle{ECA}=\\angle{CED}+\\angle{CDE}$ \\\n $=2\\angle{CDE}$ \\\n $\\angle{CDE}=\\frac{1}{2}\\angle{ECA}$ \\\n 又 $\\because EC=EA$ \\\n $\\therefore \\angle{ECA}=\\angle{EAC}$ \\\n $\\therefore \\angle{ADG}=\\frac{1}{2}\\angle{EAC}$ \\\n 又 $\\because AG$ 是 $\\angle{BAC}$ 的角平分线 \\\n $\\therefore \\angle{GAD}=\\frac{1}{2}\\angle{EAC}$ \\\n $\\therefore \\angle{GAD}=\\angle{GDA}$ \\\n $\\therefore GA=GD$"
    # printstr3 = "$\\therefore \\angle{ECA}=\\angle{CED}+\\angle{CDE}$"
    # printstr3 = "$\\therefore CE=AE=\\frac{1}{2}AB$"
    # printstr3 = "已知：\\\n 联结 $CE$ \\\n $\\because \\angle{ACB}=90^{\\circ}\\qquad AE=BE$ \\\n $\\therefore CE=AE=\\frac{1}{2}AB$ \\\n 又 $\\because CD=\\frac{1}{2}AB$ \\\n $\\therefore CD=CE$ \\\n $\\therefore \\angle{CED}=\\angle{CDE}$ \\\n 又 $\\because A 、C、 D$ 成一直线 \\\n $\\therefore \\angle{ECA}=\\angle{CED}+\\angle{CDE}$ \\\n $=2\\angle{CDE}$ \\\n $\\angle{CDE}=\\frac{1}{2}\\angle{ECA}$ \\\n 又 $\\because EC=EA$ \\\n $\\therefore \\angle{ECA}=\\angle{EAC}$ \\\n $\\therefore \\angle{ADG}=\\frac{1}{2}\\angle{EAC}$ \\\n 又 $\\because AG$ 是 $\\angle{BAC}$ 的角平分线 \\\n $\\therefore \\angle{GAD}=\\frac{1}{2}\\angle{EAC}$ \\\n $\\therefore \\angle{GAD}=\\angle{GDA}$ \\\n $\\therefore GA=GD$"
    # printstr3 = "已知：正方形 $ABCD, A、P、C $ 在一条直线上。$MN \\parallel BC, \\angle {BPQ} =90 ^{\\circ},A、M、B $ 在一条直线上，$\\angle {APM}$是锐角，$\\angle {ACB}$是锐角。 $C、Q、 N、D $ 在一条直线上。求证 $PB = PQ$"
    # 演示版本 全等三角形
    # printstr3 = "已知：三角形 $ABC, \\triangle {ACD}, \\angle {CAB} = \\angle {CAD}, \\angle {ACB} = \\angle {ACD} = 30 ^{\\circ} $。求证 $BC = CD$"
    # 中等难度 多边形
    printstr3 = "已知：正方形 $ABCD, A、P、C $ 在一条直线上。$MN \\parallel BC, \\angle {BPQ} =90 ^{\\circ},A、M、B $ 在一条直线上，$M、P、N $ 在一条直线上，$\\angle {APM}$是锐角，$\\angle {NPQ} +\\angle {BPQ} +\\angle {BPM} =180^{\\circ }, \\angle {ACB}$是锐角。 $C、Q、 N、D $ 在一条直线上。求证 $PB = PQ$"
    # 对的 1
    handestr3 = "$ \\therefore AM=PM, \\because AB=MN,\\therefore MB=PN,\\because \\angle {BPQ}=90 ^ {\\circ},\\therefore \\angle {BPM} + \\angle {NPQ} = 90 ^ {\\circ},\\because \\angle {MBP} + \\angle {BPM} = 90 ^ {\\circ},\\therefore \\angle {MBP} = \\angle {NPQ},\\because \\triangle {BPM}$ 是直角三角形。$\\because \\triangle {NPQ}$ 是直角三角形$ \\therefore \\triangle {BPM} \\cong \\triangle {NPQ},\\therefore PB = PQ $"
    # 对的 2
    handestr3 = "$ \\therefore AM=PM, \\because AB=MN,\\therefore MB=PN,\\because \\angle {BPQ}=90 ^ {\\circ},\\therefore \\angle {BPM} + \\angle {NPQ} = 90 ^ {\\circ},\\because \\angle {NPQ} + \\angle {NQP} = 90 ^ {\\circ},\\therefore \\angle {MPB} = \\angle {NQP},\\because \\triangle {BPM}$ 是直角三角形。$\\because \\triangle {NPQ}$ 是直角三角形$ \\therefore \\triangle {BPM} \\cong \\triangle {NPQ},\\therefore PB = PQ $"
    # \\therefore PB = PQ
    outelem, outtree = title_latex_prove(printstr3)
    outtree = None
    outelem, outreport = answer_latex_prove(handestr3, outtree)
    print("end")
    raise 456
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
