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

以 _ 开头的 是id 或 函数 

条件缩并函数:

抽象元类:{
  角:{
  },
  3角形:{
  }
}

抽象元类列表:[角,3角形]
抽象元类列表:{
  id: [ 角 ]
  id: [ 3角形 ]
}

对象关系:{
  id: [ xxx, 是, 表达式 ],
  id: [ {abc}, 是, 角 ],
}

属性对象:{
  id:[a, is, 角],
  id:[b, is, 3角形],
}

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
import jsonpatch
import operator
from latex_solver import latex2list_P, postfix_convert_P, latex2space, latex2unit
from latex_solver import solve_latex_formula2, solve_latex_equation
from latex_solver import latex_json, baspath, step_alist, step_blist, symblist, pmlist, addtypelist, funclist, operlist
from meta_property import triobj, properobj, setobj
from utils.path_tool import makesurepath

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


def list_set_deliver(inlistset):
    "一级列表二级集合，集合传递缩并。如平行 等值"
    inlistset = [setins for setins in inlistset if setins != set()]
    lenth_paralist = len(inlistset)
    for indmain in range(lenth_paralist - 1, 0, -1):
        for indcli in range(indmain - 1, -1, -1):
            if len(inlistset[indcli].intersection(inlistset[indmain])) > 0:
                # print(754)
                # print(inlistset[indcli])
                # print(inlistset[indmain])
                # print(inlistset[indcli].intersection(inlistset[indmain]))
                # print([i21 for i21 in inlistset[indmain] if i21 not in inlistset[indcli].intersection(inlistset[indmain])])
                inlistset[indcli] |= inlistset[indmain]
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
            self._proper_keys, self._proper_trip, self._relation_trip, self._setobj = self.storage_oper("r")
        else:
            self._proper_keys, self._proper_trip = {}, {}
            self._relation_trip = {}
            self._questproperbj = {}
            self._questtriobj = {}
            self._setobj = {"锐角集合": set(), "钝角集合": set(), "等价集合": [], "全等集合": [], "全等三角形集合": [],
                            "垂直集合": [], "平行集合": [], "直角集合": set(), "平角集合": set(), "直角三角形集合": set(),
                            "余角集合": [], "补角集合": []}
            self._stopobj = {}

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


class Steps(object):
    """ 步骤: """

    def __init__(self, oldspace, basicspace):
        self.step_name = None
        # 临时分两种 easy detail
        self.inference_type = "easy"
        self.out_type = "easy"
        self.basicspace = basicspace
        self.oldspace = oldspace
        self.newspace = copy.deepcopy(oldspace)

    def __call__(self, *args, **kwargs):
        return self.inference()

    def inference(self, ):
        self.newspace.inference()
        return self.newspace


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
        # 0. 处理句间 关系，写入部分实体。 基于符号类型的区分标签。结果全部写入内存。
        anastr = self.sentence2normal(analist)
        logger1.info("initial clean sentence: %s" % analist)
        # 1. 循环处理每句话 生成空间解析 和 步骤list
        # for sentence in analist:
        #     logger1.info("sentence: %s" % sentence)
        # 2. 基于因果的符号标签
        # self.analyize_strs(anastr)
        # 2. 内存推理，基于之前的步骤条件
        self.inference()

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
        # print(newlist)
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
        # print("deriv_relationelement")
        # print(analist)
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
        # print(space_ins._setobj)
        # print(outlatex)
        outlatex, propertyjson = self.deriv_basicelement(outlatex)
        propertyjson = [{"因为": i1} for i1 in propertyjson]
        self.language.json2space(propertyjson, basic_space_ins, space_ins)
        # 4. 语法提取 字面关系
        propertyjson = self.deriv_relationelement(outlatex)
        self.language.json2space(propertyjson, basic_space_ins, space_ins)
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

    def inference(self):
        """推理流程: 三元组 到 三元组"""
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        old_space_setobj = copy.deepcopy(space_ins._setobj)
        # 查找 具体 属性值
        step_counter = 0
        steplist = {"0": old_space_setobj}
        while True:
            # 推演步骤 打印出 用到的集合元素属性 和 集合元素属性导出的结果。
            # 根据最终结论，倒寻相关的属性概念。根据年级，忽略非考点的属性，即评判的结果。
            step_counter += 1
            logger1.info("in step {}: {}".format(step_counter, old_space_setobj))
            new_space_setobj = self.step_infere(old_space_setobj)
            logger1.info("out step {}: {}".format(step_counter, new_space_setobj))
            steplist[str(step_counter)] = copy.deepcopy(new_space_setobj)
            # 5. 判断终止
            judgeres = self.judge_stop(old_space_setobj, steplist[str(step_counter)], space_ins._stopobj,
                                       basic_space_ins)
            if step_counter == 90:
                exit()
            logger1.info("stop inference:{}".format(judgeres[0]))
            if judgeres[1]:
                break
            old_space_setobj = steplist[str(step_counter)]
        # 6. 生成思维树
        for items in steplist.items():
            print(items)
        raise Exception("end")
        return None

    def listset_deliverall(self, allobjset):
        for key in allobjset.keys():
            if setobj[key]["结构形式"] in ["一级列表二级集合"] and "二级传递" in setobj[key]["函数"]:
                allobjset[key] = list_set_deliver(allobjset[key])
        return allobjset

    def step_infere(self, oldsetobj):
        "每步推理的具体操作"
        print("step_infere")
        # print(oldsetobj)
        # 1. 概念属性 衍生关系
        newsetobj = self.conception2element(oldsetobj)
        # 2. 公理 衍生关系
        newsetobj = self.axiom2relation(newsetobj)
        # 3. 属性 提取 概念
        newsetobj = self.element2conception(newsetobj)
        # res = space_ins.find_obj_property_value(obj="矩形", property="面积")
        # print(res)
        # # 猜谜查找具体 实体
        # res = space_ins.find_property_value_child(obj="四边形", property="对角线", value="相等")
        # print(res)
        return newsetobj

    def axiom2relation(self, oldsetobj):
        "精确概念的自洽"
        print("axiom2relation")
        # 0. 空间定义
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        # 1. 遍历点，得到线段 和 角
        # print(oldsetobj)
        pointslist = [point.rstrip("}").lstrip("{点@") for point in oldsetobj["点集合"]]
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        # 直线合并
        lineslist = lines_deliver(lineslist)
        outjson = []
        for line in lineslist:
            polist = []
            for point in line:
                tname = self.language.name_symmetric(" ".join(point)).replace(" ", "")
                polist.append("{点@" + tname + "}")
            outjson.append([polist, "是", "直线"])
        outjson = [{"因为": i1} for i1 in outjson]
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=outjson, delec=[])
        outjson = []
        # 线段
        for c in combinations(pointslist, 2):
            tname = self.language.name_symmetric(" ".join(c)).replace(" ", "")
            tname = "{线段@" + tname + "}"
            outjson.append([tname, "是", "线段"])
            tpname1 = self.language.name_symmetric(" ".join(c[0:1])).replace(" ", "")
            tpname1 = "{点@" + tpname1 + "}"
            tpname2 = self.language.name_symmetric(" ".join(c[1:])).replace(" ", "")
            tpname2 = "{点@" + tpname2 + "}"
            outjson.append([[tpname1, tpname2], "是", "直线"])
        # 角 三角形
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
        # 2. 遍历直线，得到补角
        for oneline in lineslist:
            lenth_line = len(oneline)
            if lenth_line > 2:
                nolinepoint = [point for point in pointslist if point not in oneline]
                line_dic = {point: idp for idp, point in enumerate(oneline)}
                # 平角
                for ang_p in combinations(oneline, 3):
                    ang_plist = [[point, line_dic[point]] for point in ang_p]
                    ang_plist = [item[0] for item in sorted(ang_plist, key=lambda x: x[1])]
                    tname = self.language.name_symmetric(" ".join(ang_plist)).replace(" ", "")
                    outjson.append(["{角@" + tname + "}", "是", "平角"])
                    # 补角
                    for outpoint in nolinepoint:
                        t_ang_plist1 = ang_plist[0:2] + [outpoint]
                        t_ang_plist2 = [outpoint] + ang_plist[1:]
                        tname1 = self.language.name_symmetric(" ".join(t_ang_plist1)).replace(" ", "")
                        tname1 = "{角@" + tname1 + "}"
                        tname2 = self.language.name_symmetric(" ".join(t_ang_plist2)).replace(" ", "")
                        tname2 = "{角@" + tname2 + "}"
                        outjson.append([[tname1, tname2], "是", "补角"])
                        ttype1 = "钝角" if tname2 in oldsetobj["锐角集合"] else ""
                        ttype1 = "锐角" if tname2 in oldsetobj["钝角集合"] else ttype1
                        ttype2 = "钝角" if tname1 in oldsetobj["锐角集合"] or ttype1 == "锐角" else ""
                        ttype2 = "锐角" if tname1 in oldsetobj["钝角集合"] or ttype1 == "钝角" else ttype2
                        ttype1 = "钝角" if ttype2 == "锐角" and ttype1 == "" else ""
                        ttype1 = "锐角" if ttype2 == "钝角" and ttype1 == "" else ""
                        if ttype1 != "":
                            outjson.append([tname1, "是", ttype1])
                        if ttype2 != "":
                            outjson.append([tname2, "是", ttype2])
        # 写入
        outjson = [{"因为": i1} for i1 in outjson]
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=outjson, delec=[])
        space_ins._setobj["直线集合"] = lines_deliver(space_ins._setobj["直线集合"])
        oldsetobj = space_ins._setobj
        outjson = []
        # 3. 遍历垂直，得到直角，直角三角形
        vertlist = [[segm.rstrip("}").lstrip("{线段@") for segm in segms] for segms in oldsetobj["垂直集合"]]
        vertlist = [[latex_fenci(latex2space(item2)) for item2 in item1] for item1 in vertlist]
        for segm1, segm2 in vertlist:
            segmlist1, segmlist2 = [segm1], [segm2]
            # 根据直线获取垂直组
            for oneline in lineslist:
                if set(segm1).issubset(set(oneline)):
                    segmlist1 = [segm for segm in combinations(oneline, 2)]
                if set(segm2).issubset(set(oneline)):
                    segmlist2 = [segm for segm in combinations(oneline, 2)]
            # 垂直组 互相组合
            for vertsegm1, vertsegm2 in itertools.product(segmlist1, segmlist2):
                vertsegm1 = list(vertsegm1)
                vertsegm2 = list(vertsegm2)
                tname1 = self.language.name_symmetric(" ".join(vertsegm1)).replace(" ", "")
                tname2 = self.language.name_symmetric(" ".join(vertsegm2)).replace(" ", "")
                outjson.append([["{线段@" + tname1 + "}", "{线段@" + tname2 + "}"], "是", "垂直"])
                insetlist = list(set(vertsegm1).intersection(set(vertsegm2)))
                # 有公共点 生成角和三角形
                if len(insetlist) == 1:
                    vertsegm1.remove(insetlist[0])
                    vertsegm2.remove(insetlist[0])
                    tanlgelist = vertsegm1 + insetlist + vertsegm2
                    tname = self.language.name_symmetric(" ".join(tanlgelist)).replace(" ", "")
                    outjson.append(["{角@" + tname + "}", "是", "直角"])
                    tname = self.language.name_symmetric(" ".join(insetlist + vertsegm1 + vertsegm2)).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "角"])
                    outjson.append([tname, "是", "锐角"])
                    tname = self.language.name_symmetric(" ".join(vertsegm1 + vertsegm2 + insetlist)).replace(" ", "")
                    tname = "{角@" + tname + "}"
                    outjson.append([tname, "是", "角"])
                    outjson.append([tname, "是", "锐角"])
                    tname = self.language.name_cyc_one(" ".join(tanlgelist)).replace(" ", "")
                    outjson.append(["{三角形@" + tname + "}", "是", "直角三角形"])
        outjson = [{"因为": i1} for i1 in outjson]
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=outjson, delec=[])
        oldsetobj = space_ins._setobj
        outjson = []
        # 如果 线段的点 全在一条直线上，两条线上的任意一对都垂直。如果垂直的有 共同点，改组为直角。改代表角为直角三角形
        # 4. 平行传递。线段是元素，多点直线作为多个元素处理。不同组间有重复的元素，则合并。
        paralist = [[segm.rstrip("}").lstrip("{线段@") for segm in segms] for segms in oldsetobj["平行集合"]]
        paralist = [[latex_fenci(latex2space(item2)) for item2 in item1] for item1 in paralist]
        for onegroup in paralist:
            newgroup = []
            for segmi in onegroup:
                newgroup.append(segmi)
                for oneline in lineslist:
                    if set(segmi).issubset(set(oneline)):
                        newgroup += [segm for segm in combinations(oneline, 2)]
                        break
            strlist = []
            for segmi in newgroup:
                strlist.append(self.language.name_symmetric(" ".join(segmi)).replace(" ", ""))
            strlist = ["{线段@" + segmi + "}" for segmi in strlist]
            outjson.append([strlist, "是", "平行"])
        outjson = [{"因为": i1} for i1 in outjson]
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=outjson, delec=[])
        # 集合缩并
        space_ins._setobj["平行集合"] = list_set_deliver(space_ins._setobj["平行集合"])
        oldsetobj = space_ins._setobj
        outjson = []
        # print(sys.getsizeof(space_ins._setobj))
        # 5. 遍历平行，对顶角
        lineslist = [[point.rstrip("}").lstrip("{点@") for point in points] for points in oldsetobj["直线集合"]]
        paralist = [[segm.rstrip("}").lstrip("{线段@") for segm in segms] for segms in oldsetobj["平行集合"]]
        paralist = [[latex_fenci(latex2space(item2)) for item2 in item1] for item1 in paralist]
        for idl, line in enumerate(lineslist):
            noselflines = copy.deepcopy(lineslist)
            del noselflines[idl]
            corres_updn_12ang = []
            for onegroup in paralist:
                # 5.1得出 交点
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
                            corres_updn_12ang.append([inner_upp, inner_upn, inner_dnp, inner_dnn])
            # 根据类型判断相等
            acute_set = []
            obtuse_set = []
            for onegroup in corres_updn_12ang:
                for angli in onegroup:
                    if len(set(angli).intersection(space_ins._setobj["锐角集合"])) > 0:
                        acute_set += angli
                    elif len(set(angli).intersection(space_ins._setobj["钝角集合"])) > 0:
                        obtuse_set += angli
                    else:
                        pass
            outjson.append([acute_set, "是", "等值"])
            outjson.append([obtuse_set, "是", "等值"])
        outjson = [{"因为": i1} for i1 in outjson]
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=outjson, delec=[])
        oldsetobj = space_ins._setobj
        outjson = []
        # 所有直角 平角 导入 等值集合
        # print("直角集合6")
        oldsetobj = self.listset_deliverall(oldsetobj)
        oldsetobj["等值集合"] += [set([elem]) for elem in oldsetobj["线段集合"]]
        oldsetobj["等值集合"] += [set([elem]) for elem in oldsetobj["角集合"]]
        oldsetobj["等值集合"].append(copy.deepcopy(oldsetobj["直角集合"]))
        oldsetobj["等值集合"].append(copy.deepcopy(oldsetobj["平角集合"]))
        oldsetobj = self.listset_deliverall(oldsetobj)
        # 6. 等值传递。不同组间有重复的元素，则合并。余角后面和直角三角形一起做
        # 钝角锐角 根据等值传递
        space_ins._setobj["等值集合"] = list_set_deliver(oldsetobj["等值集合"])
        for ruijiao in oldsetobj["锐角集合"]:
            for equals in oldsetobj["等值集合"]:
                if ruijiao in equals:
                    for elem in equals:
                        outjson.append([elem, "是", "锐角"])
        for dunjiao in oldsetobj["钝角集合"]:
            for equals in oldsetobj["等值集合"]:
                if dunjiao in equals:
                    for elem in equals:
                        outjson.append([elem, "是", "钝角"])
        # 找等于90度和180度的，作为直角 直角三角形 补角 直线。
        for oneset in oldsetobj["等值集合"]:
            # 每个集合中找非属性的表达式，如果计算值小于误差，则为直角 或 平角
            findsig = 0
            for elem in oneset:
                if "@" not in elem:
                    # print(elem)
                    vastr = solve_latex_formula2(elem, varlist=["x"], const_dic={"\\pi": "3.14"})
                    if abs(vastr[0]["x"] - 1.57) < 1e-3:
                        findsig = "直角"
                    if abs(vastr[0]["x"] - 3.14) < 1e-3:
                        findsig = "平角"
            if findsig != 0:
                for elem in oneset:
                    if "@" in elem:
                        outjson.append([elem, "是", findsig])
                        tpoilist = latex_fenci(latex2space(elem.rstrip("}").lstrip("{角@")))
                        if "平角" == findsig:
                            strlist = []
                            for point in tpoilist:
                                strlist.append(self.language.name_symmetric(" ".join(point)).replace(" ", ""))
                            strlist = ["{点@" + point + "}" for point in strlist]
                            outjson.append([strlist, "是", "直线"])
                        else:
                            pass
        outjson = [{"因为": i1} for i1 in outjson]
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=outjson, delec=[])
        oldsetobj = space_ins._setobj
        # print(space_ins._setobj["直线集合"])
        # print(sys.getsizeof(space_ins._setobj))
        # 7. 遍历直角三角形 垂直已导出角可以略过，得到余角
        # print(outjson)
        # print("直角集合7")
        # print(len(oldsetobj["直角集合"]))
        rtlist = [elems.rstrip("}").lstrip("{三角形@") for elems in oldsetobj["直角三角形集合"]]
        rtlist = [latex_fenci(latex2space(angli)) for angli in rtlist]
        outjson = []
        for points in rtlist:
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
        outjson = [{"因为": i1} for i1 in outjson]
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=outjson, delec=[])
        # space_ins._setobj["全等三角形集合"] = list_set_deliver(space_ins._setobj["全等三角形集合"])
        space_ins._setobj = self.listset_deliverall(space_ins._setobj)
        # print(space_ins._setobj)
        for objset in basic_space_ins._setobj:
            if basic_space_ins._setobj[objset]["结构形式"] == "一级列表二级集合":
                space_ins._setobj[objset] = list_set_shrink(space_ins._setobj[objset])
        newsetobj = copy.deepcopy(space_ins._setobj)
        return newsetobj

    def conception2element(self, oldsetobj):
        "概念衍生，点， 点 生 线段 角，去掉顺序差异，再根据直线 衍生等值角"
        outjson = []
        for oneset in oldsetobj:
            if oneset == "正方形集合":
                for obj in oldsetobj[oneset]:
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
                        # print(outjson[-1])
                        tname = self.language.name_cyc_one(" ".join(tanglist[idangle:idangle + 3])).replace(" ", "")
                        tname = "{三角形@" + tname + "}"
                        outjson.append([tname, "是", "三角形"])
                        outjson.append([tname, "是", "直角三角形"])
            if oneset == "三角形集合":
                for obj in oldsetobj[oneset]:
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
            if oneset == "全等三角形集合":
                for objlist in oldsetobj[oneset]:
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
                            tpname2 = self.language.name_symmetric(" ".join(tseglist[idseg + 1:idseg + 2])).replace(" ",
                                                                                                                    "")
                            tpname2 = "{点@" + tpname2 + "}"
                            outjson.append([[tpname1, tpname2], "是", "直线"])
                        # 角
                        tanglist = tlist + tlist[0:2]
                        for idangle in range(3):
                            tname = self.language.name_symmetric(" ".join(tanglist[idangle:idangle + 3])).replace(" ",
                                                                                                                  "")
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
                                    if aea_sig[0] == 1 and (
                                                        aea_sig[1] + aea_sig[4] == 2 or aea_sig[2] + aea_sig[3] == 2):
                                        outjson.append([[eae_list[idmain][2], eae_list[idcli][2]], "是", "等值"])
                                        if aea_sig[1] == 1 and aea_sig[4] == 1:
                                            outjson.append([[eae_list[idmain][1], eae_list[idcli][1]], "是", "等值"])
                                            outjson.append([[eae_list[idmain][0], eae_list[idcli][0]], "是", "等值"])
                                        elif aea_sig[2] == 1 and aea_sig[3] == 1:
                                            outjson.append([[eae_list[idmain][1], eae_list[idcli][0]], "是", "等值"])
                                            outjson.append([[eae_list[idmain][0], eae_list[idcli][1]], "是", "等值"])
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
                                    if eae_sig[4] == 1 and (
                                                        eae_sig[0] + eae_sig[3] == 2 or eae_sig[1] + eae_sig[2] == 2):
                                        outjson.append([[eae_list[idmain][2], eae_list[idcli][2]], "是", "等值"])
                                        if eae_sig[0] == 1 and eae_sig[3] == 1:
                                            outjson.append([[aea_list[idmain][1], aea_list[idcli][1]], "是", "等值"])
                                            outjson.append([[aea_list[idmain][2], aea_list[idcli][2]], "是", "等值"])
                                        elif eae_sig[1] == 1 and eae_sig[2] == 1:
                                            outjson.append([[aea_list[idmain][1], aea_list[idcli][2]], "是", "等值"])
                                            outjson.append([[aea_list[idmain][2], aea_list[idcli][1]], "是", "等值"])
                                        outjson.append([[aea_list[idmain][0], aea_list[idcli][0]], "是", "等值"])
        outjson = [{"因为": i1} for i1 in outjson]
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj,
                                                         addc=outjson,
                                                         delec=[])
        return space_ins._setobj

    def element2conception(self, oldsetobj):
        "元素衍生概念"
        # 1. 得出 全等三角形
        # print(oldsetobj["三角形集合"])
        # print(oldsetobj["角集合"])
        # print(oldsetobj["等值集合"])
        triang_pointlist = [elems.rstrip("}").lstrip("{三角形@") for elems in oldsetobj["三角形集合"]]
        triang_pointlist = [latex_fenci(latex2space(angli)) for angli in triang_pointlist]
        # 1.1 边角边
        outjson = []
        eae_list = []
        aea_list = []
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
                            outjson.append([[aea_list[idmain][-1], aea_list[idcli][-1]], "是", "全等三角形"])
                            break
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
                            break
        outjson = [{"因为": i1} for i1 in outjson]
        field_name = "数学"
        scene_name = "解题"
        space_name = "basic"
        basic_space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_name = "customer"
        space_ins = self.gstack.readspace(space_name, scene_name, field_name)
        space_ins._setobj, _, _ = space_ins.tri2set_oper(basic_space_ins._setobj, space_ins._setobj,
                                                         space_ins._stopobj, addc=outjson, delec=[])
        space_ins._setobj = self.listset_deliverall(space_ins._setobj)
        return space_ins._setobj

    def judge_stop(self, oldsetobj, newsetobj, stopobj, basic_space_ins):
        "每步推理的具体操作 true为应该结束"
        # print(basic_space_ins._relation_trip)
        # print(basic_space_ins._proper_trip)
        # newsetobj["daian"] = 6
        # print("judge_stop")
        # print(oldsetobj)
        # print(newsetobj)
        if operator.eq(oldsetobj, newsetobj):
            message = "已知条件无法进一步推理"
            return message, True
            # print(operator.is_not(stopobj, newsetobj))
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
        message = "任务完成"
        return message, True

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
        # printstr3 = printstr3.replace("\\\\", "\\").replace("\\n", "\n")
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
    printstr3 = "已知：正方形 $ABCD, A、P、C $ 在一条直线上。$MN \\parallel BC, \\angle {BPQ} =90 ^{\\circ},A、M、B $ 在一条直线上，$\\angle {APM}$是锐角，$\\angle {NPQ} +\\angle {BPQ} +\\angle {BPM} =180^{\\circ }, \\angle {ACB}$是锐角。 $C、Q、 N、D $ 在一条直线上。求证 $PB = PQ$"
    handestr3 = "已知：正方形 $ABCD, A、P、C $ 在一条直线上。$PQ=PB, MN \\parallel BC, \\angle {BPQ} =90 ^{\\circ},A、B、M $ 在一条直线上。 $C、D、Q、 N $ 在一条直线上。求证 $PB = PQ$"
    # \\therefore AM=PM
    # \\because AB=MN
    # \\therefore MB=PN
    # \\angle BPQ=90 ^ {circ}
    # \\therefore \\angle BPM + \\angle NPQ = 90 ^ {circ}
    # \\because \\angle MBP + \\angle BPM = 90 ^ {circ}
    # \\therefore \\angle MBP = \\angle NPQ
    # \\triangle BPM 是直角三角形
    # \\triangle NPQ 是直角三角形
    # \\therefore \\triangle BPM \\cong \\triangle NPQ
    #
    # \\therefore PB = PQ
    # print(printstr3)
    # print(handestr3)
    solve_latex_prove(printstr3, handestr3)
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
