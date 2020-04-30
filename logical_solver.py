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
import logging
import logging.handlers
from utils.path_tool import makesurepath
import jsonpatch

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
            self._properobj, self._triobj = self.storage_oper("r")
        else:
            self._properobj = {}
            self._triobj = []

    def storage_oper(self, operstr):
        """硬件：存储交互操作"""
        if operstr == "r":
            properobj = {
                "n边形": {"边数": "x",
                        "点数": "x",
                        "内角和": "x * 1 8 0 - 3 6 0",
                        "面积": None,
                        "周长": None,
                        },
                "三角形": {"底": "x",
                        "高": "y",
                        "面积": "x * y / 2",
                        },
                "圆": {"半径": "x",
                      "直径": "2 x",
                      "面积": "\\pi * r * r",
                      "周长": "\\pi * r * 2",
                      },
                "矩形": {"长": "x",
                       "宽": "y",
                       "面积": "x * y",
                       "周长": "2 x + 2 y",
                       "垂直": [["长", "宽"]],
                       "对角线": "相等",
                       },
                "正方形": {"长": "x",
                        "宽": "x",
                        "面积": "x * x",
                        "周长": "4 x",
                        "垂直": [["对角线"]],
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
            return properobj, triobj
        elif operstr == "w":
            return True

    def triple_oper(self, add={"properobj": {}, "triobj": []}, dele={"properobj": [], "triobj": []}):
        """内存：triple交互操作"""
        # print(self._properobj)
        # print(self._triobj)
        for oneproper in add["properobj"]:
            self._properobj[oneproper] = add["properobj"][oneproper]
        for onetri in add["triobj"]:
            havesig = 0
            for oritri in self._triobj:
                patch = jsonpatch.JsonPatch.from_diff(onetri, oritri)
                if list(patch) == []:
                    havesig = 1
                    break
            if havesig == 0:
                self._triobj.append(onetri)
        # print(self._properobj)
        # print(self._triobj)
        for oneproper in dele["properobj"]:
            try:
                del self._properobj[oneproper]
            except Exception as e:
                logger1.info("delete %s error %s" % (oneproper, e))
        for onetri in dele["triobj"]:
            lenth = len(self._triobj)
            for id1 in range(lenth - 1, -1, -1):
                patch = jsonpatch.JsonPatch.from_diff(onetri, self._triobj[id1])
                if list(patch) == []:
                    del self._triobj[id1]
                    break
                    # print(self._properobj)
                    # print(self._triobj)

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
                for trip in self._triobj:
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
                for trip in self._triobj:
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
            if item in self._properobj and property in self._properobj[item] and self._properobj[item][
                property] is not None:
                properlist.append(self._properobj[item][property])
        return properlist

    def find_property_value_child(self, obj="四边形", property="对角线", value="相等"):
        """内存：给定子级属性值 返回 子级list"""
        # 1. 找到所有的 子级
        # print("get_child")
        childlist = self.get_child(obj)
        # 2. 查找 符合 属性值的主体
        objlist = []
        for item in childlist:
            if item in self._properobj and property in self._properobj[item] and value == self._properobj[item][
                property]:
                objlist.append(item)
        return objlist


class VirtualSpace(BasicalSpace):
    def __init__(self, origin_str):
        self._properobj = super(VirtualSpace, self)._properobj
        self._triobj = super(VirtualSpace, self)._triobj


class NLPtool(object):
    def __init__(self):
        pass

    def fenci2triple(self, fenci_list):
        """ 解析：词性逻辑 到 内存格式"""
        print(fenci_list)
        # write_json = {"add": {"properobj": {"test": {"red": 1, "blue": 2, "green": 3}},
        #                       "triobj": [['三角形n', '属于', 'n边形'], ['三角形', '属于', 'n边形']]},
        #               "dele": {"properobj": ["n边形"], "triobj": [['三角形', '属于', 'n边形']]}}
        # for item in fenci_list:
        #     if item["text"][1]=="n":
        #         pass
        # write_json = {}
        write_json = {
            "add": {"properobj": {"test": {"red": 1, "blue": 2, "green": 3}},
                    "triobj": [['三角形n', '属于', 'n边形'], ['三角形', '属于', 'n边形']]},
            "dele": {"properobj": {}, "triobj": []}
        }
        return write_json

    def nature2space(self, instr_list, gstack):
        """ 解析：自然语言 到 空间三元组"""
        # 1. 判断场景领域, 临时略过固定数学。
        field_name = "数学"
        scene_name = "解题"
        # field_ins = Field(fieldname)
        # scene_ins = Scene(scenename)
        # 2. 具体场景领域的录入
        space_name = "basic"
        if not gstack.is_inspace_list(space_name, scene_name, field_name):
            space_ins = BasicalSpace(space_name=space_name, field_name=field_name, scene_name=scene_name)
            gstack.loadspace(space_name, scene_name, field_name, space_ins)
        space_name = "customer"
        if not gstack.is_inspace_list(space_name, scene_name, field_name):
            space_ins = BasicalSpace(space_name=space_name, field_name=field_name, scene_name=scene_name)
            gstack.loadspace(space_name, scene_name, field_name, space_ins)
        # 3. 语言录入初始化, 添加语句
        fenci_list = []
        for strs in instr_list:
            # 4. 分词实体
            if "text" in strs:
                for word, gflag in pseg.cut(strs["text"]):
                    fixword = word.strip(" ,，.。:：\t")
                    # print(word, fixword, gflag)
                    if fixword != "":
                        fenci_list.append({"text": [fixword, gflag]})
            else:
                fenci_list.append(strs)
        # 5. 实体2空间
        logger1.info("language clean: %s" % fenci_list)
        gstack.lang_list.append(fenci_list)
        write_json = self.fenci2triple(fenci_list)
        field_name = "数学"
        scene_name = "解题"
        space_name = "customer"
        # space_name = "basic"
        space_ins = gstack.readspace(space_name, scene_name, field_name)
        # 写入空间
        space_ins.triple_oper(add=write_json["add"], dele=write_json["dele"])
        print(space_ins._properobj)
        print(space_ins._triobj)
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

    def __call__(self, *args, **kwargs):
        """输入为语言的 list dic 数组: text latex"""
        analist = args[0]
        logger1.info("initial analyzing: %s" % analist)
        # 1. 循环处理每句话 生成空间解析 和 步骤list
        for sentence in analist:
            logger1.info("language: %s" % sentence)
            self.analyize_strs(sentence)
        # 2. 内存推理，基于之前的步骤条件
        print(self.gstack.space_list)
        self.inference(triplelist)
        print("end")
        exit()

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
        # 分句
        sentenc_list = re.split('。;|\n', printstr3)
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
        {"题号":"3","类型":"公式","已知": ["\\sin { 4 5 ^ { \\circ } } * 2"],"varlist":["ab","ABCD"],"求解":[], "参考步骤":[{"表达式":"\\sqrt 2","分值":"0.5"}]},
        {"题号":"4","类型":"方程","已知": ["a \\sin { 4 5 ^ { \\circ } } * 2 = 6"],"varlist":["ab","ABCD"],"求解":["a"], "参考步骤":[{},{}]},
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
    # 3. latex 证明
    printstr3 = "已知：四边形 $ABCD$ 中 ， $AD\\parallel BC$，$AC=BD$ ，\n 求证 ：$AB=DC$"
    handestr3 = "已知：四边形 $ABCD$ 中 ， $AD\\parallel BC$，$AC=BD$ ，\n 求证 ：$AB=DC$"
    # print(printstr3)
    # print(handestr3)
    solve_latex_prove(printstr3, handestr3)
    print("end")
    exit()
