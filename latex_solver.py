from sympy import *
import re
import sympy as sy

operator_precedence = {
    '=': -1,
    '(': 0,
    ')': 0,
    '{': 0,
    '}': 0,
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
    '\\div': 2,
    '\\times': 2,
    '\\cdot': 2,
    '\\frac': 2,
    '\\sin': 4,
    '\\cos': 4,
    '\\tan': 4,
    '\\cot': 4,
    '^': 5,
    '\\sqrt': 5,
}
# 缩并同意符的operator_precedence = operlist + funclist + addtypelist
operlist = ["+", "-", "*", "/", "^", "\\sqrt"]
funclist = ["\\sin", "\\cos", "\\tan", "\\cot"]
addtypelist = ['\\frac']
pmlist = ["+", "-", "*", "/", "^", "(", "{", "=", "\\div", "\\cdot", "\\times", "\\pm", "\\mp"]
symblist = ["\\blacksquare", "\\triangle", "\\angle", "\\square", "\\bigodot", "\\diamondsuit", "\\Box"]
alpaist = ["\\gamma", "\\omega", "\\phi", "\\cos", "\\sin", "\\lambda", "\\zeta", "\\theta", "\\Omega", "\\pi", "\\rho",
           "\\tan"]


def is_numberj(s):
    if re.match("^[-+]?(([0-9]+)([.]([0-9]+))?|([.]([0-9]+))?)$", s):
        return float(s)


def postfix_convert(explist):
    '''
    将表达式list，转为后缀表达式list
    '''
    stack = []  # 运算符栈，存放运算符
    postfix = []  # 后缀表达式栈
    # print(explist)
    for chars in explist:
        # print(chars, stack, postfix)
        if chars not in operator_precedence:  # 非运算符号，直接进栈
            postfix.append(chars)
        else:
            if len(stack) == 0:  # 若是运算符栈啥也没有，直接将运算符进栈
                stack.append(chars)
            else:
                if chars == "(":
                    stack.append(chars)
                elif chars == ")":  # 遇到了右括号，运算符出栈到postfix中，并且将左括号出栈
                    while stack[-1] != "(":
                        postfix.append(stack.pop())
                    stack.pop()
                elif chars == "{":
                    stack.append(chars)
                elif chars == "}":  # 遇到了右括号，运算符出栈到postfix中，并且将左括号出栈
                    while stack[-1] != "{":
                        postfix.append(stack.pop())
                    stack.pop()
                elif chars == "\{":
                    stack.append(chars)
                elif chars == "\}":  # 遇到了右括号，运算符出栈到postfix中，并且将左括号出栈
                    while stack[-1] != "\{":
                        postfix.append(stack.pop())
                    stack.pop()
                elif chars == "[":
                    stack.append(chars)
                elif chars == "]":  # 遇到了右括号，运算符出栈到postfix中，并且将左括号出栈
                    while stack[-1] != "[":
                        postfix.append(stack.pop())
                    stack.pop()
                elif chars == "\\sqrt":
                    if stack[-1] == "^":
                        stack.pop()
                    stack.append(chars)
                elif operator_precedence[chars] >= operator_precedence[stack[-1]]:
                    # 只要优先级数字大，那么就继续追加
                    # print(operator_precedence[chars], operator_precedence[stack[-1]])
                    # print(chars)
                    stack.append(chars)
                else:
                    while len(stack) != 0:
                        # 运算符栈一直出栈，直到遇到了左括号或者长度为0
                        if stack[-1] == "(" or stack[-1] == "{" or stack[-1] == "\{" or stack[-1] == "[" or stack[
                            -1] == "=":
                            break
                        postfix.append(stack.pop())  # 将运算符栈的运算符，依次出栈放到表达式栈里面
                    stack.append(chars)  # 并且将当前符号追放到符号栈里面
    while len(stack) != 0:  # 如果符号站里面还有元素，就直接将其出栈到表达式栈里面
        postfix.append(stack.pop())
    # print(postfix)
    # print("out stack")
    # 3. 正则冗余表达
    postfix = normal_explist(postfix)
    # 4. 结合律
    tmlenth = len(postfix) - 1
    for i1 in range(tmlenth):
        if postfix[i1 + 1] == "-":
            if postfix[i1] == "-":
                postfix[i1] = "+"
            elif postfix[i1] == "+":
                postfix[i1] = "-"
        if postfix[i1 + 1] == "/":
            if postfix[i1] == "/":
                postfix[i1] = "*"
            elif postfix[i1] == "*":
                postfix[i1] = "/"
    return postfix


# ===========================这部分用于构造表达式树，不涉及计算================================#
class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def create_expression_tree(postfix):
    """
    利用后缀表达式，构造二叉树
    """
    stack = []
    # print postfix
    for char in postfix:
        if char not in operator_precedence:
            # 非操作符，即叶子节点
            node = Node(char)
            stack.append(node)
        else:
            # 遇到了运算符，出两个，进一个。
            node = Node(char)
            right = stack.pop()
            left = stack.pop()
            node.right = right
            node.left = left
            stack.append(node)
    # 将最后一个出了即可。
    return stack.pop()


def inorder(tree):
    if tree:
        inorder(tree.left)
        print(tree.val)
        inorder(tree.right)


# =============================这部分用于计算值===================================#
def calc_opt(num1, op, num2):
    # 1. 判断
    for i1 in num1 + num2:
        if i1 not in "0123456789.-":
            print(i1, type(i1))
            raise Exception("num error")
    try:
        if num1.isdigit():
            num1 = int(num1)
        else:
            num1 = float(num1)
        if num2.isdigit():
            num2 = int(num2)
        else:
            num2 = float(num2)
    except Exception as e:
        raise Exception("num error")
    # 2. 计算
    if op == "+":
        return num1 + num2
    elif op == "-":
        return num1 - num2
    elif op == "*":
        return num1 * num2
    elif op == "/":
        if num2 == 0:
            raise Exception("zeros error")
        else:
            return num1 / num2
    elif op == "^":
        return pow(num1, num2)
    elif op == "\\sqrt":
        return pow(num2, 1.0 / num1)
    else:
        raise Exception("op error")


def calc_func(num1, op):
    # 1. 判断
    for i1 in num1:
        if i1 not in "0123456789.":
            raise Exception("num error")
    try:
        if num1.isdigit():
            num1 = int(num1)
        else:
            num1 = float(num1)
    except Exception as e:
        raise Exception("num error")
    # 2. 计算
    if op == "\\sin":
        return math.sin(num1)
    elif op == "\\cos":
        return math.cos(num1)
    elif op == "\\tan":
        return math.tan(num1)
    elif op == "\\cot":
        return 1 / math.tan(num1)
    else:
        raise Exception("op error")


def equa_opt(num1, op, num2, vardic={}):
    # 1. 判断: 是否变量、是否是对象、是否是数字、是否是字符数组
    # print(num1)
    # print(type(num1))
    # print(num2)
    # print(type(num2))
    if num1 in vardic:
        num1 = vardic[num1]
    elif re.match("^<class 'sympy", str(type(num1))):
        pass
    elif isinstance(num1, int):
        pass
    elif isinstance(num1, float):
        pass
    elif num1.isdigit():
        num1 = int(num1)
    else:
        num1 = float(num1)
    if num2 in vardic:
        num2 = vardic[num2]
    elif re.match("^<class 'sympy", str(type(num2))):
        pass
    elif isinstance(num2, int):
        pass
    elif isinstance(num2, float):
        pass
    elif num2.isdigit():
        num2 = int(num2)
    else:
        num2 = float(num2)
    # 2. 计算
    if op == "+":
        return num1 + num2
    elif op == "-":
        return num1 - num2
    elif op == "*":
        return num1 * num2
    elif op == "/":
        if num2 == 0:
            raise Exception("zeros error")
        else:
            return num1 / num2
    elif op == "^":
        return num1 ** num2
    elif op == "\\sqrt":
        return num2 ** (1.0 / num1)
    else:
        raise Exception("op error")


def equa_equa_opt(num1, op, num2, vardic={}):
    # 1. 判断
    # print(num1)
    # print(type(num1))
    # print(num2)
    # print(type(num2))
    if num1 in vardic:
        num1 = vardic[num1]
    elif re.match("^<class 'sympy", str(type(num1))):
        pass
    elif isinstance(num1, int):
        pass
    elif isinstance(num1, float):
        pass
    elif num1.isdigit():
        num1 = int(num1)
    else:
        num1 = float(num1)
    if num2 in vardic:
        num2 = vardic[num2]
    elif re.match("^<class 'sympy", str(type(num2))):
        pass
    elif isinstance(num2, int):
        pass
    elif isinstance(num2, float):
        pass
    elif num2.isdigit():
        num2 = int(num2)
    else:
        num2 = float(num2)
    # 2. 计算
    return Eq(num1, num2)


def equa_func(num1, op, vardic={}):
    # 1. 判断
    # print(num1)
    # print(type(num1))
    if num1 in vardic:
        num1 = vardic[num1]
    elif re.match("^<class 'sympy", str(type(num1))):
        pass
    elif isinstance(num1, int):
        pass
    elif isinstance(num1, float):
        pass
    elif num1.isdigit():
        num1 = int(num1)
    else:
        num1 = float(num1)
    # 2. 计算
    if op == "\\sin":
        return sin(num1)
    elif op == "\\cos":
        return cos(num1)
    elif op == "\\tan":
        return tan(num1)
    elif op == "\\cot":
        return cot(num1)
    else:
        raise Exception("op error")


def cal_expression_tree(postfix):
    # 1. 循环计算
    stack = []
    for chars in postfix:
        stack.append(chars)
        if chars in operlist:
            # 双数值计算
            op = stack.pop()
            num2 = stack.pop()
            num1 = stack.pop()
            # print(num1, op, num2)
            value = calc_opt(num1, op, num2)
            value = str(value)
            stack.append(value)
            # print(value)
        elif chars in funclist:
            # 单数值计算
            op = stack.pop()
            num1 = stack.pop()
            # print(num1, op)
            value = calc_func(num1, op)
            value = str(value)
            stack.append(value)
            # print(value)
    if stack[0].isdigit():
        return int(stack[0])
    else:
        return float(stack[0])


def cal_equation_tree(postfix, vardic={}):
    # 1. 循环计算
    stack = []
    for chars in postfix:
        # print(chars, stack)
        stack.append(chars)
        if chars in operlist:
            # 双数值计算
            op = stack.pop()
            num2 = stack.pop()
            num1 = stack.pop()
            # print(num1, op, num2)
            value = equa_opt(num1, op, num2, vardic=vardic)
            stack.append(value)
            # print(value)
        elif chars in funclist:
            # 单数值计算
            op = stack.pop()
            num1 = stack.pop()
            # print(num1, op)
            value = equa_func(num1, op, vardic=vardic)
            stack.append(value)
            # print(value)
        elif chars == "=":
            # 双数值计算
            op = stack.pop()
            num2 = stack.pop()
            num1 = stack.pop()
            # print(num1, op, num2)
            value = equa_equa_opt(num1, op, num2, vardic=vardic)
            stack.append(value)
    return stack[0]


def latex2list(instr, varlist=[]):
    # varlist = ["acd", "bc", "cdef"]
    # varlist需要通过题干解析 1. 字母组连接 2. 数字缩并 3. 字母间算符 4. 单位换算 5. 分数 6. pm mp 符号补全
    # 1. 带空格的latex， 数字缩并，
    instr = instr.strip()
    instr = " " + instr + " "
    # 1. 字母组数连接 ×
    vardic = {i1: len(i1) for i1 in varlist}
    varitem = [i1[0] for i1 in sorted(vardic.items(), key=lambda s: s[1], reverse=True)]
    for i1 in varitem:
        instr = instr.replace(" " + " ".join([i2 for i2 in i1]) + " ", " " + i1 + " ")
    instr = instr.strip()
    inlist = instr.split(" ")
    orilenth = len(inlist)
    # 2. 处理 单组数字 含义
    for i1 in range(orilenth - 1, -1, -1):
        if i1 > 0 and inlist[i1][0] in "0123456789." and inlist[i1 - 1][0] in "0123456789.":
            inlist[i1 - 1] = inlist[i1 - 1] + inlist[i1]
            del inlist[i1]
        # 处理 函数 简写
        elif i1 > 0 and inlist[i1] in funclist and inlist[i1 - 1][0] in "0123456789.(){}":
            singl_cout = 0
            numcout = 0
            endnum = 0
            if inlist[i1 + 1] != "{":
                raise Exception("match sig error.")
            for i2 in range(i1 + 1, orilenth):
                if inlist[i2] == "{":
                    singl_cout += 1
                elif inlist[i2] == "}":
                    singl_cout -= 1
                if singl_cout == 0:
                    numcout += 1
                    if numcout == 1:
                        endnum = i2 + 1
                        break
            if numcout != 1:
                raise Exception("match num error.")
            # 后插入
            inlist.insert(endnum, ")")
            # 前插入
            inlist.insert(i1, "(")
            inlist.insert(i1, "\\times")
    # 3. 字母间算符
    orilenth = len(inlist)
    for i1 in range(orilenth - 1, -1, -1):
        # print(inlist[i1 - 1])
        if i1 > 0 and (inlist[i1][0].isalpha() or inlist[i1] in symblist + alpaist) \
                and (inlist[i1 - 1][0].isalpha() or inlist[i1 - 1] in alpaist):
            inlist.insert(i1, "\\cdot")
        elif i1 > 0 and (inlist[i1][0].isalpha() or inlist[i1] in symblist + alpaist) \
                and (inlist[i1 - 1] not in ["+", "-", "\\pm", "\\mp"] and is_numberj(inlist[i1 - 1])):
            inlist.insert(i1, "\\cdot")
    # 4. 单位换算 角度
    orilenth = len(inlist)
    for i1 in range(orilenth - 1, -1, -1):
        # 处理 单组数字 含义
        if i1 > 0 and (inlist[i1][0] in ["'", "\""] or " ".join(inlist[i1 - 2:i1 + 2]) == "^ { \\circ }"):
            if inlist[i1][0] in "'":
                del inlist[i1]
                inlist.insert(i1, "\\pi")
                inlist.insert(i1, "\\times")
                inlist.insert(i1, "180")
                inlist.insert(i1, "/")
                inlist.insert(i1, "60")
                inlist.insert(i1, "/")
            elif inlist[i1][0] in "\"":
                del inlist[i1]
                inlist.insert(i1, "\\pi")
                inlist.insert(i1, "\\times")
                inlist.insert(i1, "180")
                inlist.insert(i1, "/")
                inlist.insert(i1, "60")
                inlist.insert(i1, "/")
                inlist.insert(i1, "60")
                inlist.insert(i1, "/")
            elif " ".join(inlist[i1 - 2:i1 + 2]) == "^ { \\circ }":
                del inlist[i1 - 2]
                del inlist[i1 - 2]
                del inlist[i1 - 2]
                del inlist[i1 - 2]
                inlist.insert(i1 - 2, "\\pi")
                inlist.insert(i1 - 2, "\\times")
                inlist.insert(i1 - 2, "180")
                inlist.insert(i1 - 2, "/")
    # 5. 处理 分数
    orilenth = len(inlist)
    for i1 in range(orilenth - 1, -1, -1):
        # 遍历 分数
        if i1 > 0 and inlist[i1] in addtypelist and is_numberj(inlist[i1 - 1]):
            singl_cout = 0
            numcout = 0
            endnum = 0
            if inlist[i1 + 1] != "{":
                raise Exception("match sig error.")
            for i2 in range(i1 + 1, orilenth):
                if inlist[i2] == "{":
                    singl_cout += 1
                elif inlist[i2] == "}":
                    singl_cout -= 1
                if singl_cout == 0:
                    numcout += 1
                    if numcout == 2:
                        endnum = i2 + 1
                        break
            if numcout != 2:
                raise Exception("match num error.")
            # 后插入
            inlist.insert(endnum, ")")
            # 前插入
            inlist.insert(i1, "+")
            inlist.insert(i1 - 1, "(")
    # 6. 根下处理
    orilenth = len(inlist)
    for i1 in range(orilenth - 1, -1, -1):
        if inlist[i1] == "\\sqrt" and inlist[i1 + 1] == "{":
            # 后插入
            singl_cout = 0
            numcout = 0
            endnum = i1
            for i2 in range(i1 + 1, orilenth):
                if inlist[i2] == "{":
                    singl_cout += 1
                elif inlist[i2] == "}":
                    singl_cout -= 1
                if singl_cout == 0:
                    numcout += 1
                    if numcout == 1:
                        endnum = i2 + 1
                        break
            if numcout != 1:
                raise Exception("match end num error.")
            # 前插入
            singl_cout = 0
            numcout = 0
            startnum = i1
            for i2 in range(i1 - 1, 0, -1):
                if inlist[i2] == "}":
                    singl_cout += 1
                elif inlist[i2] == "{":
                    singl_cout -= 1
                if singl_cout == 0:
                    if inlist[i2 - 1] == "^":
                        numcout += 1
                        if numcout == 1:
                            startnum = i2 - 1
                            break
                    else:
                        break
            inlist.insert(endnum, ")")
            # 不匹配从零插添加二次项，匹配从算好的插
            if numcout == 1:
                inlist.insert(startnum, "(")
            else:
                inlist.insert(startnum, "}")
                inlist.insert(startnum, "2")
                inlist.insert(startnum, "{")
                inlist.insert(startnum, "^")
                inlist.insert(startnum, "(")
    # 7. +- 缩并
    orilenth = len(inlist)
    for i1 in range(orilenth - 1, -1, -1):
        if i1 > 0 and inlist[i1] in ["+", "-", "\\pm", "\\mp"] and inlist[i1 - 1] in pmlist:
            if inlist[i1 + 1] in funclist + ["(", "{", "\\frac", "\\sqrt"]:
                # 后面的意义单元为 函数 或 平衡符号组
                if inlist[i1 + 1] == "\\frac":
                    stop_num = 2
                else:
                    stop_num = 1
                typestartsig = ""
                typeendsig = ""
                singl_cout = 0
                numcout = 0
                endnum = 0
                for i2 in range(i1 + 1, orilenth):
                    if typestartsig == "":
                        if inlist[i2] == "{":
                            typestartsig = "{"
                            typeendsig = "}"
                            singl_cout += 1
                        elif inlist[i2] == "(":
                            typestartsig = "("
                            typeendsig = ")"
                            singl_cout += 1
                    else:
                        if inlist[i2] == typestartsig:
                            singl_cout += 1
                        elif inlist[i2] == typeendsig:
                            singl_cout -= 1
                        if singl_cout == 0:
                            numcout += 1
                            if numcout == stop_num:
                                endnum = i2 + 1
                                break
                if numcout != stop_num:
                    raise Exception("match num error.")
                # 后插入
                inlist.insert(endnum, ")")
                # 前插入
                inlist.insert(i1, "0")
                inlist.insert(i1, "(")
            else:
                # i1+1 为数字 插入数字后为 i1+2
                # 后插入
                inlist.insert(i1 + 2, ")")
                # 前插入
                inlist.insert(i1, "0")
                inlist.insert(i1, "(")
            # elif inlist[i1 + 1] in funclist + ["(", "{"]:
            #     if inlist[i1] == "+" and inlist[i1 - 1] in pmlist:
            #         del inlist[i1]
            #     elif inlist[i1] == "-" and inlist[i1 - 1] in pmlist:
            #         if inlist[i1 - 1] == "-":
            #             del inlist[i1]
            #             del inlist[i1 - 1]
            #             inlist.insert(i1 - 1, "+")
            #         elif inlist[i1 - 1] == "+":
            #             del inlist[i1 - 1]
            #         elif inlist[i1 - 1] == "\\pm":
            #             inlist[i1 - 1] = "\\mp"
            #             del inlist[i1]
            #         elif inlist[i1 - 1] == "\\mp":
            #             inlist[i1 - 1] = "\\pm"
            #             del inlist[i1]
            #         elif inlist[i1 - 1] in ["*", "/", "^", "\\div", "\\cdot", "\\times"] and inlist[
            #                     i1 + 1] in funclist + [
            #             "(", "{"]:
            #             # 后面的意义单元为 函数 或 平衡符号组
            #             typestartsig = ""
            #             typeendsig = ""
            #             singl_cout = 0
            #             numcout = 0
            #             endnum = 0
            #             for i2 in range(i1 + 1, orilenth):
            #                 if typestartsig == "":
            #                     if inlist[i2] == "{":
            #                         typestartsig = "{"
            #                         typeendsig = "}"
            #                         singl_cout += 1
            #                     elif inlist[i2] == "(":
            #                         typestartsig = "("
            #                         typeendsig = ")"
            #                         singl_cout += 1
            #                 else:
            #                     if inlist[i2] == typestartsig:
            #                         singl_cout += 1
            #                     elif inlist[i2] == typeendsig:
            #                         singl_cout -= 1
            #                     if singl_cout == 0:
            #                         numcout += 1
            #                         if numcout == 1:
            #                             endnum = i2 + 1
            #                             break
            #             if numcout != 1:
            #                 raise Exception("match num error.")
            #             # 后插入
            #             inlist.insert(endnum, ")")
            #             # 前插入
            #             inlist.insert(i1, "0")
            #             inlist.insert(i1, "(")
            #         elif inlist[i1 - 1] in "({":
            #             inlist.insert(i1, "0")
            #         elif inlist[i1 - 1] in ["*", "/", "^", "\\div", "\\cdot", "\\times"]:
            #             # 后面的意义单元为数字
            #             inlist.insert(i1 + 1, ")")
            #             inlist.insert(i1, "0")
            #             inlist.insert(i1, "(")
            pass
    return inlist


def var2num(inlist):
    # 变量替换
    outlist = []
    return outlist


def equsolver(instr):
    # 变量替换
    outlist = []
    return outlist


def normal_explist(inlist):
    # 1. 同化运算符
    for id1, i1 in enumerate(inlist):
        if i1 == "\\times" or i1 == "\\cdot":
            inlist[id1] = "*"
        elif i1 == "\\frac" or i1 == "\\div":
            inlist[id1] = "/"
    return inlist


def constance_explist(inlist, const_dic={"\\pi": "3.14"}):
    # 1. 常数替换
    for id1, i1 in enumerate(inlist):
        if i1 in const_dic:
            inlist[id1] = const_dic[i1]
    return inlist


def equat2array(expstr):
    # 1. 抽取equation
    expstr = expstr.replace("  ", " ")
    expstr = expstr.replace("\\left \\{ \\begin{aligned}", "")
    expstr = expstr.replace("\\end{aligned} \\right.", "")
    equalist = expstr.split("\\\\")
    equalist = [i1.strip() for i1 in equalist]
    return equalist
    # # 2. equation = 向左移项，忽略右边
    # outequalist = []
    # for i1 in equalist:
    #     tmplist = i1.split("=")
    #     outequalist.append(tmplist[0].strip() + " - ( " + tmplist[1].strip() + " )")
    # return outequalist


# 1. latex 公式解析
def solve_latex_formula(expstr, varlist=[], const_dic={"\\pi": "3.14"}):
    # 1. latex单元到列表
    explist = latex2list(expstr, varlist=varlist)
    # print(" ".join(explist))
    # 2. 解析堆栈形式
    postfix = postfix_convert(explist)
    # print(" ".join(postfix))
    # 4. 常数替换
    postfix = constance_explist(postfix, const_dic=const_dic)
    # print(" ".join(postfix))
    # tree = create_expression_tree(postfix)
    # inorder(tree)
    # 5. 求解器
    value = cal_expression_tree(postfix)
    return value


def solve_latex_formula2(expstr, varlist=["x"], const_dic={"\\pi": "3.14"}):
    # 1. latex单元到列表
    expstr = "x = " + expstr
    vardic = {i1: Symbol(i1) for i1 in varlist}
    # 1. 分成单行
    equalist = equat2array(expstr)
    # print(equalist)
    # 2. latex单元到列表
    tmplist = latex2list(equalist[0], varlist=varlist)
    # # 3. 解析堆栈形式
    postfix = postfix_convert(tmplist)
    # print(" ".join(postfix))
    # # 4. 常数替换
    postfix = constance_explist(postfix, const_dic=const_dic)
    exprestion = cal_equation_tree(postfix, vardic)
    # print(exprestion.simplify())
    # print(exprestion.expand())
    results = [{list(vardic.keys())[id2]: i2} for i1 in sy.nonlinsolve([exprestion], list(vardic.values())) for id2, i2
               in enumerate(i1)]
    return results


# 2. latex 方程解析
def solve_latex_equation(expstr, varlist=["x", "y"], const_dic={"\\pi": "3.14"}):
    vardic = {i1: Symbol(i1) for i1 in varlist}
    # 1. 分成单行
    equalist = equat2array(expstr)
    # 2. latex单元到列表
    equaoutlist = []
    for i1 in equalist:
        tmplist = latex2list(i1, varlist=varlist)
        # # 3. 解析堆栈形式
        postfix = postfix_convert(tmplist)
        # # 4. 常数替换
        postfix = constance_explist(postfix, const_dic=const_dic)
        exprestion = cal_equation_tree(postfix, vardic)
        equaoutlist.append(exprestion)
        # print(equaoutlist[-1].simplify())
        # print(equaoutlist[-1].expand())
    # eq = [x ** 2 + y ** 3 - 2, x ** 3 + y ** 3]
    # results = sy.nonlinsolve(equaoutlist, list(vardic.values()))
    results = [{list(vardic.keys())[id2]: i2} for i1 in sy.nonlinsolve(equaoutlist, list(vardic.values())) for id2, i2
               in enumerate(i1)]
    return results


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
    # expstr = "1 + 2 * ( 3 - 1 + ( 1 - 2 ) ) - 6"
    # expstr = '( 2 + 3 ) * ( 5 + 7 ) + 9 / 3 . 3 - ( ( 8 / 4 ) - 6 )'
    # expstr = "0 . 3 a + 4 . 2 = 6 ; 0 . 3 a + 4 . 2 - 4 . 2 = 6 - 4 . 2 ; 0 . 3 a = 1 . 8 ; 0 . 3 a \\div 0 . 3 = 1 . 8 \\div 0 . 3 ; a = 6"
    # expstr = "( \\sqrt { 1 0 } ) ^ { 2 0 1 5 } \\cdot \\frac { 1 } { ( \\sqrt { - 1 . 2 } ) ^ { 2 0 1 3 } }"
    # expstr = "2 . 3 - 5 * 4 \\frac { 3 \\times ( 1 . 2 - - 2 ) } { - 2 0 1 5 } + ( + 3 - 6 ) \\sqrt { 1 5 + + 1 } ^ { 2 }"
    # expstr = "2 . 3 \\times - \\sin { 4 5 ^ { \\circ } } - 5 * 4 \\frac { 3 \\times ( 1 . 2 - - 2 ) } { - 2 0 1 5 } + ( + 3 - 6 ) \\sqrt { 1 5 + + 1 } ^ { 2 }"
    # expstr = "\\sin { 4 5 ^ { \\circ } } * 2"
    # expstr = "\\sin { \\pi \\div 2 } * 2"
    expstr = "^ { 4 - 1 } \\sqrt { 3 * 9 } * \\sin { \\pi \\div 2 } * 2"
    # expstr = "4 ^ { \\sqrt { 3 - 2 } } * \\sin { \\pi \\div 2 } * 2"
    # expstr = "\\sqrt { 3 - - 1 } * \\sin { \\pi \\div 2 } * 2"
    # expstr = "1 - 2 - 3 - 4"
    # print(expstr)
    # # 1. latex 公式解析
    # value = solve_latex_formula(expstr)
    # print(value)
    # value = solve_latex_formula2(expstr)
    # print(value)
    # expstr2 = "\\left \\{ \\begin{aligned} 2 x + 1 . 1 y = 6 - 2 \\\\ x - y = 3 - 1 \\end{aligned} \\right."
    expstr2 = "\\left \\{ \\begin{aligned} x + y = 6 - 2 \\\\ x - y = 3 - 1 \\end{aligned} \\right."
    # print(expstr2)
    # 2. latex 方程解析
    results = solve_latex_equation(expstr2)
    print(results)
