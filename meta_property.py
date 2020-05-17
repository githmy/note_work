properobj = {
    "同义代指": {
        "传递": "_函数"
    },
    "平行集合": {
        "性质": "关系",
        "函数": ["二级传递"],
        "内容属性": "平行",
        "结构形式": "一级列表二级集合"
    },
    "垂直集合": {
        "性质": "关系",
        "函数": [],
        "内容属性": "垂直",
        "结构形式": "一级列表二级集合"
    },
    "等值集合": {
        "性质": "关系",
        "函数": ["二级传递"],
        "内容属性": "等值",
        "结构形式": "一级列表二级集合"
    },
    "等价集合": {
        "性质": "关系",
        "函数": ["二级传递"],
        # 如{等角三角形, 等边三角形, 正三角形}
        "内容属性": "等价",
        "结构形式": "一级列表二级集合"
    },
    "线段集合": {
        "性质": "实体",
        "函数": [],
        "内容属性": "线段",
        # 线段不能代表 多边形的边，之后升级为边才行。
        "结构形式": "一级集合"
    },
    "边集合": {
        "性质": "实体",
        "函数": [],
        "内容属性": "线段",
        "结构形式": "一级集合"
    },
    "直线集合": {
        "性质": "实体",
        "函数": [],
        "内容属性": "直线",
        # 线段 代表 直线
        "结构形式": "一级列表二级集合"
    },
    "点集合": {
        "性质": "实体",
        "函数": [],
        "内容属性": "点",
        "结构形式": "一级集合"
    },
    "表达式": {
        "性质": "实体",
        "函数": ["方程解析函数"],
        "内容属性": "方程",
        "结构形式": "一级集合"
    },
    "n边形": {
        "边数": "n",
        "点数": "n",
        "内角和": "n * 1 8 0 - 3 6 0",
        "面积": None,
        "周长": None,
    },
    # 如果属性的 有抽象类，抽象类的属性有 集合，则该属性的的值为 集合的长度
    "三角形": {
        "边": {"数量": 3},
        "点": {"数量": 3},
        "底": "x",
        "高": "y",
        "面积": "x * y / 2",
    },
    "全等三角形": {
        "函数充分导出": ["三边相等", "三边相等"],
        "函数必要导入": [["三角形", "两边等的夹角等"], ["三角形", "两角等", "任一对应边等"]]
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
    ["边", "数量", "3"],
    ["边", "包含", ""],
]

equal_sets = [{"等角三角形", "等边三角形", "正三角形"}]

# 是:
# 附属:
# 属于:

# abcid     ["abc", "是", "三角形"],
# abid      ["ab", "是", "线段"],
# # 递归 id
# 关系id1   ["_abid", "附属", "_abcid"],
# 关系id2   [ ["_acid","_adid","_abid","_aeid"], "属于", "平行集合" ],
# 关系id3   ["xxxx", "是", "表达式"],
