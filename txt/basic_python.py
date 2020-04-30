import jieba
import pandas as pd
import jsonpatch
# 警告过滤
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# !pip install goto-statement
from goto import with_goto


def garbage_collector():
    # 垃圾回收
    import gc
    gc.collect()


def color():
    print("\033[1;30m 字体颜色：白色\033[0m")
    print("\033[1;31m 字体颜色：红色\033[0m")
    print("\033[1;32m 字体颜色：深黄色\033[0m")
    print("\033[1;33m 字体颜色：浅黄色\033[0m")
    print("\033[1;34m 字体颜色：蓝色\033[0m")
    print("\033[1;35m 字体颜色：淡紫色\033[0m")
    print("\033[1;36m 字体颜色：青色\033[0m")
    print("\033[1;37m 字体颜色：灰色\033[0m")
    print("\033[1;38m 字体颜色：浅灰色\033[0m")

    print("背景颜色：白色   \033[1;40m    \033[0m")
    print("背景颜色：红色   \033[1;41m    \033[0m")
    print("背景颜色：深黄色 \033[1;42m    \033[0m")
    print("背景颜色：浅黄色 \033[1;43m    \033[0m")
    print("背景颜色：蓝色   \033[1;44m    \033[0m")
    print("背景颜色：淡紫色 \033[1;45m    \033[0m")
    print("背景颜色：青色   \033[1;46m    \033[0m")
    print("背景颜色：灰色   \033[1;47m    \033[0m")

    print('\033[1;35;0m字体变色，但无背景色 \033[0m')  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')
    print('\033[1;45m 字体不变色，有背景色 \033[0m')  # 有高亮
    print('\033[1;35;46m 字体有色，且有背景色 \033[0m')  # 有高亮
    print('\033[0;35;46m 字体有色，且有背景色 \033[0m')  # 无高亮
    print('不换行', end='')
    print('换行')


def propertys():
    变量名称 = 3
    print(变量名称)
    print(0b100)
    print(0o77)
    print(0x123)
    print(True)
    print(False == 0)
    print(True == 1)
    print(True + 3)
    print(False + 3)
    print(bool({}))
    print(bool("False"))
    print(bool(False))
    print(bool(2))
    print(bool(1))
    print(bool(0))
    print(bool(None))
    # 复数
    ComplexNumber1 = complex(3, 6)
    print(ComplexNumber1)
    # print(3 + 6i)
    print(3 + 6j)
    print(4 + 7J)
    a = 32.5
    b = 32.5
    print(bool(a))
    print(a and 34)
    print(a or 34)
    print(not a)
    print(id(a), id(b))
    print(id(True), id(1))
    print(divmod(6, 2))  # 整 除 与 取 模
    print(6 // 2, 6 % 2)
    print(all([3 > 2, 6 < 9]))
    print(all([2, 3, 5, 6, 7]))  # 如果所有为空、0、false，则返回true
    print(any([2, 3, 5, 6, 7]))  # 如果不都为空、0、false，则返回true


@with_goto
def goto_demo():
    i = 1
    result = []
    label.begin
    if i == 2:
        goto.end

    result.append(i)
    i += 1
    goto.begin
    label.end


def list_deal():
    listA = [1, 2, 3, 4, 5]
    listB = [3, 4, 5, 6, 7]
    # 逆序列
    listB.reverse()
    # # 元素的序号
    your_list.index('your_item')
    your_list.sort(cmp=None, key=None, reverse=False)
    # 删除元素
    # your_list.remove('your_item')
    # your_list.pop(-1)
    # del (your_list[0])
    # 交集
    retA = [i for i in listA if i in listB]
    retB = list(set(listA).intersection(set(listB)))
    print("retA is: ", retA)
    print("retB is: ", retB)
    # 求并集
    retC = list(set(listA).union(set(listB)))
    print("retC1 is: ", retC)

    # 求差集，在B中但不在A中
    retD = list(set(listB).difference(set(listA)))
    print("retD is: ", retD)
    retE = [i for i in listB if i not in listA]
    print("retE is: ", retE)

    dict_data = {6: 109, 10: 105, 3: 211, 8: 102, 7: 106}
    # 比较数组是否相同 对比
    import operator
    operator.ne(set(old_id["mainReviewPoints"]), set(new_id["mainReviewPoints"]))


def list_transpose():
    def transpose(matrix):
        return zip(*matrix)

    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
    print(transpose(matrix))


def json_sort():
    # json 字典排序
    dict_data = {6: 109, 10: 105, 3: 211, 8: 102, 7: 106}

    # 对字典按键（key）进行排序（默认由小到大）
    test_data_0 = sorted(dict_data.keys())
    # 输出结果
    print(test_data_0)  # [3, 6, 7, 8, 10]
    test_data_1 = sorted(dict_data.items(), key=lambda x: x[0])
    # 输出结果
    print(test_data_1)  # [(3, 211), (6, 109), (7, 106), (8, 102), (10, 105)]

    # 对字典按值（value）进行排序（默认由小到大）
    test_data_2 = sorted(dict_data.items(), key=lambda x: x[1])
    # 输出结果
    print(test_data_2)  # [(8, 102), (10, 105), (7, 106), (6, 109), (3, 211)]
    test_data_3 = sorted(dict_data.items(), key=lambda x: x[1], reverse=True)
    # 输出结果
    print(test_data_3)  # [(3, 211), (6, 109), (7, 106), (10, 105), (8, 102)]


def cut_one_file(filename):
    print("cuting:" + filename)
    with open("bcut." + filename + ".bcut", 'rt', encoding="utf8") as f:
        result = f.readlines()
        with open(filename, 'wt', encoding="utf8") as f2:
            for i in result:
                f2.write(" ".join(jieba.cut(i)))


def cut_files():
    cut_one_file("dev.zh")
    cut_one_file("train.zh")
    cut_one_file("tst.zh")


def static_top_n(n):
    arrayobjs = {}

    def countrer(filename):
        with open(filename, 'rt', encoding="utf8") as f:
            results = f.readlines()
            for result in results:
                arry1 = result.split(" ")
                for ele1 in arry1:
                    if ele1 not in arrayobjs:
                        arrayobjs.__setitem__(ele1, 0)
                    arrayobjs[ele1] += 1

    # 生成排序
    filename = "dev.zh"
    countrer(filename)
    filename = "train.zh"
    countrer(filename)
    filename = "tst.zh"
    countrer(filename)

    # 不在字典的排除
    with open("bak.vocab.zh.bak", 'rt', encoding="utf8") as f:
        resultv = f.readlines()
    resultv = [i.rstrip("\n") for i in resultv]

    print(len(arrayobjs))
    arraynews = {i[0]: i[1] for i in arrayobjs.items() if i[0] in resultv}
    print(len(arraynews))

    # 排序后最大的
    resdic = sorted(arrayobjs.items(), key=lambda x: x[1])
    with open("vocab.zh", 'wt', encoding="utf8") as f2:
        for i in resdic.items():
            f2.write(i[0])

    # 排序最大的n个
    import heapq
    a = [43, 5, 65, 4, 5, 8, 87]
    re1 = heapq.nlargest(3, a)  # 求最大的三个元素，并排序
    re2 = map(a.index, heapq.nlargest(3, a))  # 求最大的三个索引    nsmallest与nlargest相反，求最小
    print(re1)
    print(list(re2))  # 因为re2由map()生成的不是list，直接print不出来，添加list()就行了
    cheap = heapq.nsmallest(4, computers, key=lambda s: s['price'])
    expensive = heapq.nlargest(4, computers, key=lambda s: s['price'])


def regex():
    import re
    dir_name = "abv ADF：我们"
    # 1. 直接替换
    # 非英文和数字之后的替换为空
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    print(label_name)
    # 2. 先编译再替换
    p = re.compile(r'[^a-z0-9]+')
    # 非英文和数字之后的替换为空,最多2次
    label_name = p.sub(' ', dir_name.lower(), 2)
    print(label_name)
    # 3. 正则匹配 输出前n个
    pattern = re.compile(r'\d+')  # 查找数字
    result2 = pattern.findall('run88oob123google456', 0, 6)
    print(result2)
    # 3.1 返回括号里的
    label_name = re.findall(r'a(.*?)b', 'strss')
    # 4. 匹配任意字符
    a = r"^[\d\D]*?"
    # 5. 正则多字符分割
    re.split(r'[; |, |\*|\n]', "asdfdsdfgdff")
    # 6. 正则常用
    # \s 包含空白 \t|\r\n


def itertools():
    # 1. 排列组合
    shelf_list = [1, 2, 3]
    combnum = 4
    for i2 in itertools.combinations(shelf_list, combnum):
        print(i2)
    # 2. lists 合并
    batchids = list(itertools.chain(*[[i1, i1, i1, i1] for i1 in range(9)]))
    batchids = list(itertools.chain(*[i1["exercises"] for i1 in list(tts)]))
    # 3. json 合并
    batchbetch_all = {"19": 2, "31": 6}
    batchbetch1 = {"1": 2, "9": 2, "3": 4, "5": 6}
    batchbetch_all.update(batchbetch1)


def json_compare():
    src = {'numbers': [1, 3, 4, 8], 'foo': 'bar'}
    dst = {'foo': 'bar', 'numbers': [1, 3, 8]}
    patch = jsonpatch.JsonPatch.from_diff(src, dst)
    print(patch)


def json_manipulate():
    import json
    tmpjson = {"a": 1, "b": 2}
    # 删除
    del tmpjson['foo']
    # 不用ascii编码，缩进4格
    jsonstr = json.dumps(tmpjson, ensure_ascii=False, indent=4)
    # 用utf-8编码，缩进4格
    jsonobj = json.loads(jsonstr, encoding="utf-8")


def uuid_demo():
    import uuid

    name = 'test_name'
    # namespace = 'test_namespace'
    namespace = uuid.NAMESPACE_URL
    print(uuid.uuid1())
    print(uuid.uuid3(namespace, name))
    print(uuid.uuid4())
    print(uuid.uuid5(namespace, name))
    print(uuid.uuid1("2"))


def 判断是否是汉字():
    if u'\u4e00' <= ch <= u'\u9fff':
        return True


def nlp资料():
    "https://github.com/fighting41love/funNLP"
    # 1. 中英文敏感词过滤
    from txt.nlp_filter import DFAFilter
    f = DFAFilter()
    f.add("sexy")
    res = f.filter("hello sexy baby")
    # hello ** ** baby
    print(res)
    # 2. 97种语言检测
    import langid
    res = langid.classify("This is a test")
    print(res)
    # 3. 另一个语言检测
    from langdetect import detect
    from langdetect import detect_langs

    s1 = "本篇博客主要介绍两款语言探测工具，用于区分文本到底是什么语言，"
    s2 = 'We are pleased to introduce today a new technology'
    print(detect(s1))
    print(detect(s2))
    print(detect_langs(s1 + s2))  # detect_langs()输出探测出的所有语言类型及其所占的比例
    # 4. 中国手机归属地查询
    from phone import Phone
    p = Phone()
    res = p.find(18100065143)
    print(res)
    # return {'phone': '18100065143', 'province': '上海', 'city': '上海', 'zip_code': '200000', 'area_code': '021', 'phone_type': '电信'}
    # 5. 根据名字判断性别
    import ngender
    print(ngender.guess('赵本山'))
    # ('male', 0.9836229687547046)
    print(ngender.guess('宋丹丹'))
    # ('female', 0.9759486128949907)
    # 7. 抽取email的正则表达式
    import re
    text = "aaa"
    email_pattern = '^[*#\u4e00-\u9fa5 a-zA-Z0-9_.-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z0-9]{2,6}$'
    emails = re.findall(email_pattern, text, flags=0)
    # 8. 抽取phone_number的正则表达式
    cellphone_pattern = '^((13[0-9])|(14[0-9])|(15[0-9])|(17[0-9])|(18[0-9]))\d{8}$'
    phoneNumbers = re.findall(cellphone_pattern, text, flags=0)
    # 9. 抽取身份证号的正则表达式
    IDCards_pattern = r'^([1-9]\d{5}[12]\d{3}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])\d{3}[0-9xX])$'
    IDs = re.findall(IDCards_pattern, text, flags=0)
    # 10. 人名语料库 wainshine/Chinese-Names-Corpus
    # 11. 中文缩写库
    # 12. 汉语拆字词典 kfcd/chaizi
    # 13. 词汇情感值 rainarch/SentiBridge
    # 14. 中文词库、停用词、敏感词 dongxiexidian/Chinese
    # 15. 汉字转拼音 mozillazg/python-pinyin
    # 16. 中文繁简体互转 skydark/nstools
    # 17. 英文模拟中文发音引擎18. 汪峰歌词生成器
    # 19. 同义词库、反义词库、否定词库：guotong1988/chinese_dictionary
    # 20. 无空格英文串分割、抽取单词：wordinja
    import wordninja
    print(wordninja.split('derekanderson'))
    # ['derek', 'anderson']
    print(wordninja.split('imateapot'))
    # 21. IP地址正则表达式：
    # (25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)
    # 22. 腾讯QQ号正则表达式
    # [1-9]([0-9]{5,11})
    # 23. 国内固话号码正则表达式：
    # [0-9-()（）]{7,18}
    # 24. 用户名正则表达式：
    # [A-Za-z0-9_\-\u4e00-\u9fa5]+
    # 25. 汽车品牌、汽车零件相关词汇：
    # 26. 时间抽取 https://github.com/zhanzecheng/Time_NLP
    # Hi，all。下周一下午三点开会
    # 周一开会
    # 下下周一开会
    # 27. 各种中文词向量： https://github.com/Embedding/Chinese-Word-Vectors
    # 中文词向量大全
    # 28. 公司名字大全： https://github.com/wainshine/Company-Names-Corpus
    # 29. 古诗词库： https://github.com/chinese-poetry/chinese-poetry
    # 30. THU整理的词库： http://thuocl.thunlp.org/
    # 已整理到本repo的data文件夹中.
    # IT词库、财经词库、成语词库、地名词库、历史名人词库、诗词词库、医学词库、饮食词库、法律词库、汽车词库、动物词库
    # 31. 中文聊天语料 https://github.com/codemayq/chinese_chatbot_corpus
    # 该库搜集了包含:豆瓣多轮, PTT八卦语料, 青云语料, 电视剧对白语料, 贴吧论坛回帖语料,微博语料,小黄鸡语料
    # 32. 中文谣言数据: https://github.com/thunlp/Chinese_Rumor_Dataset
    # 33. 情感波动分析：https://github.com/CasterWx/python-girlfriend-mood/
    # 词库已整理到本repo的data文件夹中.
    # 本repo项目是一个通过与人对话获得其情感值波动图谱, 内用词库在data文件夹中.
    # 34. 百度中文问答数据集：https://pan.baidu.com/share/init?surl=QUsKcFWZ7Tg1dk_AbldZ1A 提取码: 2dva
    # 35. 句子、QA相似度匹配: https://github.com/NTMC-Community/MatchZoo
    # 37. Texar - Toolkit for Text Generation and Beyond: https://github.com/asyml/texar
    # 38. 中文事件抽取： https://github.com/liuhuanyong/ComplexEventExtraction
    # 中文复合事件抽取，包括条件事件、因果事件、顺承事件、反转事件等事件抽取，并形成事理图谱。
    # 39. cocoNLP: https://github.com/fighting41love/cocoNLP
    # 人名、地址、邮箱、手机号、手机归属地 等信息的抽取，rake短语抽取算法。
    from cocoNLP.extractor import extractor
    ex = extractor()
    text = '急寻特朗普，男孩，于2018年11月27号11时在陕西省安康市汉滨区走失。丢失发型短发，...如有线索，请迅速与警方联系：18100065143，132-6156-2938，baizhantang@sina.com.cn 和yangyangfuture at gmail dot com'
    # 抽取邮箱
    emails = ex.extract_email(text)
    print(emails)
    # ['baizhantang@sina.com.cn', 'yangyangfuture@gmail.com.cn']
    # 抽取手机号
    cellphones = ex.extract_cellphone(text, nation='CHN')
    print(cellphones)
    # ['18100065143', '13261562938']
    # 抽取手机归属地、运营商
    cell_locs = [ex.extract_cellphone_location(cell, 'CHN') for cell in cellphones]
    print(cell_locs)
    # cellphone_location [{'phone': '18100065143', 'province': '上海', 'city': '上海', 'zip_code': '200000', 'area_code': '021', 'phone_type': '电信'}]
    # 抽取地址信息
    locations = ex.extract_locations(text)
    print(locations)
    # ['陕西省安康市汉滨区', '安康市汉滨区', '汉滨区']
    # 抽取时间点
    times = ex.extract_time(text)
    print(times)
    # time {"type": "timestamp", "timestamp": "2018-11-27 11:00:00"}
    # 抽取人名
    name = ex.extract_name(text)
    print(name)
    # 40. 国内电话号码正则匹配（三大运营商+虚拟等） https://github.com/VincentSit/ChinaMobilePhoneNumberRegex
    # 41. 清华大学XLORE:中英文跨语言百科知识图谱:  https://xlore.org/download.html
    # 42. 清华大学人工智能技术系列报告
    # 47.用户名黑名单列表 https://github.com/marteinn/The-Big-Username-Blacklist
    # 48.罪名法务名词及分类模型: https://github.com/liuhuanyong/CrimeKgAssitant
    # 49.微信公众号语料: https://github.com/nonamestreet/weixin_public_corpus
    # 52.中文自然语言处理 语料/数据集：https://github.com/thunlp/THUOCL
    # 54.分词语料库+代码：https://pan.baidu.com/share/init?surl=MXZONaLgeaw0_TxZZDAIYQ  pea6
    # 59. Microsoft多语言数字/单位/如日期时间识别包 https://github.com/Microsoft/Recognizers-Text
    # 60. chinese-xinhua 中华新华字典数据库及api，包括常用歇后语、成语、词语和汉字 https://github.com/pwxcoo/chinese-xinhua
    # 61. 文档图谱自动生成 https://github.com/liuhuanyong/TextGrapher
    # 62. SpaCy 中文模型 https://github.com/howl-anderson/Chinese_models_for_SpaCy
    # 63. Common Voice语音识别数据集新版 https://voice.mozilla.org/en/datasets
    # 66. 关键词(Keyphrase)抽取包  https://github.com/boudinfl/pke
    # 67. 基于医疗领域知识图谱的问答系统 https://github.com/zhihao-chen/QASystemOnMedicalGraph
    # 68. 基于依存句法与语义角色标注的事件三元组抽取 https://github.com/liuhuanyong/EventTriplesExtraction
    # 69. 依存句法分析4万句高质量标注数据 by 苏州大学汉语依存树库（SUCDT） Homepage 数据下载详见homepage底部，需要签署协议，需要邮件接收解压密码。 http://hlt.suda.edu.cn/index.php/Nlpcc-2019-shared-task
    # 70. cnocr：用来做中文OCR的Python3包，自带了训练好的识别模型 https://github.com/breezedeus/cnocr
    # 71. 中文人物关系知识图谱项目 https://github.com/liuhuanyong/PersonRelationKnowledgeGraph
    # 72. 中文nlp竞赛项目及代码汇总 https://github.com/geekinglcq/CDCS
    # 73. 中文字符数据 https://github.com/skishore/makemeahanzi
    # 74. speech-aligner: 从“人声语音”及其“语言文本”，产生音素级别时间对齐标注的工具 https://github.com/open-speech/speech-aligner
    # 75. AmpliGraph: 知识图谱表示学习(Python)库：知识图谱概念链接预测 埃森哲出品，目前尚不支持中文 https://github.com/Accenture/AmpliGraph
    # 76. Scattertext 文本可视化(python) 很好用的工具包，简单修改后可支持中文 能否分析出某个类别的文本与其他文本的用词差异 https://github.com/JasonKessler/scattertext
    # 77. 语言/知识表示工具：BERT & ERNIE github  百度出品，ERNIE也号称在多项nlp任务中击败了bert https://github.com/PaddlePaddle/ERNIE
    # 79. Synonyms中文近义词工具包 Synonyms 中文近义词工具包，可以用于自然语言理解的很多任务：文本对齐，推荐算法，相似度计算，语义偏移，关键字提取，概念提取，自动摘要，搜索引擎等
    # https://github.com/huyingxi/Synonyms
    # 80. HarvestText领域自适应文本挖掘工具（新词发现-情感分析-实体链接等） https://github.com/blmoistawinde/HarvestText
    # 81. word2word：(Python)方便易用的多语言词-词对集：类似翻译 62种语言/3,564个多语言对 https://github.com/Kyubyong/word2word
    # 82. 语音识别语料生成工具：从具有音频/字幕的在线视频创建自动语音识别(ASR)语料库 https://github.com/yc9701/pansori
    # 84. 构建医疗实体识别的模型，包含词典和语料标注，基于python: https://github.com/yixiu00001/LSTM-CRF-medical
    # 85. 单文档非监督的关键词抽取： https://github.com/LIAAD/yake
    # 86. Kashgari中使用gpt-2语言模型 https://github.com/BrikerMan/Kashgari
    # 87. 开源的金融投资数据提取工具 https://github.com/PKUJohnson/OpenData
    # 89. 人民日报语料处理工具集 https://github.com/howl-anderson/tools_for_corpus_of_people_daily
    # 90. 一些关于自然语言的基本模型 https://github.com/lpty/nlp_base
    # 91. 基于14W歌曲知识库的问答尝试，功能包括歌词接龙，已知歌词找歌曲以及歌曲歌手歌词三角关系的问答 https://github.com/liuhuanyong/MusicLyricChatbot
    # 92. 基于Siamese bilstm模型的相似句子判定模型,提供训练数据集和测试数据集 https://github.com/liuhuanyong/SiameseSentenceSimilarity
    # 提供了10万个训练样本
    # 93. 用Transformer编解码模型实现的根据Hacker News文章标题自动生成评论 https://github.com/leod/hncynic
    # 94. 用BERT进行序列标记和文本分类的模板代码 https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification
    # 95. LitBank：NLP数据集——支持自然语言处理和计算人文学科任务的100部带标记英文小说语料 https://github.com/dbamman/litbank
    # 96. 百度开源的基准信息抽取系统 https://github.com/baidu/information-extraction
    # 97. 虚假新闻数据集 fake news corpus https://github.com/several27/FakeNewsCorpus
    # 98. Facebook: LAMA语言模型分析，提供Transformer-XL/BERT/ELMo/GPT预训练语言模型的统一访问接口 https://github.com/facebookresearch/LAMA
    # 99. CommonsenseQA：面向常识的英文QA挑战 https://www.tau-nlp.org/commonsenseqa
    # 100. 中文知识图谱资料、数据及工具 https://github.com/husthuke/awesome-knowledge-graph
    # 101. 各大公司内部里大牛分享的技术文档 PDF 或者 PPT https://github.com/0voice/from_coder_to_expert
    # 102. 自然语言生成SQL语句（英文） https://github.com/paulfitz/mlsql
    # 103. 中文NLP数据增强（EDA）工具 https://github.com/zhanlaoban/eda_nlp_for_Chinese 英文版 https://github.com/makcedward/nlpaug
    # 104. 基于医药知识图谱的智能问答系统 https://github.com/YeYzheng/KGQA-Based-On-medicine
    # 105. 京东商品知识图谱 https://github.com/liuhuanyong/ProductKnowledgeGraph
    # 基于京东网站的1300种商品上下级概念，约10万商品品牌，约65万品牌销售关系，商品描述维度等知识库，基于该知识库可以支持商品属性库构建，商品销售问答，品牌物品生产等知识查询服务，也可用于情感分析等下游应用．
    # *106. 基于mongodb存储的军事领域知识图谱问答项目 https://github.com/liuhuanyong/QAonMilitaryKG
    # 基于mongodb存储的军事领域知识图谱问答项目，包括飞行器、太空装备等8大类，100余小类，共计5800项的军事武器知识库，该项目不使用图数据库进行存储，通过jieba进行问句解析，问句实体项识别，基于查询模板完成多类问题的查询，主要是提供一种工业界的问答思想demo。
    # nlp工具 https://liuhuanyong.github.io./
    # 医药知识图谱的自动问答 https://github.com/liuhuanyong/QASystemOnMedicalKG.git
    # 抽象知识图谱 https://github.com/liuhuanyong/AbstractKnowledgeGraph.git
    # 107. 基于远监督的中文关系抽取 https://github.com/xiaolalala/Distant-Supervised-Chinese-Relation-Extraction
    # 108. 语音情感分析 https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer
    # 109. 中文ULMFiT 情感分析 文本分类 语料及模型 https://github.com/bigboNed3/chinese_ulmfit
    # 110. 一个拍照做题程序。输入一张包含数学计算题的图片，输出识别出的数学计算式以及计算结果 https://github.com/Roujack/mathAI
    # 110. 打印公式识别：https://github.com/LinXueyuanStdio/LaTeX_OCR
    # 110. 打印公式识别：https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO.git
    # 110. 打印公式识别：https://github.com/kullaheyo/LaTeX_OCR_PRO.git
    # 110. yolo原理与实现：https://blog.csdn.net/qq8993174/article/details/90038730
    # 111. 世界各国大规模人名库 https://github.com/philipperemy/name-dataset
    # 112. 一个利用有趣中文语料库 qingyun 训练出来的中文聊天机器人 https://github.com/Doragd/Chinese-Chatbot-PyTorch-Implementation
    # 使用了青云语料10万语料，本repo中也有该语料的链接
    # 113. 中文聊天机器人， 根据自己的语料训练出自己想要的聊天机器人，可以用于智能客服、在线问答、智能聊天等场景 https://github.com/zhaoyingjun/chatbot
    # 根据自己的语料训练出自己想要的聊天机器人，可以用于智能客服、在线问答、智能聊天等场景。加入seqGAN版本。
    # repo中提供了一份质量不太高的语料
    # 114. 省市区镇行政区划数据带拼音标注 https://github.com/xiangyuecn/AreaCity-JsSpider-StatsGov
    # 国家统计局中的省市区镇行政区划数据带拼音标注，高德地图的坐标和行政区域边界范围，在浏览器里面运行js代码采集的2019年发布的最新数据，含采集源码，提供csv格式数据，支持csv转成省市区多级联动js代码
    # 坐标、边界范围、名称、拼音、行政区等多级地址
    # 115. 教育行业新闻 自动文摘 语料库 https://github.com/wonderfulsuccess/chinese_abstractive_corpus
    # 116. 开放了对话机器人、知识图谱、语义理解、自然语言处理工具及数据 https://www.ownthink.com/#header-n30
    # 117. 中文知识图谱：基于百度百科中文页面，抽取三元组信息，构建中文知识图谱 https://github.com/lixiang0/WEB_KG
    # 118. masr: 中文语音识别，提供预训练模型，高识别率 https://github.com/libai3/masr
    # 119. Python音频数据增广库 https://github.com/iver56/audiomentations
    # 120. 中文全词覆盖BERT及两份阅读理解数据 https://github.com/ymcui/Chinese-BERT-wwm
    # DRCD数据集由中国台湾台达研究院发布，其形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集。
    # CMRC 2018数据集是哈工大讯飞联合实验室发布的中文机器阅读理解数据。根据给定问题，系统需要从篇章中抽取出片段作为答案，形式与SQuAD相同。
    # 121. ConvLab：开源多域端到端对话系统平台 https://github.com/ConvLab/ConvLab
    # 122. 中文自然语言处理数据集 https://github.com/InsaneLife/ChineseNLPCorpus
    # 123. 基于最新版本rasa搭建的对话系统 https://github.com/GaoQ1/rasa_chatbot_cn
    # 124. 基于TensorFlow和BERT的管道式实体及关系抽取 https://github.com/yuanxiaosc/Entity-Relation-Extraction
    # Entity and Relation Extraction Based on TensorFlow and BERT. 基于TensorFlow和BERT的管道式实体及关系抽取，2019语言与智能技术竞赛信息抽取任务解决方案。Schema based Knowledge Extraction, SKE 2019
    # 125. 一个小型的证券知识图谱/知识库 https://github.com/lemonhu/stock-knowledge-graph
    # 126. 复盘所有NLP比赛的TOP方案 https://github.com/zhpmatrix/nlp-competitions-list-review
    # 127. OpenCLaP：多领域开源中文预训练语言模型仓库 包含如下语言模型及百度百科数据 https://github.com/thunlp/OpenCLaP
    # 民事文书BERT bert-base 全部民事文书 2654万篇文书 22554词 370MB
    # 刑事文书BERT bert-base 全部刑事文书 663万篇文书 22554词 370MB
    # 百度百科BERT bert-base 百度百科 903万篇词条 22166词 367MB
    # 128. UER：基于不同语料、编码器、目标任务的中文预训练模型仓库（包括BERT、GPT、ELMO等） https://github.com/dbiir/UER-py
    # 基于PyTorch的预训练模型框架，支持对编码器，目标任务等进行任意的组合，从而复现已有的预训练模型，或在已有的预训练模型上进一步改进。基于UER训练了不同性质的预训练模型（不同语料、编码器、目标任务），构成了中文预训练模型仓库，适用于不同的场景。
    # 129. 中文自然语言处理向量合集 https://github.com/liuhuanyong/ChineseEmbedding
    # 包括字向量,拼音向量,词向量,词性向量,依存关系向量.共5种类型的向量
    # 130. 基于金融-司法领域(兼有闲聊性质)的聊天机器人 https://github.com/charlesXu86/Chatbot_CN
    # 其中的主要模块有信息抽取、NLU、NLG、知识图谱等，并且利用Django整合了前端展示,目前已经封装了nlp和kg的restful接口
    # 131. g2pC：基于上下文的汉语读音自动标记模块 https://github.com/Kyubyong/g2pC
    # 132. Zincbase 知识图谱构建工具包 https://github.com/tomgrek/zincbase
    # 133. 诗歌质量评价/细粒度情感诗歌语料库 https://github.com/THUNLP-AIPoet/Datasets
    # 134. 快速转化「中文数字」和「阿拉伯数字」 https://github.com/HaveTwoBrush/cn2an
    # 中文、阿拉伯数字互转
    # 中文与阿拉伯数字混合的情况，在开发中
    # 135. 百度知道问答语料库 https://github.com/liuhuanyong/MiningZhiDaoQACorpus
    # 超过580万的问题，938万的答案，5800个分类标签。基于该问答语料库，可支持多种应用，如闲聊问答，逻辑挖掘
    # *136. 基于知识图谱的问答系统 https://github.com/WenRichard/KBQA-BERT
    # BERT做命名实体识别和句子相似度，分为online和outline模式
    # 137. jieba_fast 加速版的jieba https://github.com/deepcs233/jieba_fast
    # 使用cpython重写了jieba分词库中计算DAG和HMM中的vitrebi函数，速度得到大幅提升
    # 138. 正则表达式教程 https://github.com/ziishaned/learn-regex/blob/master/translations/README-cn.md
    # 139. 中文阅读理解数据集 https://github.com/ymcui/Chinese-RC-Datasets
    # 140. 基于BERT等最新语言模型的抽取式摘要提取 https://github.com/Hellisotherpeople/CX_DB8
    # 141. Python利用深度学习进行文本摘要的综合指南 https://mp.weixin.qq.com/s/gDZyTbM1nw3fbEnU--y3nQ
    # 142. 知识图谱深度学习相关资料整理 https://github.com/lihanghang/Knowledge-Graph
    # 深度学习与自然语言处理、知识图谱、对话系统。包括知识获取、知识库构建、知识库应用三大技术研究与应用
    # 143. 维基大规模平行文本语料 https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix
    # 85种语言、1620种语言对、135M对照句
    # 144. StanfordNLP 0.2.0：纯Python版自然语言处理包 https://stanfordnlp.github.io/stanfordnlp/
    # 145. NeuralNLP-NeuralClassifier：腾讯开源深度学习文本分类工具 https://github.com/Tencent/NeuralNLP-NeuralClassifier
    # 146. 端到端的封闭域对话系统 https://github.com/cdqa-suite/cdQA
    # 147. 中文命名实体识别：NeuroNER vs. BertNER https://github.com/EOA-AILab/NER-Chinese
    # 148. 新闻事件线索抽取 https://github.com/liuhuanyong/ImportantEventExtractor
    # An exploration for Eventline (important news Rank organized by pulic time)，针对某一事件话题下的新闻报道集合，通过使用docrank算法，对新闻报道进行重要性识别，并通过新闻报道时间挑选出时间线上重要新闻
    # 149. 2019年百度的三元组抽取比赛，“科学空间队”源码(第7名) https://github.com/bojone/kg-2019
    # 150. 基于依存句法的开放域文本知识三元组抽取和知识库构建 https://github.com/lemonhu/open-entity-relation-extraction
    # 151. 中文的GPT2训练代码 https://github.com/Morizeyao/GPT2-Chinese
    # 152. ML-NLP - 机器学习(Machine Learning)、NLP面试中常考到的知识点和代码实现 https://github.com/NLP-LOVE/ML-NLP
    # 153. nlp4han:中文自然语言处理工具集(断句/分词/词性标注/组块/句法分析/语义分析/NER/N元语法/HMM/代词消解/情感分析/拼写检查 https://github.com/kidden/nlp4han
    # 154. XLM：Facebook的跨语言预训练语言模型 https://github.com/facebookresearch/XLM
    # 155. 用基于BERT的微调和特征提取方法来进行知识图谱百度百科人物词条属性抽取 https://github.com/sakuranew/BERT-AttributeExtraction
    # 156. 中文自然语言处理相关的开放任务，数据集, 以及当前最佳结果 https://github.com/didi/ChineseNLP
    # 157. CoupletAI - 基于CNN+Bi-LSTM+Attention 的自动对对联系统 https://github.com/WiseDoge/CoupletAI
    # 158. 抽象知识图谱，目前规模50万，支持名词性实体、状态性描述、事件性动作进行抽象 https://github.com/liuhuanyong/AbstractKnowledgeGraph
    # 159. MiningZhiDaoQACorpus - 580万百度知道问答数据挖掘项目 github


"""
    """

if __name__ == '__main__':
    nlp资料()
    exit()
    # cut_files()
    # json_sort()
    # n = 3000
    # static_top_n(n)
    uuid_demo()
