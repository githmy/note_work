import codecs
from nltk.corpus import stopwords
from textrank4zh import TextRank4Keyword, TextRank4Sentence

datastr = """
感受习主席六次新年贺词里的改革开放
学习专稿  来源：央视网 发布时间：2019年01月01日A-A+
我要分享
　　“我们改革的脚步不会停滞，开放的大门只会越开越大。”
　　在新年钟声即将敲响之际，国家主席习近平发表的二〇一九年新年贺词让我们对新的一年满怀信心和期待。
　　每到岁末年初，习近平都会发表新年贺词，回顾总结这一年的成绩，展望次年的美好前景，为中国人民鼓劲，并对全党全国各族人民发出接续奋斗的动员令。这些温暖人心、催人奋进的贺词，不仅是对党的十八大以来历史性成就的画龙点睛，更是全面深化改革璀璨成绩之精华版，为我们奏响新年的序曲和奋斗的旋律。
　　二〇一九年新年贺词：我们都是追梦人
　　“这一年，我们战胜各种风险挑战，推动经济高质量发展，加快新旧动能转换，保持经济运行在合理区间。蓝天、碧水、净土保卫战顺利推进……”
　　新年前夕，国家主席习近平通过中央广播电视总台和互联网，发表二〇一九年新年贺词。新华社记者鞠鹏摄
　　这一年，是全面贯彻党的十九大精神的开局之年，也是改革开放40周年。这一年，中国设立海南自贸区彰显扩大对外开放决心；博鳌亚洲论坛年会上，习近平向全世界自豪宣告：改革开放这场中国的第二次革命，不仅深刻改变了中国，也深刻影响了世界。
　　这一年，深化党和国家机构改革全面启动，100多项重要改革举措陆续推出，《关于实施乡村振兴战略的意见》、《关于打赢脱贫攻坚战三年行动的指导意见》等一系列政策文件推动改革不断深化。
　　2019年，中华人民共和国将迎来70周年华诞，这是决胜全面建成小康社会关键之年，更是机遇和挑战相互交织的一年。习近平信心百倍地发出新年动员令：“大家还要一起拼搏、一起奋斗”，唯有努力奔跑，才能追梦成真！
　　二〇一八年新年贺词：幸福都是奋斗出来的
　　“2017年，我们召开了中国共产党第十九次全国代表大会，开启了全面建设社会主义现代化国家新征程。我国国内生产总值迈上80万亿元人民币的台阶……”
　　这一年，河北雄安新区设立；营业税改征增值税改革全面完成；建设粤港澳大湾区成为国家战略。
　　这一年，习近平在金砖国家工商论坛开幕式上明示决心：事实证明，全面深化改革的路走对了，还要大步走下去。亚太经合组织工商领导人峰会上，习近平向世界宣告：中国改革的领域将更广、举措将更多、力度将更强。
　　面对即将到来的2018年，习近平用两个“必由之路”来强调改革开放的重大意义，并向全党全国发出奋斗的号召，“要以庆祝改革开放40周年为契机，逢山开路，遇水架桥，将改革进行到底。”
　　二〇一七年新年贺词：撸起袖子加油干
　　“十三五”实现了开门红；供给侧结构性改革迈出重要步伐；各领域具有四梁八柱性质的改革主体框架已经基本确立……习近平在新年贺词中对这非凡又难忘的一年做出盘点。
　　这一年，农村集体产权制度改革向全国推开；《关于深化投融资体制改革的意见》、《关于完善产权保护制度依法保护产权的意见》陆续印发，改革正蹄疾步稳地全面推进。
　　促改革、惠民生。这一年，习近平在新年贺词中欣慰地告诉大家：农村转移人口市民化更便利了，许多贫困地区孩子们上学条件改善了，老百姓异地办理身份证不用来回奔波了……
　　新年之际，习近平最牵挂的还是困难群众，他强调，全面深化改革要继续发力，“让改革发展成果惠及更多群众，让人民生活更加幸福美满”。
　　二〇一六年新年贺词：梦想总是可以实现的
　　“我国经济增长继续居于世界前列，改革全面发力，司法体制改革继续深化，‘三严三实’专题教育推动了政治生态改善……”
　　2015年，这是付出和收获都很多的一年。
　　这一年，“十二五”规划圆满收官，《生态文明体制改革总体方案》、《深化农村改革综合性实施方案》先后印发，确立了生态文明体制改革和农村改革的“四梁八柱”；《关于深化国有企业改革的指导意见》成为新时期指导和推进国企改革的纲领性文件；改革强军战略全面实施。
　　这一年，中共十八届五中全会明确了未来5年我国发展的方向，提出新发展理念；由习近平倡议筹建、旨在促进亚洲区域互联互通建设和经济一体化进程的亚洲基础设施投资银行正式成立。
　　“我们要树立必胜信念、继续埋头苦干，贯彻创新、协调、绿色、开放、共享的发展理念，着力推进结构性改革，着力推进改革开放......”，习近平对2016年前景发出令人鼓舞、催人奋进的动员令。
　　二〇一五年新年贺词：奋斗必将是艰巨的
　　“这一年，我们锐意推进改革，啃下了不少硬骨头，出台了一系列重大改革举措，许多改革举措同老百姓的利益密切相关。”
　　2014年被视为中国全面深化改革元年。这一年，中国通过立法确定了烈士纪念日；《关于全面深化农村改革加快推进农业现代化的若干意见》用“改革”推进“农业现代化”；《关于进一步推进户籍制度改革的意见》强力推进户籍改革，力争到2020年1亿农民落户城镇……
　　也是在这一年，习近平首次提出“新常态”重要论断，这一执政新理念给中国带来了新的发展机遇；这一年12月，习近平在江苏调研时首次将“四个全面”并提，“四个全面”成为引领民族复兴的战略布局。
　　蓝图越是宏伟，奋斗也必将艰巨。习近平在新年贺词中发出号令：“我们要继续全面深化改革，开弓没有回头箭，改革关头勇者胜”。
　　二〇一四年新年贺词：中国人民必将创造出新的辉煌
　　“2013年，我们对全面深化改革作出总体部署，共同描绘了未来发展的宏伟蓝图。2014年，我们将在改革的道路上迈出新的步伐。”
　　这一年，《中共中央关于全面深化改革若干重大问题的决定》提出全面深化改革的总目标并对全面深化改革作出总部署、总动员；9月、10月，习近平在出访时先后提出共同建设“丝绸之路经济带”与“21世纪海上丝绸之路”，即“一带一路”倡议；11月，习近平在湖南十八洞村首次提出“精准扶贫”理念;这一年岁末，由习近平任组长的中央全面深化改革领导小组成立。
　　2013年10月7日，国家主席习近平在印度尼西亚巴厘岛出席亚太经合组织工商领导人峰会，并发表《深化改革开放共创美好亚太》的重要演讲。新华社记者王晔摄
　　这一年，《大气污染防治行动计划》印发，“蓝天保卫战”全面打响；《关于调整完善生育政策的意见》印发，提出单独两孩的政策；中国首次超过美国，一跃成为世界第一货物贸易大国……
　　“在改革开放的伟大实践中，我们已经创造了无数辉煌。我坚信，中国人民必将创造出新的辉煌。”五年前，习近平由衷地发出赞叹，这是对改革开放的点赞，更是对中国人民奋斗精神的点赞；五年来，中国在富起来、强起来的征程上自信前行；展望未来，从大有作为到大有可为，我们还要一棒接着一棒跑下去，创造让世界刮目相看的新的更大奇迹！（文/刘雅虹）
"""


def extraction(datastr):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=datastr, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
    print('关键词：')
    for item in tr4w.get_keywords(20, word_min_len=1):
        print(item.word, item.weight)
    print()

    print('关键短语：')
    for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num=2):
        print(phrase)
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=datastr, lower=True, source='all_filters')
    print()

    print('摘要：')
    for item in tr4s.get_key_sentences(num=3):
        print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重


def MMR(datastr, extract_num, alpha):
    import os
    import re
    import jieba
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import operator

    # f = open(r'C:\Users\user\Documents\Python Scripts/stopword.dic')  # 停止词
    # stopwords = f.readlines()
    # stopwords = [i.replace("\n", "") for i in stopwords]
    stopwords = [""]

    def cleanData(name):
        setlast = jieba.cut(name, cut_all=False)
        seg_list = [i.lower() for i in setlast if i not in stopwords]
        return " ".join(seg_list)

    def calculateSimilarity(sentence, doc):  # 根据句子和句子，句子和文档的余弦相似度
        if doc == []:
            return 0
        vocab = {}
        for word in sentence.split():
            vocab[word] = 0  # 生成所在句子的单词字典，值为0

        docInOneSentence = '';
        for t in doc:
            docInOneSentence += (t + ' ')  # 所有剩余句子合并
            for word in t.split():
                vocab[word] = 0  # 所有剩余句子的单词字典，值为0

        cv = CountVectorizer(vocabulary=vocab.keys())

        docVector = cv.fit_transform([docInOneSentence])
        sentenceVector = cv.fit_transform([sentence])
        return cosine_similarity(docVector, sentenceVector)[0][0]

    tr4w = TextRank4Keyword()
    tr4w.analyze(text=datastr, lower=True, window=2)
    # data = open(r"C:\Users\user\Documents\Python Scripts\test.txt")  # 测试文件
    texts = tr4w.sentences
    texts = [i[:-1] if i[-1] == '\n' else i for i in texts]

    sentences = []
    clean = []
    originalSentenceOf = {}

    import time
    start = time.time()

    # Data cleansing
    for part in texts:
        cl = cleanData(part)  # 句子切分以及去掉停止词
        #		print (cl)
        sentences.append(part)  # 原本的句子
        clean.append(cl)  # 干净有重复的句子
        originalSentenceOf[cl] = part  # 字典格式
    setClean = set(clean)  # 干净无重复的句子

    # calculate Similarity score each sentence with whole documents
    scores = {}
    for data in clean:
        temp_doc = setClean - set([data])  # 在除了当前句子的剩余所有句子
        score = calculateSimilarity(data, list(temp_doc))  # 计算当前句子与剩余所有句子的相似度
        scores[data] = score  # 得到相似度的列表

    # calculate MMR
    # n = 25 * len(sentences) / 100  # 摘要的比例大小
    summarySet = []
    while extract_num > 0:
        mmr = {}
        # kurangkan dengan set summary
        for sentence in scores.keys():
            if not sentence in summarySet:
                mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence,
                                                                                             summarySet)  # 公式
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        summarySet.append(selected)
        #	print (summarySet)
        extract_num -= 1

    # rint str(time.time() - start)

    print('Summary:')
    for sentence in summarySet:
        print(originalSentenceOf[sentence].lstrip(' '))
    print('=============================================================')
    print('Original Passages:')


if __name__ == '__main__':
    # extraction(datastr)
    alpha = 0.4  # 越小差异越大
    extract_num = 5
    MMR(datastr, extract_num, alpha)
    from sklearn.datasets import make_regression
    from sklearn.lda import LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
