import jieba


def fenci(instr, paras):
    if paras["type"] == "jieba":
        return " ".join(jieba.cut(instr))
    elif paras["type"] == "hanlp":
        pass
    else:
        raise "分词error!"
