import jieba


def fenci4(datalist, paras, env_json):
    """func_add
    "分词": {
      "func_name": "fenci",
      "illustration": "对句子进行分词。",
      "data": [[["说明1"],["说明2"],[]],[[输出说明1]]]",
      "paras": {
          "分词选项": ["jieba","hanlp"], 
          "特征选择默认值": 5,
          "自定义的参数": "sfdsaf",
          "函数功能名": {
              "函数1":{
                "para31": None
              }
          }
      }
      "performance": {
         "指标1": 说明
      }
    }
    """
    print("in function: fenci")
    if paras["type"] == "jieba":
        print(datalist[0])
        performance = {"a": 33}
        return [[(" ".join(jieba.cut(i)) for i in datalist[0])], performance]
    elif paras["type"] == "hanlp":
        pass
    else:
        raise "分词error!"
