import numpy as np
import re
import copy
import pandas as pd
from harvesttext.harvesttext import HarvestText
from abstract_concept import ConceptNet
import nltk
import sys

sys.path.append("code")  # 先跳出当前目录
from core.nlp import NLP
from core.extractor import Extractor


class QS(object):
    "解题类"

    def __init__(self):
        # 抽象实体
        self.abstract_entity = [""]
        # 知识图谱
        self.KG = [""]
        # 上下位图谱
        self.PG = [""]
        # 动作方式
        self.predicate_type = [""]
        # 解决空间
        self.solver_space = [{"steps": 0, "conditions": []}]
        self.ht = HarvestText()
        self.nlp = NLP()
        self.hiearchy_abs = ConceptNet()

    def question_solve_main(self, origin_question):
        "问题求解主函数"
        # 1. 句子拆分
        print("solving: ", origin_question)
        # split_sentences = re.split(r'[；;，,！。!\n]', origin_question)
        split_sentences = self.ht.cut_sentences(origin_question)
        print(split_sentences)
        # 2. 解析实体关系 origin_triples 的值含有最终求解类型
        # 三元组提取
        for sentence in split_sentences:
            # 分词处理
            lemmas = self.nlp.segment(sentence)
            print(lemmas)
            # 层级
            res_hiearchy = self.hiearchy_abs.gets_hiearchy(lemmas)
            print(res_hiearchy)
            # 词性标注
            words_postag = nltk.word_tokenize(" ".join(lemmas))
            # words_postag = nltk.pos_tag(" ".join(lemmas))
            words_postag = nltk.pos_tag(words_postag)
            # words_postag = nltk.word_tokenize("Dive into NLTK: Part-of-speech tagging and POS Tagger")
            # words_postag = self.nlp.postag(lemmas)
            print(words_postag)
            # 命名实体识别
            words_netag = self.nlp.netag(words_postag)
            print(words_netag)
            # 依存句法分析
            split_sentence = self.nlp.parse(words_netag)
            print(split_sentence)
        origin_triples = [self.ht.triple_extraction(sentence.strip()) for sentence in split_sentences]
        print(origin_triples)
        # origin_triples = [self.single_ner(sentence) for sentence in split_sentences]
        # 3. 步骤衍生条件triple
        pre_triples = copy.deepcopy(origin_triples)
        current_triples = []
        stepnum = 0
        while True:
            if len(pre_triples) == len(current_triples):
                break
            stepnum += 1
            current_triples = self.step_derivate(pre_triples)
            pre_triples = current_triples
            print("step {}. triples are: ".format(stepnum))
            print(current_triples)
        # 4. 条件问题检验
        solve_result = self.result_check(origin_triples, current_triples)
        print("results are: ", solve_result)
        return solve_result

    def single_ner(self, sentence):
        "单句解析实体关系"
        return ["", "", sentence]

    def step_derivate(self, pre_triples):
        "步骤衍生条件triple"
        current_triples = copy.deepcopy(pre_triples)
        return current_triples

    def result_check(self, origin_triples, triples):
        "检查结果"
        pdori_triple = pd.DataFrame(origin_triples)
        pdori_triple = pdori_triple[pdori_triple["value"].isnull()]
        pdfinal_triple = pd.DataFrame(triples)
        pdjoin = pd.merge(pdori_triple, pdfinal_triple, on="keys", how="left")
        return pdjoin

    def graph(self):
        "图谱分为：上下位图谱表"
        pass


def main():
    # question_str = """小丁丁在计算一道减法题时,把减数 $37$ 错看成 $73$ ，算出错误得数是 $209$ ，正确的得数时多少？"""
    question_str = """小丁丁在计算一道减法题时,把减数 $37$ 错看成 $73$ ，算出错误得数是 $209$ ，正确的得数时多少？"""
    ins = QS()
    solve_result = ins.question_solve_main(question_str)
    print(solve_result)


def test():
    pass


if __name__ == '__main__':
    # test()
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('maxent_treebank_pos_tagger')
    # exit()
    main()
