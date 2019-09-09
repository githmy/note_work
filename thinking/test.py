# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/24 16:16
# @Author  : abc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
from pprint import pprint
import pandas as pd
from dateutil import parser
from sklearn.cluster import KMeans
import copy
from utils.connect_mongo import MongoDB
from utils.connect_mysql import MysqlDB
from utils.log_tool import logger


class OutPutResult(object):
    def __init__(self, ):
        # 1. 采集数据源
        self.student_quality = [
            {"学生id": 1, "skill": 1, "thinking": 1, "hobby": 1, "school_type": 1, "student_point": 1}]
        self.point_info = [{"知识点": 1, "技能水平": 1, }]
        self.question_detial = [{"题目id": 1, "知识点": 1, "技能水平": 1, "建议时间": 1}]
        self.question_main = [{"题目id": 1, "思维水平": 1, "技能水平": 1}]
        self.quiz_detail = [{"学生id": 1, "题目id": 1, "测试时间": 1, "知识点id": 1, "答题时长": 1}]
        self.quiz_main = [{"测试id": 1, "学生id": 1, "试卷id": 1, "测试时间": 1, "skill": 1, "thinking": 1}]
        self.quiz_point = [
            {"测试id": 1, "知识点id": 1, "know": 1, "skill": 1, "thinking": 1, "答点时长": 1, "建议时长": 1.5, "是否超前": 1}]
        self.learing_info = [{"学生id": 1, "学习时间": 1, "知识点": 1, "学习时长": 1}]
        self.triple_info = [{"主知识点id": 1, "关系id": 1, "客知识点id": 1, "场景id": 1}]
        self.pd_student_quality = pd.DataFrame(self.student_quality)
        self.pd_point_info = pd.DataFrame(self.point_info)
        self.pd_question_detial = pd.DataFrame(self.question_detial)
        self.pd_question_main = pd.DataFrame(self.question_main)
        self.pd_quiz_detail = pd.DataFrame(self.quiz_detail)
        self.pd_quiz_main = pd.DataFrame(self.quiz_main)
        self.pd_quiz_point = pd.DataFrame(self.quiz_point)
        self.pd_learing_info = pd.DataFrame(self.learing_info)
        self.pd_triple_info = pd.DataFrame(self.triple_info)

    def update_point_skill_level(self):
        # 1. 输入：所有人 历次 学习 测试的平均时间消耗 和错误率，及相关知识点，输出：更新题目和知识点的技能难度等级。
        pass

    def points_teach_weigh(self):
        # 2. 输入：学习详情，历次测试，所属素质群体，输出：某素质群体，学习方式的比例。
        pass

    def points_file_weigh(self):
        # 3. 输入：学习详情，历次测试，所属素质群体，输出：知识点更新资料表的权重。（文件过多，可能不支持）
        pass

    def get_learn_info_multistudents(self, datalist):
        # 4. 学生的表现。
        curve_lists = []
        for student_s_list in datalist:
            curve_lists.append({"datetime": student_s_list["datetime"], "studentid": student_s_list["studentid"],
                                "data": self.get_learn_info_student(student_s_list["data"])})
        return curve_lists

    def get_learn_info_student(self, student_s_list):
        single_curve = []
        for execise_s_list in student_s_list:
            single_curve.append(self.get_learn_info_examination(execise_s_list))
        return single_curve

    def get_learn_info_examination(self, examination_obj):
        # 0:具体课程号，1:相对时间 2:详情
        # print(examination_obj)
        # 0:概念，1:技能，2:思维
        # 1 知识点维度，2 时间消耗，3 正确率
        point_base_obj = {}
        # 第一维 学习类型, 第二维 时间, 第三维 正确率
        way_base_obj = {"videinfo": [0, 0.0], "exampleinfo": [0, 0.0], "execiseinfo": [0, 0.0]}
        # 第一维 文件id, 第二维 时间, 第三维 知识点名, 第四维 正确率
        file_base_obj = {}
        # 象限图 按做题的 1. 次序号, 2. k, 3. S, 3. L
        curve_dim_obj = [examination_obj[1], 0, 0, 0, 0, 0, 0]
        # 线性图 按做题的 总分 1. 次序号，2. 得分，3. 总分
        score_p_curve = [examination_obj[1], 0, 0]
        # 三种学习形式
        for tmpobj in examination_obj[2]["videinfo"]:
            if tmpobj["PointCode"] not in point_base_obj:
                point_base_obj[tmpobj["PointCode"]] = [0, 0.0, 0.0]
            if tmpobj["SpentTime"] is None:
                tmpobj["SpentTime"] = 0
            point_base_obj[tmpobj["PointCode"]][0] += tmpobj["SpentTime"]
            way_base_obj["videinfo"][0] += tmpobj["SpentTime"]
            if tmpobj["FileId"] not in file_base_obj:
                file_base_obj[tmpobj["FileId"]] = [tmpobj["SpentTime"], tmpobj["PointCode"], 0.0]
            file_base_obj[tmpobj["FileId"]][0] += tmpobj["SpentTime"]
        for tmpobj in examination_obj[2]["exampleinfo"]:
            if tmpobj["SpentTime"] is None:
                tmpobj["SpentTime"] = 0
            if tmpobj["PointCode"] not in point_base_obj:
                point_base_obj[tmpobj["PointCode"]] = [0, 0.0, 0.0]
            point_base_obj[tmpobj["PointCode"]][0] += tmpobj["SpentTime"]
            way_base_obj["exampleinfo"][0] += tmpobj["SpentTime"]
            if tmpobj["ExampleId"] not in file_base_obj:
                file_base_obj[tmpobj["ExampleId"]] = [tmpobj["SpentTime"], tmpobj["PointCode"], 0.0]
            file_base_obj[tmpobj["ExampleId"]][0] += tmpobj["SpentTime"]
        for tmpobj in examination_obj[2]["execiseinfo"]:
            if tmpobj["spentTime"] is None:
                tmpobj["spentTime"] = 0
            if tmpobj["mainReviewPoints"][0] not in point_base_obj:
                point_base_obj[tmpobj["mainReviewPoints"][0]] = [0, 0.0, 0.0]
            point_base_obj[tmpobj["mainReviewPoints"][0]][0] += tmpobj["spentTime"]
            point_base_obj[tmpobj["mainReviewPoints"][0]][1] += tmpobj["actualScore"]
            point_base_obj[tmpobj["mainReviewPoints"][0]][2] += tmpobj["score"]
            way_base_obj["execiseinfo"][0] += tmpobj["spentTime"]
            if tmpobj["question"] not in file_base_obj:
                file_base_obj[tmpobj["question"]] = [tmpobj["spentTime"], tmpobj["mainReviewPoints"][0], 0.0]
            file_base_obj[tmpobj["question"]][0] += tmpobj["spentTime"]
            if tmpobj["qCategory"] == "K":
                curve_dim_obj[1] += tmpobj["actualScore"]
                curve_dim_obj[2] += tmpobj["score"]
            elif tmpobj["qCategory"] == "S":
                curve_dim_obj[3] += tmpobj["actualScore"]
                curve_dim_obj[4] += tmpobj["score"]
            elif tmpobj["qCategory"] == "L":
                curve_dim_obj[5] += tmpobj["actualScore"]
                curve_dim_obj[6] += tmpobj["score"]
        score_p_curve[1] = curve_dim_obj[1] + curve_dim_obj[3] + curve_dim_obj[5]
        score_p_curve[2] = curve_dim_obj[2] + curve_dim_obj[4] + curve_dim_obj[6]
        return {"point_base_obj": point_base_obj, "way_base_obj": way_base_obj, "file_base_obj": file_base_obj,
                "curve_dim_obj": curve_dim_obj, "score_p_curve": score_p_curve}

    def _lines2rank(self):
        # 知识点列变为行
        self.pd_quiz_point["efficient_rate"] = self.pd_quiz_point["建议时长"] / self.pd_quiz_point["答点时长"]
        self.pd_quiz_point["S"] = self.pd_quiz_point["skill"] * self.pd_quiz_point["efficient_rate"]
        self.pd_quiz_point["T"] = self.pd_quiz_point["thinking"] * self.pd_quiz_point["efficient_rate"]
        print(self.pd_quiz_point)
        # groupby分组操作
        reslist = []
        for name, group in self.pd_quiz_point.groupby('测试id'):
            print('*' * 13 + name + '*' * 13 + '\n', group)
            print()
            tmpjson = {"quizid": name}
            for i2 in self.pd_point_info["知识点"]:
                if i2 in group["知识点id"]:
                    ind = group["知识点id"].index(i2)
                    tmpjson[i2 + "_K"] = group.iloc[ind, "K"]
                    tmpjson[i2 + "_T"] = group.iloc[ind, "T"]
                else:
                    tmpjson[i2 + "_K"] = 0
                    tmpjson[i2 + "_T"] = 0
            reslist.append(tmpjson)
        res_pd = pd.DataFrame(reslist)
        return res_pd

    def _single_lines2rank(self, pd_in):
        # 知识点列变为行
        reslist = []
        tmpjson = {"quizid": pd_in.iloc[0, "测试id"]}
        for i2 in self.pd_point_info["知识点"]:
            if i2 in pd_in["知识点id"]:
                ind = pd_in["知识点id"].index(i2)
                tmpjson[i2 + "_K"] = pd_in.iloc[ind, "K"]
                tmpjson[i2 + "_T"] = pd_in.iloc[ind, "T"]
            else:
                tmpjson[i2 + "_K"] = 0
                tmpjson[i2 + "_T"] = 0
        reslist.append(tmpjson)
        res_pd = pd.DataFrame(reslist)
        return res_pd

    def update_student_quality(self, single_pd_in):
        # 5. 输入：历次测试2个维度的知识点降维结果：更新学生的素质属性。（为了得出发展轨迹）
        lines_pd = self._lines2rank()
        # 1. 统计 同一科目 同一学段同一学期 的 各个报名时间 各个学生 的发展轨迹，做学习模式聚类，作为背景参照。
        cls = KMeans(n_clusters=4, init='k-means++')
        y_hat = cls.fit_predict(lines_pd)
        # 2. 选   某一科目 某一学段某一学期 的 某个报名时间 某个学生 的评测结果。
        single_pd = self._single_lines2rank(single_pd_in)
        single_y_hat = cls.fit(single_pd)
        # 3. 属于 查看该学生属于某个类
        # 4. 画出 该科目 该学段该学期 该聚类 的 均值轨迹和波动值，外加拟合延长线
        return None

    def update_point_weigh(self):
        # 6. 输入：题库的统计结果，输出：所有相关的知识点按权重排序（同权重的或全选或全不先，题目出现多就代表重要）。
        pass

    def _get_point_parent(self):
        # 关系为 词"属于"，词域为"哲学概念"的id。场景 为词"考试知识点"，词域为"人事"的id。
        # 1. 对学生做错误知识点的聚类，
        # 2. 取最近的10个人离散到知识图谱上，
        # 3. 以错误基点为基础倒推父级下的有标记知识点
        pass

    def recommand_wrong_points(self):
        # 7. 输入：知识点权重，评测结果，对比同类错误学生(未必是同素质学生，只考虑错点，不考虑数量，取10个测试样本，离散到知识图谱上，以错误基点为基础倒退父级下的有标记知识点)，输出：同样错误的知识点。
        pass

    def output_general_curve_paras(self):
        # 统计准确率保持在95%以上，否则要减小素质群体的分类
        # 8. 输入：不同素质群体，不同时间(规范化学期开始为时间0)综合降维知识点的掌握情况，输出：不同素质群体，2个维度的曲线的拟合参数方差。
        pass

    def get_curve_trend_data(self, data_reform):
        # 9. 输入：学生素质，曲线基本参数，起始学习时间，知识点评测，返回最合理的预测：当前学生表现的象限能力，输出发展曲线 及标准差。
        two_curves = [[], []]
        for tmpobj in data_reform:
            oneline = copy.deepcopy(tmpobj)
            oneline["data"] = []
            for i1 in tmpobj["data"]:
                tt = i1["score_p_curve"]
                oneline["data"].append([tt[0], 100 * (tt[1] / tt[2])])
            multline = copy.deepcopy(tmpobj)
            multline["data"] = []
            for i1 in tmpobj["data"]:
                tt = i1["curve_dim_obj"]
                if tt[6] == 0:
                    tt[6] = 1e10
                multline["data"].append([tt[0], (tt[1] + tt[3]) / (tt[2] + tt[4]), tt[5] / tt[6]])
            two_curves[0].append(oneline)
            two_curves[1].append(multline)
        return two_curves

    def get_point_accum_data(self, data_reform):
        # 9. 输入：学生素质，曲线基本参数，起始学习时间，知识点评测，返回最合理的预测：当前学生表现的象限能力，输出发展曲线 及标准差。
        two_curves = [[], []]
        # for i in data_reform:
        #     two_curves[0]
        return None


class GetData(object):
    def __init__(self):
        # mysql
        config_my = {
            'host': "192.168.1.252",
            'port': 3306,
            'user': "thinking",
            'password': "thinking2018",
            'database': "htdb",
            'charset': 'utf8mb4',  # 支持1-4个字节字符
        }
        # config_my = {
        #     'host': "192.168.1.52",
        #     'port': 3306,
        #     'user': "thinking",
        #     'password': "thinking2019",
        #     'database': "htdb",
        #     'charset': 'utf8mb4',  # 支持1-4个字节字符
        # }
        self.insmysql = MysqlDB(config_my)
        # mongo
        config_mon = {
            'host': "192.168.1.252",
            'port': 27017,
            'database': "thinking2ht-test",
            'col': 'col',
            'charset': 'utf8mb4',  # 支持1-4个字节字符
        }
        # config_mon = {
        #     'host': "192.168.1.52",
        #     'port': 27017,
        #     'database': "thinking2ht",
        #     'col': 'col',
        #     'charset': 'utf8mb4',  # 支持1-4个字节字符
        # }
        self.insmongo = MongoDB(config_mon)

    # 2. 筛选课程详情编号
    def get_course_subject_stage_info(self, studentid, subjectid, sectionid, edition, timestart, timeend):
        # 教学相关
        # 课程id， 学习材料id，学生id, 学习日期, 学习形式（学练辅评）, 包含知识点的名称，技能等级，思维等级，学习时间， 建议学习时间
        sectionid = "aaaa"
        # 试卷id， 题目id，学生id, 考试日期, 包含知识点的名称，技能等级，思维等级， 答题时间， 建议答题时间
        if studentid is None:
            sqls = """
            SELECT studentid, CourseInstanceId, Id as CourseInstanceDtlId, startTime FROM bus_course_instance_dtl WHERE finishtime is not NULL and CourseInstanceId in (
              select Id from bus_course_instance WHERE Title='{}' AND startTime>='{}' AND startTime<='{}') ORDER BY StartTime;
            """.format(sectionid, timestart, timeend)
        else:
            sqls = """
            SELECT studentid, CourseInstanceId, Id as CourseInstanceDtlId, startTime FROM bus_course_instance_dtl WHERE finishtime is not NULL and studentid='{}' 
              AND CourseInstanceId in (select Id from bus_course_instance WHERE Title='{}' AND startTime>='{}' AND startTime<='{}') ORDER BY StartTime;
            """.format(studentid, sectionid, timestart, timeend)
        # subjectid,
        return self.insmysql.exec_sql(sqls)

    def get_course_video_info(self, coursedtlid):
        sqls = """
        SELECT FileId, PointCode, SpentTime FROM bus_course_file WHERE CourseDtlId='{}';""".format(coursedtlid)
        # print(sqls)
        return self.insmysql.exec_sql(sqls)

    def get_course_example_info(self, coursedtlid):
        sqls = """
        SELECT ExampleId, PointCode, SpentTime FROM bus_course_example WHERE CourseDtlId='{}';""".format(coursedtlid)
        return self.insmysql.exec_sql(sqls)

    def get_course_execise_info(self, coursedtlid):
        sqls = """
        SELECT CourseDtlId, QuestionId, PointCode FROM bus_course_question WHERE CourseDtlId='{}';""".format(
            coursedtlid)
        questioninfos = self.insmysql.exec_sql(sqls)
        return self.get_course_question_info(questioninfos)

    def get_course_question_info(self, questioninfos):
        # 1. 找出一次课程的所有习题id
        filter_dic = {"courseId": {"$in": [i1["CourseDtlId"] for i1 in questioninfos]}}
        show_dic = {"exercises": 1, "_id": 0}
        sql_tuple = (filter_dic, show_dic)
        colname = "studentassignments"
        execise_dtl = self.insmongo.exec_require(sql_tuple, colname)
        execise_dtl = list(execise_dtl)
        if len(execise_dtl) == 0:
            return None
        # 2. 找出题目详情
        colname = "studentexercises"
        filter_dic = {"_id": {"$in": list(itertools.chain(*[i1["exercises"] for i1 in execise_dtl]))}}
        show_dic = {"question": 1, "mainReviewPoints": 1, "reviewPoints": 1, "spentTime": 1, "mode": 1, "qCategory": 1,
                    "score": 1, "actualScore": 1, "_id": 0}
        sql_tuple = (filter_dic, show_dic)
        return list(self.insmongo.exec_require(sql_tuple, colname))

    def get_course_single_info(self, coursedtlid):
        # 2.1 找出测验id
        videinfo = self.get_course_video_info(coursedtlid)
        exampleinfo = self.get_course_example_info(coursedtlid)
        execiseinfo = self.get_course_execise_info(coursedtlid)
        return {"videinfo": videinfo, "exampleinfo": exampleinfo, "execiseinfo": execiseinfo}

    # 获取统计数据主要
    def get_course_result_info(self, courseinfo):
        # 筛选课程id 下的数据 整理成标准形式 [{起始时差:datatime, 课时id:xx, 知识点列表:[{知识点名称:xx, 学习时间:xx, 学习形式:xx, 成绩分值:xx, 成绩类型:xx}]}]
        # 1. 得出主列表Id
        cm_list = set([i1['CourseInstanceId'] for i1 in courseinfo])
        cd_dict = []
        for i1 in cm_list:
            tmp_cd_list = []
            tmptimelist = []
            tmppointinfo = []
            for i2 in courseinfo:
                if i2['CourseInstanceId'] == i1:
                    tmp_cd_list.append(i2['CourseInstanceDtlId'])
                    tmptimelist.append(i2['startTime'])
                    tmppointinfo.append(self.get_course_single_info(tmp_cd_list[-1]))
            tmptimelist = [(i2 - tmptimelist[0]).days for i2 in tmptimelist]
            cd_dict.append({"datetime": str(courseinfo[0]["startTime"]),
                            "studentid": courseinfo[0]["studentid"],
                            "data": list(zip(tmp_cd_list, tmptimelist, tmppointinfo))})
        # # [{课时id:xx, 起始时差:datatime, 知识评测完善度, 知识点列表:[{知识点名称:xx, 学习时间:xx, 学习形式:xx, 成绩分值:xx, 成绩类型:xx}]}]
        return cd_dict

    def get_quiz_subject_stage_info(self, subjectid, sectionid, edition, timestart, timeend):
        # 2. 答题相关
        # 2.1 找出测验id
        sectionid = "七年级上学期"
        # 试卷id， 题目id，学生id, 考试日期, 包含知识点的名称，技能等级，思维等级， 答题时间， 建议答题时间
        sqls = """
        SELECT Id FROM bus_course_instance_dtl WHERE CourseInstanceId in (
          select Id from bus_course_instance WHERE Title='{}' AND startTime>='{}' AND startTime<='{}');
        """.format(sectionid, timestart, timeend)
        # subjectid,
        cousids = self.insmysql.exec_sql(sqls)
        print(cousids)
        # 2.2 找出测验
        colname = "studentassignments"
        filter_dic = {"courseId": {"$in": [i1["Id"] for i1 in cousids]}}
        show_dic = {"exercises": 1, "_id": 0}
        sql_tuple = (filter_dic, show_dic)
        tts = self.insmongo.exec_require(sql_tuple, colname)
        # 2.3 找出题目详情
        colname = "studentexercises"
        filter_dic = {"testingTime": {"$gte": parser.parse(timestart), "$lte": parser.parse(timeend)},
                      "_id": {"$in": list(itertools.chain(*[i1["exercises"] for i1 in list(tts)]))}}
        show_dic = None
        sql_tuple = (filter_dic, show_dic)
        return self.insmongo.exec_require(sql_tuple, colname)

    def get_single_quiz_subject_stage_info(self, studentid, subjectid, sectionid, edition, timestart, timeend):
        # 2. 答题相关
        # 试卷id, 题目id, 学生id, 考试日期, 包含知识点的名称, 技能等级, 思维等级, 答题时间, 建议答题时间
        sqls = """
        select * from bus_course_instance WHERE StudentId='{}' and Title='{}' AND startTime>='{}' AND startTime<='{}';
        """.format(subjectid, sectionid, timestart, timeend)
        return self.insmysql.exec_sql(sqls)

    def get_student_info(self, subjectid, sectionid, edition):
        # 3. 背景信息
        # 学生的学校等级, 父母教育程度, 其它业余培训班类型，其它业余培训时间(多个培训班分别算)
        sqls = """
        SELECT * FROM bus_account_info WHERE `Section`="{}" AND Edition IN (
            SELECT `Value` FROM sys_dictdetail WHERE `Name`='{}');
        """.format(sectionid, edition)
        # subjectid,
        return self.insmysql.exec_sql(sqls)

    def get_subject_stage_point_info(self, subjectid, sectionid, edition):
        # 4. 学期的知识点
        # sys_dictdetail 教材版本id, --> 学段相应的章节 --> 章节相应的知识点
        sqls = """
        SELECT PointCode FROM bus_chapter_point WHERE ChapterId IN (
            SELECT Id FROM bus_chapter_info WHERE BookId IN (
                    SELECT Id FROM bus_book_info WHERE `subject`='{}' AND `Section`='{}' AND Edition IN (
                        SELECT `Value` FROM sys_dictdetail WHERE `Name`='{}')));
        """.format(subjectid, sectionid, edition)
        return self.insmysql.exec_sql(sqls)


def main():
    subjectid = "M"
    sectionid = "J"
    edition = "沪教版"
    timestart = "2019-01-01 00:00:00"
    timeend = "2019-09-17 00:00:00"
    insdata = GetData()
    # # 0. 单个学生的课程信息 studentid is not None
    # studentid = "6e3fedf0-33ab-4315-ba1b-31809d160c06"
    # coursedetailkeys = insdata.get_course_subject_stage_info(studentid, subjectid, sectionid, edition, timestart,
    #                                                          timeend)
    # single_course_res_info = insdata.get_course_result_info(coursedetailkeys)
    # 1. 筛选课程相关id  [{课程id:xx, 课程时间:xx}]  todo: 需要补充：sectionid
    studentid = None
    coursedetailkeys = insdata.get_course_subject_stage_info(studentid, subjectid, sectionid, edition, timestart,
                                                             timeend)
    # 2. 筛选课程id 下的数据 整理成标准形式 [{起始时差:datatime, 课时id:xx, 知识点列表:[{知识点名称:xx, 学习时间:xx, 学习形式:xx, 成绩分值:xx, 成绩类型:xx}]}]
    course_res_info = insdata.get_course_result_info(coursedetailkeys)
    # print(course_res_info)
    # quizres = insdata.get_quiz_subject_stage_info(subjectid, sectionid, edition, timestart, timeend)
    # 3. 学生相关信息    todo: 需要补充：是否根据 学科 学段 过滤，还是从 课程主表里筛选。
    studentinfo = insdata.get_student_info(subjectid, sectionid, edition)
    # print(studentinfo)
    # 4. 学期的知识点 ok
    pointobj = insdata.get_subject_stage_point_info(subjectid, sectionid, edition)
    datadic = {"course_res_info": course_res_info, "studentinfo": studentinfo,
               "pointobj": [i1["PointCode"] for i1 in pointobj]}
    # pprint(datadic)

    # 2. 分析返回
    insres = OutPutResult()
    # 数据格式 按各个维度 重新整理
    data_reform = insres.get_learn_info_multistudents(datadic["course_res_info"])
    # pprint(data_reform)
    # todo: 1. 根据 学生的 表现 得出 多条 成长曲线
    # todo: 1. 根据 学生的 表现 做 推题，预测
    # 2.1 按用户测评时间 输出所有 分值线 数据
    line_datas = insres.get_curve_trend_data(data_reform)
    # pprint(line_datas)
    # 2.2 按用户测评时间 输出所有 知识点累计值分值 数据
    line_datas = insres.get_point_accum_data(data_reform)
    pprint(line_datas)
    exit(0)
    # 2.3 按用户当前 知识点累计值分值 和 用户的素质 筛选类似的数据
    # 2.4 按用户 的错题 和 知识图谱的顺序，倒推数据
    single_pd_in = pd.DataFrame()
    insres.update_student_quality(single_pd_in)
    # 由于题目的思维难度根据步骤来，是客观的，不需要人为干预（暂时没考虑不同解决方案的步骤不同，即一道题多个解法的情况）


if __name__ == '__main__':
    logger.info("".center(100, "*"))
    logger.info("welcome to Delphis".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
    logger.info("")
    main()
    logger.info("")
    logger.info("bye bye".center(30, " ").center(100, "*"))
    logger.info("".center(100, "*"))
