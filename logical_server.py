from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from io import BytesIO
import os
import json
import six
from functools import wraps
import glob
from builtins import str
from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet.defer import Deferred
from utils.path_tool import makesurepath
from utils.timet import timeit
import simplejson
from logical_solver import LogicalInference
from logical_solver import title_latex_prove
import pymysql
import threading
import uuid
import logging
from klein import Klein
import copy
from multiprocessing import Process
# from multiprocessing import Pool
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as ProcessPool

DEFERRED_RUN_IN_REACTOR_THREAD = True


def packanalyizefunc(t_content, dbconfig, ansid):
    # nodejson, edgelist = title_latex_prove(t_content)
    edgelist = json.load(open("../edgejson.json", "r"))
    nodejson = json.load(open("../nodejson.json", "r"))
    edgelist = json.dumps(edgelist, ensure_ascii=False)
    nodejson = json.dumps(nodejson, ensure_ascii=False)
    if nodejson is None:
        raise Exception("题目解析 或 生成思维树错误！")
    insert_title_sql = """UPDATE `titletab` SET `condition` = '{}' , trees ='{}' WHERE titleid='{}'""".format(nodejson,
                                                                                                              edgelist,
                                                                                                              ansid)
    mysql_ins = MysqlDB(dbconfig)
    title_content = mysql_ins.exec_sql(insert_title_sql)
    return title_content


class MysqlDB:
    def __init__(self, conf):
        self.config = {
            'host': conf['host'],
            'user': conf['user'],
            'port': conf['port'],
            'password': conf['password'],
            'database': conf['database'],
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        # self.config['cursorclass'] = pymysql.cursors.DictCursor
        self.conn = pymysql.connect(**self.config)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()
        # Log.sql_log.(error|info|debug)

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def exec_sql(self, strsql):
        """
        write data, such as insert, delete, update
        :param strsql: string sql
        :return: affected rows number
        return 0 when errors
        """
        try:
            self.lock.acquire()
            self.conn.ping(True)
            # print("sql in")
            res = self.cursor.execute(strsql)
            # print("sql res:",res)
            if strsql.strip().lower().startswith("select"):
                res = self.cursor.fetchall()
            self.conn.commit()
            return res
        except Exception as ex:
            print("exec sql error:")
            print(strsql, exc_info=True)
            return 0
        finally:
            self.lock.release()


class TrainingException(Exception):
    """Exception wrapping lower level exceptions that may happen while training

      Attributes:
          failed_target_project -- name of the failed project
          message -- explanation of why the request is invalid
      """

    def __init__(self, failed_target_project=None, exception=None):
        self.failed_target_project = failed_target_project
        if exception:
            self.message = exception.args[0]

    def __str__(self):
        return self.message


def check_cors(f):
    """Wraps a request handler with CORS headers checking."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]
        origin = request.getHeader('Origin')
        request.setHeader('Access-Control-Allow-Origin', '*')
        # request.setHeader('content-type', 'application/json;charset=utf-8')
        # if origin:
        #     if '*' in self.cors_origins:
        #         request.setHeader('Access-Control-Allow-Origin', '*')
        #     elif origin in self.cors_origins:
        #         request.setHeader('Access-Control-Allow-Origin', origin)
        #     else:
        #         request.setResponseCode(403)
        #         return 'forbidden'
        #     # if '*' in self.config['cors_origins']:
        #     #     request.setHeader('Access-Control-Allow-Origin', '*')
        #     # elif origin in self.config['cors_origins']:
        #     #     request.setHeader('Access-Control-Allow-Origin', origin)
        #     # else:
        #     #     request.setResponseCode(403)
        #     #     return 'forbidden'
        if request.method.decode('utf-8', 'strict') == 'OPTIONS':
            return ''  # if this is an options call we skip running `f`
        else:
            return f(*args, **kwargs)

    return decorated


def requires_auth(f):
    """Wraps a request handler with token authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        # return f(*args, **kwargs)
        self = args[0]
        request = args[1]
        if six.PY3:
            token = request.args.get(b'token', [b''])[0].decode("utf8")
        else:
            token = str(request.args.get('token', [''])[0])
        if self.config['token'] is None or token == self.config['token']:
            return f(*args, **kwargs)
        request.setResponseCode(401)
        return 'unauthorized'

    return decorated


class Delphis(object):
    """Class representing Ai-sets http server"""

    app = Klein()

    def __init__(self, server_json):
        # 1. 路径
        self._server_json = server_json
        self.dbconfig = {
            'host': self._server_json["database_ip"],
            'port': 3306,
            'database': self._server_json["database_name"],
            'user': self._server_json["user"],
            'password': self._server_json["password"],
            'charset': 'utf8mb4',  # 支持1-4个字节字符
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.mysql = MysqlDB(self.dbconfig)
        # 临时固定图片，映射 题目字符串 和 解答字符串
        pic_sql = """SELECT * FROM `picmap` """
        pic_content = self.mysql.exec_sql(pic_sql)
        self.picmap = {pic["picid"]: [pic["titleid"], pic["ansid"]] for pic in pic_content}
        self.ins_LI = LogicalInference()
        self.process_num = None
        self.process_pool = self.prepare_process()
        # result = self.process_pool.apply_async(long_time_task, args=(i,))

    def __del__(self):
        # self.process_pool.join()
        # self.process_pool.close()
        self.process_pool.shutdown()

    def prepare_process(self):
        print('Parent process %s.' % os.getpid())
        cores = multiprocessing.cpu_count()
        self.process_num = cores - 2
        # reactor.suggestThreadPoolSize(self.process_num)
        return ProcessPool(self.process_num)

    @app.route("/", methods=['GET', 'OPTIONS'])
    @check_cors
    def hello(self, request):
        """Main delphis route to check if the server is online"""
        print("hello")
        return "hello from delphis: "

    def get_report(self, piccontent):
        # 1. 图片转成 token
        namespace = uuid.NAMESPACE_URL
        picid = str(uuid.uuid3(namespace, piccontent))
        # print(piccontent)
        # 2. 如果没有图片token，手动改表 picmap, title answer 表补全字符串
        pic_sql = """SELECT * FROM `picmap` """
        pic_content = self.mysql.exec_sql(pic_sql)
        self.picmap = {pic["picid"]: [pic["titleid"], pic["ansid"]] for pic in pic_content}
        report = {"aa": 123}
        error = {}
        if picid not in self.picmap:
            error["图片id"] = "没有对应 题目id 和 解答id"
            pic_sql = """INSERT INTO `picmap` (picid) values ("{}")""".format(picid)
            pic_content = self.mysql.exec_sql(pic_sql)
            print("newpicture uuid:", picid)
            pic_sql = """SELECT * FROM `picmap` """
            pic_content = self.mysql.exec_sql(pic_sql)
            self.picmap = {pic["picid"]: [pic["titleid"], pic["ansid"]] for pic in pic_content}
        # 3. 如果有title内容，没有思维树，生成思维树，写入
        titleid, ansid = self.picmap[picid]
        if titleid is not None and ansid is not None:
            # 测试注销 start
            titlein_sql = """SELECT content, `condition`, trees FROM `titletab` where titleid={}""".format(titleid)
            # print(titlein_sql)
            title_content = self.mysql.exec_sql(titlein_sql)
            if len(title_content) == 0:
                error["题目id"] = "没有对应内容, 待写入"
            else:
                condition = title_content[0]["condition"]
                trees = title_content[0]["trees"]
                if condition is None or trees is None:
                    # 1. 解析题目，2. 序列化后写入数据库
                    # print("b process")
                    # paras0 = json.dumps(title_content[0]["content"], ensure_ascii=False)
                    # paras1 = json.dumps(self.dbconfig, ensure_ascii=False)
                    paras0 = title_content[0]["content"]
                    paras1 = self.dbconfig

                    def training_callback(model_path):
                        print(model_path)
                        return model_path

                    def training_errback(failure):
                        print("failure")
                        print(failure)
                        return failure

                    def deferred_from_future(future):
                        d = Deferred()

                        def callback(future):
                            e = future.exception()
                            if e:
                                if DEFERRED_RUN_IN_REACTOR_THREAD:
                                    reactor.callFromThread(d.errback, e)
                                else:
                                    d.errback(e)
                            else:
                                if DEFERRED_RUN_IN_REACTOR_THREAD:
                                    reactor.callFromThread(d.callback, future.result())
                                else:
                                    d.callback(future.result())

                        future.add_done_callback(callback)
                        return d

                    result = self.process_pool.submit(packanalyizefunc, paras0, paras1, titleid)
                    result = deferred_from_future(result)
                    result.addCallback(training_callback)
                    result.addErrback(training_errback)
                    # packanalyizefunc(paras0, paras1, titleid)
                    # retsig = self.process_pool.apply_async(packanalyizefunc, args=(paras0, paras1, titleid,))
                    error["解答id"] = "对应题目没有解析, 处理中"
        # 4. 测试注销 end
        if len(error.keys()) == 0:
            ansin_sql = """SELECT anstrs, anspoints, ansreports FROM `answertab` where ansid={}""".format(ansid)
            ansin_content = self.mysql.exec_sql(ansin_sql)
            if len(ansin_content) == 0:
                error["解答id"] = "没有对应内容, 待写入"
            else:
                anspoints = ansin_content[0]["anspoints"]
                ansreports = ansin_content[0]["ansreports"]
                if anspoints is None or ansreports is None:
                    # 1. 解析答案，2. 对比内容，3. 序列化后写入数据库
                    pass
                report = ansreports
            # 4. 返还信息
            # print("tree ok branch")
            report = json.dumps(report)
            error = json.dumps(error)
            return report, error
        else:
            error = {"message": "图片没有对应的 题目id 或 解答id,待写入。"}
            report = json.dumps(report)
            error = json.dumps(error)
            # print("tree waiting branch")
            return report, error

    @app.route("/recommand", methods=['POST', 'OPTIONS'])
    @check_cors
    @inlineCallbacks
    # @timeit
    def recommand_back(self, request):
        bstr = request.content.read()
        # print(111)
        # bstr = copy.deepcopy(bstr)
        # print(bstr)
        # print(bstr.decode('utf-8', 'strict'))
        request_params = simplejson.loads(bstr.decode('utf-8', 'strict'))
        # print(request_params)
        piccontent = request_params["content"]
        response, error = self.get_report(piccontent)
        # return json.dumps({'info': 'new model trained: {}'.format(datas)}, indent=4, ensure_ascii=False)
        # error = {}
        # response = [
        #     {"content": "因为 AM=PM","point": "平行定理", "istrue": True},
        #     {"content": "所以 AM=PM","point": "平行定理", "istrue": True},
        #     {"content": "因为 AB=MN","point": "全等三角形充分条件", "istrue": True},
        #     {"content": "所以 MB=PN","point": "全等三角形必要条件", "istrue": True},
        #     {"content": "因为 角BPQ=90度", "point": "平行定理", "istrue": True},
        #     {"content": "所以 角BPM+角NPQ=90度", "point": "全等三角形充分条件", "istrue": True},
        #     {"content": "因为 角MBP+角BPM=90度", "point": "全等三角形必要条件", "istrue": True},
        #     {"content": "所以 角MBP=角NPQ", "point": "平行定理", "istrue": True},
        #     {"content": "所以 三角形MBP 全等 三角形NPQ", "point": "全等三角形充分条件", "istrue": True},
        #     {"content": "所以 PB=PQ", "point": "全等三角形必要条件", "istrue": True},
        # ]
        # response = [
        #     {"content": "因为 AM=PM","point": "平行定理", "istrue": True},
        #     {"content": "所以 AM=PM","point": "平行定理", "istrue": True},
        #     {"content": "因为 AB=MN","point": "全等三角形充分条件", "istrue": True},
        #     {"content": "所以 MB=PN","point": "全等三角形必要条件", "istrue": True},
        #     {"content": "因为 角BPQ=90度", "point": "平行定理", "istrue": True},
        #     {"content": "所以 角BPM+角NPQ=90度", "point": "全等三角形充分条件", "istrue": True},
        #     {"content": "因为 角MBP+角BPM=90度", "point": "全等三角形必要条件", "istrue": True},
        #     {"content": "所以 角MPB=角NQP", "point": "平行定理", "istrue": True},
        #     {"content": "所以 三角形MBP 全等 三角形NPQ", "point": "全等三角形充分条件", "istrue": True},
        #     {"content": "所以 PB=PQ", "point": "全等三角形必要条件", "istrue": True},
        # ]
        if error is not None:
            # print(error)
            # dumped = yield json.dumps(error, indent=4, ensure_ascii=False)
            dumped = yield error
        else:
            # print(response)
            # dumped = yield json.dumps(response, indent=4, ensure_ascii=False)
            dumped = yield response
        returnValue(dumped)


def main():
    server_json = simplejson.load(open(os.path.join(".", "logical_server.json"), encoding="utf8"))
    instd = Delphis(server_json)
    print('Started http server on port %s' % server_json["port"])
    instd.app.run('0.0.0.0', server_json["port"])


if __name__ == '__main__':
    main()
