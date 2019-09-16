from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from io import BytesIO
from utils.log_tool import logger, conf_path
import os
import json
import six
from functools import wraps
import glob
from builtins import str
from klein import Klein
from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks, returnValue
from utils.log_tool import basic_path, project_path, model_path, log_path
from utils.path_tool import makesurepath
from utils.timet import timeit
from data_router import InvalidProjectError, AlreadyTrainingError, DataRouter
import simplejson
from models.project_model import Project
from models.model_cnn import TextCNN
from interdata import *


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

        if origin:
            if '*' in self.config['cors_origins']:
                request.setHeader('Access-Control-Allow-Origin', '*')
            elif origin in self.config['cors_origins']:
                request.setHeader('Access-Control-Allow-Origin', origin)
            else:
                request.setResponseCode(403)
                return 'forbidden'

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

    def __init__(self, server_json, model_json):
        # 1. 路径
        self._server_json = server_json
        self._model_json = model_json
        self.model_instance_name = "{}-{}".format(self._server_json["model_type"], self._server_json["model_name"])
        self.model_instance = None
        # self._load_model()
        # # 2. 数据功能路由
        # self.data_router = DataRouter(self._model_json, model_path)
        # 3.
        # 3.1 训练接口-修改server配置文件后，调用程序后自动重启。如果没有该模型自动训练，如果有自动加载。
        # 3.2 预测接口-只调函数。
        # 3.3 修改权重接口-只改配置文件调用函数，不需重启。
        # 3.4 查看模型的接口-之列目录。
        # 3.5 更改模型-根据选项reload 或 修改配置server文件自动重启。

    @app.route("/", methods=['GET', 'OPTIONS'])
    @check_cors
    def hello(self, request):
        """Main delphis route to check if the server is online"""
        logger.info("in home")
        print("hello")
        return "hello from delphis: "

    def _collect_projects(self):
        allprojects = []
        if os.path.isdir(model_path):  # 1. 模型目录
            projects = os.listdir(model_path)
            for project in projects:
                projectpaths = os.path.join(model_path, project)
                if os.path.isdir(projectpaths):  # 2. 项目目录
                    filepas = os.listdir(projectpaths)
                    for filepa in filepas:
                        filepaths = os.path.join(projectpaths, filepa)
                        if os.path.isdir(filepaths):  # 3. timepath目录
                            timepas = os.listdir(filepaths)
                            for timepa in timepas:
                                timepaths = os.path.join(filepaths, timepa)
                                if os.path.isdir(timepaths):  # 3. timepath目录
                                    contenpas = os.listdir(timepaths)
                                    for content in contenpas:
                                        prolist = os.path.join(timepaths, content)
                                        allprojects.append(prolist)
        return allprojects

    def _create_project_store(self):
        projects = self._collect_projects()
        project_store = {}
        self.config = None
        self.component_builder = None
        for project in projects:
            project_store[project] = Project(self.config, project)
        if not project_store:
            project_store[DEFAULT_PROJECT_NAME] = Project(self.config)
        return project_store

    @staticmethod
    def _list_projects(path):
        """List the projects in the path, ignoring hidden directories."""
        return [os.path.basename(fn)
                for fn in glob.glob(os.path.join(path, '*'))
                if os.path.isdir(fn)]

    def parse(self, data):
        project = data.get("project") or DEFAULT_PROJECT_NAME
        model = data.get("model")

        if project not in self.project_store:
            # projects = self._list_projects(self.config['path']) model dir
            projects = self._list_projects(model_path)
            print(projects)

            if project not in projects:
                raise InvalidProjectError("No project found with name '{}'.".format(project))
            else:
                try:
                    self.project_store[project] = Project(self.config, project)
                except Exception as e:
                    raise InvalidProjectError("Unable to load project '{}'. Error: {}".format(project, e))

        time = data.get('time')
        response, used_model = self.project_store[project].parse(data['text'], time, model)

        if self.responses:
            self.responses.info('', user_input=response, project=project, model=used_model)

        return self.format_response(response)

    def _load_model(self):
        logger.info("***************  load ***********************")
        self.model_instance = TextCNN(self._server_json["model_type"], self._server_json["model_name"],
                                      self._model_json)
        self.model_instance.build()
        self.model_instance.load_mode("")

    @app.route("/reload", methods=['POST', 'OPTIONS'])
    # @requires_auth
    @check_cors
    @inlineCallbacks
    @timeit
    def reload_model(self, request):
        logger.info("***************  reload ***********************")
        print(request.content.read())
        data_string = request.content.read().decode('utf-8', 'strict')
        serverjson = json.loads(data_string, encoding="utf-8")
        self.model_instance = TextCNN(serverjson["model_type"], serverjson["model_name"], self._model_json)
        self.model_instance.model_instance_name = "{}-{}".format(serverjson["model_type"], serverjson["model_name"])
        self.model_instance.build()
        self.model_instance.load_mode("")
        returnValue(json.dumps({"response": "{}".format("ok")}, ensure_ascii=False, indent=4))

    @app.route("/trend", methods=['POST', 'OPTIONS'])
    @check_cors
    @inlineCallbacks
    def trend_back(self, request):
        logger.info("in home")
        bstr = request.content.read()
        request_params = simplejson.loads(bstr.decode('utf-8', 'strict'))
        return None
        datas = trend_back_interface(**request_params)
        return datas

    @app.route("/recommand", methods=['POST', 'OPTIONS'])
    @check_cors
    @inlineCallbacks
    def recommand_back(self, request):
        logger.info("in home")
        bstr = request.content.read()
        request_params = simplejson.loads(bstr.decode('utf-8', 'strict'))
        return None
        datas = recommand_back_interface(**request_params)
        return datas

    @app.route("/model", methods=['POST', 'OPTIONS'])
    @check_cors
    @inlineCallbacks
    def model_back(self, request):
        logger.info("in home")
        bstr = request.content.read()
        request_params = simplejson.loads(bstr.decode('utf-8', 'strict'))
        datas = model_back_interface(**request_params)
        return datas

    @app.route("/train", methods=['POST', 'OPTIONS'])
    # @requires_auth
    @check_cors
    @inlineCallbacks
    @timeit
    def train(self, request):
        logger.info("***************  in train ***********************")
        # 只接受 模型类型 和 自定义后缀，参数固化在对应目录的配置文件里。
        print(request.content.read())
        data_string = request.content.read()
        request.setHeader('Content-Type', 'application/json')
        #
        model = TextCNN(self._server_json["model_type"], self._server_json["model_name"], self._model_json)
        x_test, x_train, x_dev, y_train_m, y_dev_m, y_train_r, y_dev_r, y_train_l, y_dev_l = model.data4train()
        model.build()
        model.fit(x_train, x_dev, y_train_m, y_dev_m, y_train_r, y_dev_r, y_train_l, y_dev_l)
        # model.load_mode(model_json["model_name"])
        model.load_mode("")
        #
        try:
            request.setResponseCode(200)
            outjson = simplejson.loads(data_string)
            response = yield self.data_router.single_func(self._model_json.as_dict(), outjson)
            logger.info("***************  out train ***********************")
            returnValue(json.dumps({"response": "{}".format(response)}, ensure_ascii=False, indent=4))
        except AlreadyTrainingError as e:
            request.setResponseCode(403)
            returnValue(json.dumps({"error": "{}".format(e)}, ensure_ascii=False, indent=4))
        except InvalidProjectError as e:
            request.setResponseCode(404)
            returnValue(json.dumps({"error": "{}".format(e)}, ensure_ascii=False, indent=4))
        except TrainingException as e:
            request.setResponseCode(500)
            returnValue(json.dumps({"error": "{}".format(e)}, ensure_ascii=False, indent=4))

    @app.route("/predict", methods=['POST', 'OPTIONS'])
    # @requires_auth
    @check_cors
    @inlineCallbacks
    @timeit
    def predict(self, request):
        logger.info("***************  in predict ***********************")
        data_string = request.content.read().decode('utf8', 'strict')
        request.setHeader('Content-Type', 'application/json')
        print(data_string)
        print(json.loads(data_string))
        print(simplejson.loads(data_string))
        print(json.loads(data_string, encoding="utf-8"))
        x_test, x_train, x_dev, y_train_m, y_dev_m, y_train_r, y_dev_r, y_train_l, y_dev_l = self.model_instance.data4train()
        reslist = self.model_instance.predict(x_test)
        print(reslist[0].aa)
        print(reslist[0][0:2])
        print(reslist[1].shape)
        print(reslist[1][0:2])
        print(reslist[2].shape)
        print(reslist[2][0:2])
        request.setHeader('Content-Type', 'application/json')
        try:
            request.setResponseCode(200)
            tmpjson = simplejson.loads(data_string)
            logger.info(tmpjson)
            response = yield self.data_router.start_process(self._model_json, tmpjson)
            print("***************  end predict ***********************")
            returnValue(json.dumps({'info': 'new model trained: {}'.format(response)}))
        except Exception as e:
            returnValue(json.dumps({"error": "{}".format(e)}, ensure_ascii=False, indent=4))

    @app.route("/parse", methods=['GET', 'POST', 'OPTIONS'])
    # @requires_auth
    @check_cors
    @inlineCallbacks
    def parse_get(self, request):
        print("in server")
        request.setHeader('Content-Type', 'application/json')
        if request.method.decode('utf-8', 'strict') == 'GET':
            request_params = {key.decode('utf-8', 'strict'): value[0].decode('utf-8', 'strict')
                              for key, value in request.args.items()}
        else:
            request_params = simplejson.loads(request.content.read().decode('utf-8', 'strict'))

        if 'query' in request_params:
            request_params['q'] = request_params.pop('query')

        if 'q' not in request_params:
            request.setResponseCode(404)
            dumped = json.dumps({"error": "Invalid parse parameter specified"}, ensure_ascii=False, indent=4)
            returnValue(dumped)
        else:
            data = self.extract_data(request_params)
            try:
                request.setResponseCode(200)
                response = yield (threads.deferToThread(self.parse, data))
                returnValue(json.dumps(response, ensure_ascii=False, indent=4))
            except InvalidProjectError as e:
                request.setResponseCode(404)
                returnValue(json.dumps({"error": "{}".format(e)}, ensure_ascii=False, indent=4))
            except Exception as e:
                request.setResponseCode(500)
                logger.exception(e)
                returnValue(json.dumps({"error": "{}".format(e)}, ensure_ascii=False, indent=4))


def main():
    server_json = simplejson.load(open(os.path.join("config", "server.json"), encoding="utf8"))
    model_json = simplejson.load(open(os.path.join("config", "model.json"), encoding="utf8"))
    instd = Delphis(server_json, model_json)
    logger.info('Started http server on port %s' % server_json["port"])
    instd.app.run('0.0.0.0', server_json["port"])


if __name__ == '__main__':
    main()
