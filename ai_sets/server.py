# -*- coding: utf-8 -*-
from builtins import str
from functools import wraps
from utils.cmd_paras_check import create_argparser
import datetime
import logging
import os
import json
import six
from functools import wraps
import glob
from builtins import str
from klein import Klein
from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks, returnValue
from utils.path_tool import makesurepath
from utils.timet import timeit
from utils import json_to_string
import simplejson
from ai_sets.train import *
from ai_sets.tool_db import Datadb, RDdb
from ai_sets.config import AisetsConfig
from ai_sets.project_model import Project
from ai_sets.config import __version__, DEFAULT_PROJECT_NAME
from ai_sets.data_router import DataRouter, InvalidProjectError, AlreadyTrainingError

logger = logging.getLogger(__name__)


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


class Ai_sets(object):
    """Class representing Ai-sets http server"""

    app = Klein()

    def __init__(self, config, component_builder=None, testing=False):
        # 1. 路径
        self._config = config
        # rootpath = os.getcwd()
        rootpath = "."
        for i in config["paths"]["rootpath"]:
            rootpath = os.path.join(rootpath, i)
        log_path = rootpath
        for i in config["paths"]["syslogpath"]:
            log_path = os.path.join(log_path, i)
        makesurepath(log_path)
        model_path = rootpath
        for i in config["paths"]["sysknowunit"]:
            model_path = os.path.join(model_path, i)
        makesurepath(model_path)
        self._modelpath = model_path

        # 2. 数据功能路由
        logger = logging.getLogger(__name__)

        logging.basicConfig(filename=os.path.join(log_path, "%s_%s.log" % (
            config["paths"]["sysloghead"], datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))), level='INFO')
        logging.captureWarnings(True)
        logger.debug("Creating a new data router")

        logger.debug("Configuration: " + config.view())
        logger.debug("Creating a new data router")
        self.config = config
        # 模型硬编码列表
        self.data_router = self._create_data_router(self.config, self._modelpath, component_builder)
        self._set_env = self._set_env_func(self._config)

    def _set_env_func(self, config):
        a = RDdb(config)
        a.set_env(config["server_ip"], config["port"], config["env_id"])

    def _create_data_router(self, config, modelpath, component_builder):
        return DataRouter(config, modelpath, component_builder)

    @app.route("/", methods=['GET', 'OPTIONS'])
    @check_cors
    def hello(self, request):
        """Main AI-SETS route to check if the server is online"""
        return "hello from AI-SETS: " + __version__

    def extract_data(self, data):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]
        _data = {}
        # _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]
        _data["text"] = data["q"] if type(data["q"]) == dict else None

        if not data.get("project"):
            _data["project"] = "default"
        elif type(data["project"]) == list:
            _data["project"] = data["project"][0]
        else:
            _data["project"] = data["project"]

        if data.get("model"):
            _data["model"] = data["model"][0] if type(data["model"]) == list else data["model"]

        if data.get("file"):
            _data["file"] = data["file"][0] if type(data["file"]) == list else data["file"]

        _data['time'] = data["time"] if "time" in data else None
        return _data

    def json_to_string(self, obj, **kwargs):
        indent = kwargs.pop("indent", 2)
        ensure_ascii = kwargs.pop("ensure_ascii", False)
        return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

    def _collect_projects(self):
        allprojects = []
        if os.path.isdir(self._modelpath):  # 1. 模型目录
            projects = os.listdir(self._modelpath)
            for project in projects:
                projectpaths = os.path.join(self._modelpath, project)
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
            project_store[project] = Project(self.config,
                                             self.component_builder,
                                             project)

        if not project_store:
            project_store[DEFAULT_PROJECT_NAME] = Project(
                self.config)
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
            projects = self._list_projects(self._modelpath)
            print(projects)

            if project not in projects:
                raise InvalidProjectError(
                    "No project found with name '{}'.".format(project))
            else:
                try:
                    self.project_store[project] = Project(self.config,
                                                          self.component_builder,
                                                          project)
                except Exception as e:
                    raise InvalidProjectError(
                        "Unable to load project '{}'. Error: {}".format(
                            project, e))

        time = data.get('time')
        response, used_model = self.project_store[project].parse(data['text'],
                                                                 time,
                                                                 model)

        if self.responses:
            self.responses.info('', user_input=response, project=project,
                                model=used_model)

        return self.format_response(response)

    @app.route("/version", methods=['GET', 'OPTIONS'])
    @requires_auth
    @check_cors
    def version(self, request):
        """Returns the Rasa server's version"""

        request.setHeader('Content-Type', 'application/json')
        print("version cost time")
        return json_to_string({'version': __version__})

    @app.route("/status", methods=['GET', 'OPTIONS'])
    @requires_auth
    @check_cors
    def status(self, request):
        request.setHeader('Content-Type', 'application/json')
        return json_to_string(self.data_router.get_status())

    @app.route("/func_single", methods=['POST', 'OPTIONS'])
    @requires_auth
    @check_cors
    @inlineCallbacks
    @timeit
    def func_single(self, request):
        print("***************  in other enviroment ***********************")
        data_string = request.content.read().decode('utf-8', 'strict')
        request.setHeader('Content-Type', 'application/json')
        try:
            request.setResponseCode(200)
            outjson = simplejson.loads(data_string)
            response = yield self.data_router.single_func(self._config.as_dict(), outjson)
            returnValue(json_to_string(response).encode("utf-8"))
        except Exception as e:
            returnValue(json_to_string({"error": "{}".format(e)}).encode("utf-8"))

    @app.route("/process", methods=['POST', 'OPTIONS'])
    @requires_auth
    @check_cors
    @inlineCallbacks
    @timeit
    def process(self, request):
        print("***************  in train ***********************")
        data_string = request.content.read().decode('utf-8', 'strict')
        request.setHeader('Content-Type', 'application/json')
        try:
            request.setResponseCode(200)
            tmpjson = simplejson.loads(data_string)
            try:
                tmpjson["usertype"]
            except:
                tmpjson["usertype"] = "general"
            try:
                tmpjson["userid"]
            except:
                tmpjson["userid"] = 0
            response = yield self.data_router.start_process(self._config, tmpjson)
            print("***************  end train ***********************")
            returnValue(json_to_string({'info': 'new model trained: {}'.format(response)}))
        except AlreadyTrainingError as e:
            request.setResponseCode(403)
            returnValue(json_to_string({"error": "{}".format(e)}))
        except InvalidProjectError as e:
            request.setResponseCode(404)
            returnValue(json_to_string({"error": "{}".format(e)}))
        except TrainingException as e:
            request.setResponseCode(500)
            returnValue(json_to_string({"error": "{}".format(e)}))

    @app.route("/parse", methods=['GET', 'POST', 'OPTIONS'])
    @requires_auth
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
            dumped = self.json_to_string({"error": "Invalid parse parameter specified"})
            returnValue(dumped)
        else:
            data = self.extract_data(request_params)
            try:
                request.setResponseCode(200)
                response = yield (self.parse(data) if self._testing
                                  else threads.deferToThread(self.parse, data))
                print(200000)
                returnValue(self.json_to_string(response))
            except InvalidProjectError as e:
                request.setResponseCode(404)
                returnValue(self.json_to_string({"error": "{}".format(e)}))
            except Exception as e:
                request.setResponseCode(500)
                logger.exception(e)
                returnValue(self.json_to_string({"error": "{}".format(e)}))

    @app.route("/update_model", methods=['POST', 'OPTIONS'])
    @requires_auth
    @check_cors
    @inlineCallbacks
    @timeit
    def update_model(self, request):
        """Returns the Rasa server's version"""

        request.setHeader('Content-Type', 'application/json')
        print("update_model cost time")
        return json_to_string({'version': __version__})


if __name__ == '__main__':
    # 1. 清洗数据
    arg_parser = create_argparser()
    # 1.1 命令行检验
    # args = arg_parser.parse_args()
    # 1.2  格式传入
    cmdline_args = {key: val
                    for key, val in list(vars(arg_parser.parse_args()).items())
                    if val is not None}
    aisets_config = AisetsConfig(os.environ, cmdline_args)
    # 2. 服务数据
    aisets = Ai_sets(aisets_config)
    logger.info('Started http server on port %s' % aisets_config["port"])
    aisets.app.run('0.0.0.0', aisets_config["port"])
    print("end")
