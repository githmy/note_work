from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import io
import logging
import tempfile
import datetime
import os
from builtins import object
from concurrent.futures import ProcessPoolExecutor as ProcessPool
import utils
from future.utils import PY3
# from rasa_nlu import utils
# from rasa_nlu.components import ComponentBuilder
from ai_sets.config import AisetsConfig, parse_paras
# from rasa_nlu.evaluate import get_evaluation_metrics, clean_intent_labels
# from rasa_nlu.model import InvalidProjectError
from ai_sets.project_model import Project
# from rasa_nlu.train import do_train_in_worker
# from rasa_nlu.training_data.loading import load_data
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.logger import jsonFileLogObserver, Logger
from typing import Text, Dict, Any, Optional
from utils.path_tool import makesurepath
from utils.cmd_paras_check import config_key_parser, config_val_parser
from process_common.train_handl import do_pipline_in_worker
import simplejson

logger = logging.getLogger(__name__)

# in some execution environments `reactor.callFromThread` can not be called as it will result in a deadlock as
# the `callFromThread` queues the function to be called by the reactor which only happens after the call to `yield`.
# Unfortunately, the test is blocked there because `app.flush()` needs to be called to allow the fake server to
# respond and change the status of the Deferred on which the client is yielding. Solution: during tests we will set
# this Flag to `False` to directly run the calls instead of wrapping them in `callFromThread`.
DEFERRED_RUN_IN_REACTOR_THREAD = True


class InvalidProjectError(Exception):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class AlreadyTrainingError(Exception):
    """Raised when a training request is received for an Project already being trained.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = 'The project is already being trained!'

    def __str__(self):
        return self.message


def deferred_from_future(future):
    """Converts a concurrent.futures.Future object to a twisted.internet.defer.Deferred object.
    See: https://twistedmatrix.com/pipermail/twisted-python/2011-January/023296.html
    """
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


class DataRouter(object):
    def __init__(self, config, connection, modelpath, component_builder):
        self.responses = self._create_query_logger(config)
        self.model_dir = modelpath
        self.config = config
        self.connect = connection
        self.project_store = self._create_project_store()
        self._training_processes = max(self.config['max_training_processes'], 1)
        self.pool = ProcessPool(self._training_processes)

    def __del__(self):
        """Terminates workers pool processes"""
        self.pool.shutdown()

    def _create_query_logger(self, config):
        """Creates a logger that will persist incoming queries and their results."""

        # response_log_dir = config['response_log']
        rootpath = os.getcwd()
        for i in config["paths"]["rootpath"]:
            rootpath = os.path.join(rootpath, i)
        response_log_dir = rootpath
        for i in config["paths"]["syslogpath"]:
            response_log_dir = os.path.join(response_log_dir, i)
        makesurepath(response_log_dir)
        # Ensures different log files for different processes in multi worker mode
        if response_log_dir:
            # We need to generate a unique file name, even in multiprocess environments
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file_name = "aisets_request-{}-{}.log".format(timestamp,
                                                              os.getpid())
            response_logfile = os.path.join(response_log_dir, log_file_name)
            # Instantiate a standard python logger, which we are going to use to log requests
            utils.create_dir_for_file(response_logfile)
            query_logger = Logger(observer=jsonFileLogObserver(
                io.open(response_logfile, 'a', encoding='utf8')),
                namespace='query-logger')
            # Prevents queries getting logged with parent logger --> might log them to stdout
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info(
                "Logging of requests is disabled. (No 'request_log' directory configured)")
            return None

    def _collect_projects(self):
        if os.path.isdir(self.model_dir):
            # projects = os.listdir(self.model_dir)
            projects = [os.path.relpath(model, self.model_dir) for model in glob.glob(os.path.join(self.model_dir, '*'))
                        if os.path.isdir(model)]
        else:
            projects = []
        return projects

    def _create_project_store(self):
        projects = self._collect_projects()

        project_store = {}

        for project in projects:
            project_store[project] = Project(self.config, self.model_dir,
                                             None,
                                             project)

        if not project_store:
            project_store[AisetsConfig.DEFAULT_PROJECT_NAME] = Project(self.model_dir)
        return project_store

    def _list_projects_in_cloud(self):
        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(self.config)
            if p is not None:
                return p.list_projects()
            else:
                return []
        except Exception:
            logger.exception("Failed to list projects.")
            return []

    def _create_emulator(self):
        """Sets which NLU webservice to emulate among those supported by Rasa"""

        mode = self.config['emulate']
        if mode is None:
            from rasa_nlu.emulators import NoEmulator
            return NoEmulator()
        elif mode.lower() == 'wit':
            from rasa_nlu.emulators.wit import WitEmulator
            return WitEmulator()
        elif mode.lower() == 'luis':
            from rasa_nlu.emulators.luis import LUISEmulator
            return LUISEmulator()
        elif mode.lower() == 'dialogflow':
            from rasa_nlu.emulators.dialogflow import DialogflowEmulator
            return DialogflowEmulator()
        else:
            raise ValueError("unknown mode : {0}".format(mode))

    def extract(self, data):
        return self.emulator.normalise_request_json(data)

    def parse(self, data):
        project = data.get("project") or RasaNLUConfig.DEFAULT_PROJECT_NAME
        model = data.get("model")

        if project not in self.project_store:
            projects = self._list_projects(self.model_dir)

            cloud_provided_projects = self._list_projects_in_cloud()
            projects.extend(cloud_provided_projects)

            if project not in projects:
                raise InvalidProjectError(
                    "No project found with name '{}'.".format(project))
            else:
                try:
                    self.project_store[project] = Project(self.model_dir,
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

    @staticmethod
    def _list_projects(path):
        """List the projects in the path, ignoring hidden directories."""
        return [os.path.basename(fn)
                for fn in glob.glob(os.path.join(path, '*'))
                if os.path.isdir(fn)]

    @staticmethod
    def create_temporary_file(data, suffix=""):
        """Creates a tempfile.NamedTemporaryFile object for data"""

        if PY3:
            f = tempfile.NamedTemporaryFile("w+", suffix=suffix,
                                            delete=False, encoding="utf-8")
            f.write(data)
        else:
            f = tempfile.NamedTemporaryFile("w+", suffix=suffix,
                                            delete=False)
            f.write(data.encode("utf-8"))

        f.close()
        return f

    def parse_training_examples(self, examples, project, model):
        # type: (Optional[List[Message]], Text, Text) -> List[Dict[Text, Text]]
        """Parses a list of training examples to the project interpreter"""

        predictions = []
        for ex in examples:
            logger.debug("Going to parse: {}".format(ex.as_dict()))
            response, _ = self.project_store[project].parse(ex.text,
                                                            None,
                                                            model)
            logger.debug("Received response: {}".format(response))
            predictions.append(response)

        return predictions

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.

        return {
            "available_projects": {
                name: project.as_dict()
                for name, project in self.project_store.items()
            }
        }

    def _config_parse_one(self, singleconfig, keyt, wayt, typet):
        rule_col = "%s_%s_%s_1" % (keyt, wayt, typet)
        onerule = [i[rule_col] for i in self.connect.config_rule(rule_col)]
        if wayt == "key":
            if typet == "lest1":
                config_key_parser(singleconfig, lest1=onerule)
            elif typet == "neces":
                config_key_parser(singleconfig, neces=onerule)
        elif wayt == "val":
            if typet == "lest1":
                config_val_parser(singleconfig, keyt, lest1=onerule)
            elif typet == "neces":
                config_val_parser(singleconfig, keyt, neces=onerule)

    def _config_parse_twe(self, singleconfig, keyt, wayt, typet, classt):
        rule_col = "%s_%s_%s_%s_1" % (keyt, wayt, typet, classt)
        # print(rule_col)
        onerule = [i[rule_col] for i in self.connect.config_rule(rule_col)]
        if wayt == "key":
            if typet == "lest1":
                config_key_parser(singleconfig, lest1=onerule)
            elif typet == "neces":
                config_key_parser(singleconfig, neces=onerule)
        elif wayt == "val":
            if typet == "lest1":
                config_val_parser(singleconfig, keyt, lest1=onerule)
            elif typet == "neces":
                config_val_parser(singleconfig, keyt, neces=onerule)

    def _config_func(self, singleconfig, keyt, wayt, typet, classt):
        # rule_col = "%s_%s_%s_%s_1" % (keyt, wayt, typet, classt)
        onerule = [i[keyt] for i in self.connect.all_func()]
        if wayt == "key":
            if typet == "lest1":
                config_key_parser(singleconfig, lest1=onerule)
            elif typet == "neces":
                config_key_parser(singleconfig, neces=onerule)
        elif wayt == "val":
            if typet == "lest1":
                config_val_parser(singleconfig, keyt, lest1=onerule)
            elif typet == "neces":
                config_val_parser(singleconfig, keyt, neces=onerule)

    def _config_parse_all(self, config):
        pass

    def start_process(self, config_values):
        # type: (Text, Dict[Text, Any]) -> Deferred
        """Start a model training."""
        # 1. 路径配置
        _config = self.config.as_dict()
        rootpath = os.getcwd()
        for i in _config["paths"]["rootpath"]:
            rootpath = os.path.join(rootpath, i)
        model_path = rootpath
        for i in _config["paths"]["sysknowunit"]:
            model_path = os.path.join(model_path, i)
        model_path = os.path.join(model_path, str(config_values["project"]), str(config_values["branch"]))
        makesurepath(model_path)
        # 2. 配置解析
        configtree = self.connect.config_tree()
        configtree = simplejson.loads(configtree[0]["tree"])
        singleconfig = self.connect.unit_config(config_values["project"], config_values["branch"])
        singleconfig = simplejson.loads(singleconfig[0]["config"])
        customkey1, parajson = parse_paras(configtree, singleconfig, self._config_func, self._config_parse_twe)
        # 3. 流程调用
        print(customkey1)
        print(parajson)
        if customkey1 == "train":
            def training_callback(model_path):
                model_dir = os.path.basename(os.path.normpath(model_path))
                # self.project_store[project].update(model_dir)
                return model_dir

            def training_errback(failure):
                target_project = self.project_store.get(
                    failure.value.failed_target_project)
                if target_project:
                    target_project.status = 0
                return failure

            logger.debug("New training queued")
            result = self.pool.submit(do_pipline_in_worker, parajson)
            result = deferred_from_future(result)
            result.addCallback(training_callback)
            result.addErrback(training_errback)
        elif customkey1 == "predict":
            pass
        elif customkey1 == "hard":
            pass
        print("train end")
        return result

    def evaluate(self, data, project=None, model=None):
        # type: (Text, Optional[Text], Optional[Text]) -> Dict[Text, Any]
        """Perform a model evaluation."""

        project = project or RasaNLUConfig.DEFAULT_PROJECT_NAME
        model = model or None
        f = self.create_temporary_file(data, "_training_data")
        test_data = load_data(f.name)

        if project not in self.project_store:
            raise InvalidProjectError("Project {} could not "
                                      "be found".format(project))

        preds_json = self.parse_training_examples(test_data.intent_examples,
                                                  project,
                                                  model)

        predictions = [
            {"text": e.text,
             "intent": e.data.get("intent"),
             "predicted": p.get("intent", {}).get("name"),
             "confidence": p.get("intent", {}).get("confidence")}
            for e, p in zip(test_data.intent_examples, preds_json)
        ]

        y_true = [e.data.get("intent") for e in test_data.intent_examples]
        y_true = clean_intent_labels(y_true)

        y_pred = [p.get("intent", {}).get("name") for p in preds_json]
        y_pred = clean_intent_labels(y_pred)

        report, precision, f1, accuracy = get_evaluation_metrics(y_true,
                                                                 y_pred)

        return {
            "intent_evaluation": {
                "report": report,
                "predictions": predictions,
                "precision": precision,
                "f1_score": f1,
                "accuracy": accuracy}
        }
