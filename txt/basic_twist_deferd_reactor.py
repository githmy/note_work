import multiprocessing
from concurrent.futures import ProcessPoolExecutor as ProcessPool
from klein import Klein
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet.defer import Deferred
from twisted.internet import reactor, threads
import json

DEFERRED_RUN_IN_REACTOR_THREAD = True


def packfunctt(A, B):
    "这部分不能卸载调用的类里"
    print(556)
    return "1"


class Delphis(object):
    """Class representing Ai-sets http server"""

    app = Klein()

    def __init__(self):
        cores = multiprocessing.cpu_count()
        self.process_pool = ProcessPool(cores - 1)

    def __del__(self):
        # self.process_pool.join()
        # self.process_pool.close()
        self.process_pool.shutdown()

    @app.route("/recommand", methods=['POST', 'GET', 'OPTIONS'])
    @inlineCallbacks
    def recommand_back(self, request):
        """Main delphis route to check if the server is online"""
        piccontent = "AAA"
        response = yield self.get_report(piccontent)
        returnValue(response)

    def get_report(self, piccontent):

        def training_callback(model_path):
            print("success")
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

        # result = self.process_pool.submit(packfunc, paras0, paras1)
        result = self.process_pool.submit(packfunctt, "para0", "paras1")
        result = deferred_from_future(result)
        result.addCallback(training_callback)
        result.addErrback(training_errback)
        return result


if __name__ == '__main__':
    instd = Delphis()
    print('Started http server on port %s' % "8080")
    instd.app.run('0.0.0.0', 8080)
