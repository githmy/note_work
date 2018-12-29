import numpy as np
from os import path
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout


# This is abstract class. You need to implement yours.
class AbstractModeltensor(object):
    def __init__(self, config=None):
        self.config = config

    def getModel(self):
        model = self.buildModel()

        if "model_%s" % self.config["tailname"] and path.isdir("model_%s" % self.config["tailname"]):
            try:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                saver = tf.train.Saver()
                latest_ckpt = tf.train.latest_checkpoint("model_%s" % self.config["tailname"])
                saver.restore(sess, latest_ckpt)
            except Exception as e:
                print(e)

        return model

    # You need to override this method.
    def buildModel(self):
        raise NotImplementedError("You need to implement your own model.")
