import os
import time

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.util import deprecation


def tf_disable_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tf_disable_deprecation_warnings():
    deprecation._PRINT_DEPRECATION_WARNINGS = False


class RateMeasure:
    def __init__(self, val):
        self.prev_t = self.prev_value = None
        self.reset(val)

    def reset(self, val):
        self.prev_value = val
        self.prev_t = time.time()

    def measure(self, val):
        val_change = val - self.prev_value
        cur_t = time.time()
        interval = cur_t - self.prev_t
        rate = val_change / interval

        self.prev_t = cur_t
        self.prev_value = val

        return rate


class Features(Model):
    def __init__(self):
        super().__init__()
        self.ls = [Dense(256, activation='relu'),
                   Dense(256, activation='relu')]

    def call(self, x, **kwargs):
        for l in self.ls:
            x = l(x)
        return x


class MLP(Model):
    def __init__(self, n_outputs):
        super().__init__()
        self.features = Features()
        self.dense = Dense(n_outputs, activation=None)

    def call(self, x, **kwargs):
        x = self.features(x)
        x = self.dense(x)
        return x
