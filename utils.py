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


def get_features_model(n_inputs, n_hidden=(256, 256)):
    obs = Input(shape=[n_inputs])
    h = obs
    for n, n_h in enumerate(n_hidden):
        h = Dense(n_h, activation='relu', name=f'h{n}')(h)
    return Model(inputs=obs, outputs=h)


def get_mlp_model(n_inputs, n_outputs):
    x = Input(shape=[n_inputs])
    features_model = get_features_model(n_inputs)
    features = features_model(x)
    outputs = Dense(n_outputs, activation=None, name='fc')(features)
    return Model(inputs=x, outputs=outputs)
