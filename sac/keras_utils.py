from collections import namedtuple
from typing import List, Tuple

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Concatenate
from tensorflow.python.layers.base import Layer

PolicyOps = namedtuple('PolicyOps', 'raw_mean mean log_std pi log_prob_pi')


class LinearOutputMLP(Model):
    def __init__(self, network: List[Tuple[int, str]], n_outputs):
        super().__init__()
        self._layers = []
        for n_units, activation in network:
            self._layers.append(Dense(n_units, activation))
        self._layers.append(Dense(n_outputs, activation=None))

    def call(self, x, **kwargs):
        if isinstance(x, list):
            x = Concatenate(axis=-1)(x)
        for layer in self._layers:
            x = layer(x)
        return x


class NamedInputsLayer(Layer):
    def __call__(self, **kwargs):
        self.names, inputs = list(zip(*kwargs.items()))
        return super().__call__(inputs=inputs)

    # noinspection PyMethodOverriding
    def call(self, inputs):
        kwargs = dict(zip(self.names, inputs))
        return self.call_named(**kwargs)

    def call_named(self, **kwargs):
        raise NotImplementedError()


class NamedInputsModel(Model):
    def __call__(self, **kwargs):
        self.names, inputs = list(zip(*kwargs.items()))
        return super().__call__(inputs=inputs)

    # noinspection PyMethodOverriding
    def call(self, inputs):
        kwargs = dict(zip(self.names, inputs))
        return self.call_named(**kwargs)

    def call_named(self, **kwargs):
        raise NotImplementedError()


class Policy(Model):
    def __call__(self, inputs, *args, **kwargs) -> PolicyOps:
        return super().__call__(inputs, *args, **kwargs)


class Squash(Layer):
    def __init__(self, in_min, in_max, out_min, out_max):
        super().__init__()
        self.input_min = in_min
        self.input_max = in_max
        self.input_scale = in_max - in_min
        self.output_min = out_min
        self.output_max = out_max
        self.output_scale = out_max - out_min

    def call(self, x, **kwargs):
        return (x - self.input_min) / self.input_scale * self.output_scale + self.output_min


def clip_but_pass_gradient(x, low=-1., high=1.):
    # From Spinning Up implementation
    clip_up = tf.cast(x > high, tf.float32)
    clip_low = tf.cast(x < low, tf.float32)
    return x + tf.stop_gradient((high - x) * clip_up + (low - x) * clip_low)
