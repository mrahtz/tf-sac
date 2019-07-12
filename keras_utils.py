from collections import namedtuple

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.layers.base import Layer


class MLP(Model):
    def __init__(self, n_outputs):
        super().__init__()
        self.features = MLPFeatures()
        self.dense = Dense(n_outputs, activation=None)

    def call(self, x, **kwargs):
        x = self.features(x)
        x = self.dense(x)
        return x


class MLPFeatures(Model):
    def __init__(self):
        super().__init__()
        self.ls = [Dense(256, activation='relu'),
                   Dense(256, activation='relu')]

    def call(self, x, **kwargs):
        for l in self.ls:
            x = l(x)
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


PolicyOps = namedtuple('PolicyOps', 'raw_mean mean log_std pi log_prob_pi')


class Policy(Model):
    def __call__(self, inputs, *args, **kwargs) -> PolicyOps:
        return super().__call__(inputs, *args, **kwargs)


class Scale(Layer):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def call(self, x, **kwargs):
        return x * self.scale


class Tanh(Layer):
    def call(self, x, **kwargs):
        return tf.tanh(x)


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
        return (x + self.input_min) / self.input_scale * self.output_scale + self.output_min


def clip_but_pass_gradient(x, low=-1., high=1.):
    # From Spinning Up implementation
    clip_up = tf.cast(x > high, tf.float32)
    clip_low = tf.cast(x < low, tf.float32)
    return x + tf.stop_gradient((high - x) * clip_up + (low - x) * clip_low)
