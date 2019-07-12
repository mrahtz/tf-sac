from collections import namedtuple
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.layers.base import Layer

from utils import Features

EPS = 1e-8
Policy = namedtuple('Policy', 'mean log_std pi')
PolicyOps = namedtuple('PolicyOps', 'raw_mean mean log_std pi log_prob_pi')


class Policy(Model):
    def __call__(self, inputs, *args, **kwargs) -> PolicyOps:
        return super().__call__(inputs, *args, **kwargs)


class TanhDiagonalGaussianPolicy(Policy):
    def __init__(self, n_actions: int, act_lim: np.ndarray, std_min_max: Tuple[float, float]):
        assert act_lim.shape == (n_actions,)
        assert len(std_min_max) == 2
        super().__init__()

        self.log_std_min, self.log_std_max = np.log(std_min_max)
        self.act_lim = act_lim

        self.features = Features()
        self.mean = Dense(n_actions, activation=None)
        self.log_std = Dense(n_actions, activation=None)

    def call(self, obses, **kwargs):
        features = self.features(obses)

        mean = self.mean(features)
        log_std = self.log_std(features)

        # Limit range of log_std to prevent numerical errors if it gets too large
        log_std = Tanh()(log_std)
        log_std = Squash(in_min=-1, in_max=1, out_min=self.log_std_min, out_max=self.log_std_max)(log_std)

        pi = DiagonalGaussianSample()(mean=mean, log_std=log_std)

        tanh_pi = Tanh()(pi)
        log_prob_tanh_pi = TanhDiagonalGaussianLogProb()(gaussian_samples=pi,
                                                         tanh_gaussian_samples=tanh_pi,
                                                         mean=mean,
                                                         log_std=log_std)

        # Note that we limit the mean /after/ we've taken samples.
        # Otherwise, we would limit the mean, then also limit the sample,
        # leading to actions of scale tanh(1) when std is small.
        tanh_mean = Tanh()(mean)

        scaled_tanh_pi = Scale(self.act_lim)(tanh_pi)
        scaled_tanh_mean = Scale(self.act_lim)(tanh_mean)

        return PolicyOps(
            raw_mean=mean, mean=scaled_tanh_mean,
            log_std=log_std, pi=scaled_tanh_pi, log_prob_pi=log_prob_tanh_pi,
        )


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


class DiagonalGaussianSample(NamedInputsLayer):
    # noinspection PyMethodOverridin
    def call_named(self, mean, log_std):
        eps = tf.random.normal(tf.shape(mean))
        std = tf.exp(log_std)
        sample = mean + std * eps
        return sample


class TanhDiagonalGaussianLogProb(NamedInputsLayer):
    def call_named(self, gaussian_samples, tanh_gaussian_samples, mean, log_std):
        assert len(gaussian_samples.shape) == 2
        assert len(tanh_gaussian_samples.shape) == 2
        log_prob = DiagonalGaussianLogProb()(gaussian_samples=gaussian_samples,
                                             mean=mean,
                                             log_std=log_std)
        # tf.tanh can sometimes be > 1 due to precision errors
        tanh_gaussian_samples = clip_but_pass_gradient(tanh_gaussian_samples, low=-1, high=1)
        correction = tf.reduce_sum(tf.log(1 - tanh_gaussian_samples ** 2 + EPS), axis=1, keepdims=True)
        log_prob -= correction
        return log_prob


class DiagonalGaussianLogProb(NamedInputsLayer):
    def call_named(self, gaussian_samples, mean, log_std):
        assert len(gaussian_samples.shape) == 2
        n_dims = gaussian_samples.shape[1]
        assert gaussian_samples.shape.as_list() == [None, n_dims]

        std = tf.exp(log_std)
        log_probs_each_dim = -0.5 * np.log(2 * np.pi) - log_std - (gaussian_samples - mean) ** 2 / (2 * std ** 2 + EPS)
        assert log_probs_each_dim.shape.as_list() == [None, n_dims]

        # For a diagonal Gaussian, the probability of the random vector is the product of the probabilities
        # of the individual random variables. We're operating in log-space, so we can just sum.
        log_prob = tf.reduce_sum(log_probs_each_dim, axis=1, keepdims=True)
        assert log_prob.shape.as_list() == [None, 1]

        return log_prob


# From Spinning Up implementation
def clip_but_pass_gradient(x, low=-1., high=1.):
    clip_up = tf.cast(x > high, tf.float32)
    clip_low = tf.cast(x < low, tf.float32)
    return x + tf.stop_gradient((high - x) * clip_up + (low - x) * clip_low)
