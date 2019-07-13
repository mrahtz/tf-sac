from typing import Tuple

import numpy as np
import tensorflow as tf

from sac.keras_utils import NamedInputsLayer, Squash, clip_but_pass_gradient, Policy, PolicyOps, LinearOutputMLP

EPS = 1e-8


class TanhDiagonalGaussianPolicy(Policy):
    def __init__(self, n_actions: int, act_lim: np.ndarray, std_min_max: Tuple[float, float]):
        assert act_lim.shape == (n_actions,)
        assert len(std_min_max) == 2
        super().__init__()

        self.log_std_min, self.log_std_max = np.log(std_min_max)
        self.act_lim = act_lim

        self.mean = LinearOutputMLP(n_actions)
        self.log_std = LinearOutputMLP(n_actions)

    def call(self, obses, **kwargs):
        mean = self.mean(obses)
        log_std = self.log_std(obses)

        # Limit range of log_std to prevent numerical errors if it gets too large
        log_std = tf.tanh(log_std)
        log_std = Squash(in_min=-1, in_max=1, out_min=self.log_std_min, out_max=self.log_std_max)(log_std)

        pi = DiagonalGaussianSample()(mean=mean, log_std=log_std)

        tanh_pi = tf.tanh(pi)
        log_prob_tanh_pi = TanhDiagonalGaussianLogProb()(gaussian_samples=pi,
                                                         tanh_gaussian_samples=tanh_pi,
                                                         mean=mean,
                                                         log_std=log_std)

        # Note that we limit the mean /after/ we've taken samples.
        # Otherwise, we would limit the mean, then also limit the sample,
        # leading to actions of scale tanh(1) when std is small.
        tanh_mean = tf.tanh(mean)

        scaled_tanh_pi = tanh_pi * self.act_lim
        scaled_tanh_mean = tanh_mean * self.act_lim

        return PolicyOps(
            raw_mean=mean, mean=scaled_tanh_mean,
            log_std=log_std, pi=scaled_tanh_pi, log_prob_pi=log_prob_tanh_pi,
        )


class DiagonalGaussianSample(NamedInputsLayer):
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
