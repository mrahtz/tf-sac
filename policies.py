from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Lambda

from utils import get_features_model

Policy = namedtuple('Policy', 'mean log_std pi log_prob_pi')
EPS = 1e-8


def get_diagonal_gaussian_model(obs_dim: int, n_actions: int, act_lim: np.ndarray):
    assert act_lim.shape == (n_actions,)

    obs = Input(shape=(obs_dim,))
    features_model = get_features_model(obs_dim)
    features = features_model(obs)
    mean = Dense(n_actions, activation=None, name='fc_mean')(features)
    log_std = Dense(n_actions, activation=None, name='fc_std')(features)

    # Limit range of log_std to prevent numerical errors if it gets too large
    std_min, std_max = 1e-4, 4
    log_std_min, log_std_max = np.log([std_min, std_max])
    log_std_limit = lambda x: (tf.tanh(x) + 1) * 0.5 * (log_std_max - log_std_min) + log_std_min
    log_std = Lambda(log_std_limit)(log_std)

    pi = DiagonalGaussianSample([mean, log_std])

    tanh_pi = Lambda(lambda x: tf.tanh(x))(pi)
    log_prob_tanh_pi = TanhDiagonalGaussianLogProb([pi, tanh_pi, mean, log_std])

    scaled_tanh_pi = Lambda(lambda x: x * act_lim, name='pi_scale')(tanh_pi)
    scaled_tanh_mean = Lambda(lambda x: tf.tanh(x) * act_lim, name='mean_scale')(mean)

    mean_model = Model(inputs=[obs], outputs=[scaled_tanh_mean])
    log_std_model = Model(inputs=[obs], outputs=[log_std])
    pi_model = Model(inputs=[obs], outputs=[scaled_tanh_pi])
    log_prob_pi_model = Model(inputs=[obs], outputs=[log_prob_tanh_pi])

    return Policy(mean=mean_model,
                  log_std=log_std_model,
                  pi=pi_model,
                  log_prob_pi=log_prob_pi_model)


DiagonalGaussianSample = Lambda(
    name='act_sample',
    function=lambda mean_logstd_tup: diagonal_gaussian_sample(mean=mean_logstd_tup[0],
                                                              log_std=mean_logstd_tup[1]),
)

TanhDiagonalGaussianLogProb = Lambda(
    name='act_prob',
    function=lambda pi_tanhpi_mean_logstd_tup: tanh_diagonal_gaussian_log_prob(
        gaussian_samples=pi_tanhpi_mean_logstd_tup[0],
        tanh_gaussian_samples=pi_tanhpi_mean_logstd_tup[1],
        mean=pi_tanhpi_mean_logstd_tup[2],
        log_std=pi_tanhpi_mean_logstd_tup[3]),
)


def diagonal_gaussian_sample(mean, log_std):
    eps = tf.random.normal(tf.shape(mean))
    std = tf.exp(log_std)
    sample = mean + std * eps
    return sample


def tanh_diagonal_gaussian_log_prob(gaussian_samples, tanh_gaussian_samples, mean, log_std):
    assert len(tanh_gaussian_samples.shape) == 2
    log_prob = diagonal_gaussian_log_prob(gaussian_samples, mean, log_std)
    # tf.tanh can sometimes be > 1 due to precision errors
    tanh_gaussian_samples = clip_but_pass_gradient(tanh_gaussian_samples, l=0, u=1)
    log_prob -= tf.reduce_sum(tf.log(1 - tanh_gaussian_samples ** 2 + EPS), axis=1, keepdims=True)
    return log_prob


def diagonal_gaussian_log_prob(gaussian_samples, mean, log_std):
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
def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)
