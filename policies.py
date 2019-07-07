from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, Lambda

from utils import get_features_model

Policy = namedtuple('Policy', 'pi log_prob_pi')


def get_diagonal_gaussian_model(obs_dim, n_actions):
    obs = Input(shape=(obs_dim,))
    features_model = get_features_model(obs_dim)
    features = features_model(obs)
    mean = Dense(n_actions, activation=None, name='fc_mean')(features)
    # Standard deviation must be non-negative.
    # Instead of trying to incorporate that constraint into training,
    # it's easier to have the network output the log standard deviation
    # (which has no constraints) and then take the exp.
    log_std = Dense(n_actions, activation=None, name='fc_std')(features)

    pi = TanhDiagonalGaussianSample([mean, log_std])
    log_prob_pi = TanhDiagonalGaussianLogProb([pi, mean, log_std])

    pi_model = Model(inputs=obs, outputs=pi)
    log_prob_pi_model = Model(inputs=obs, outputs=log_prob_pi)

    return Policy(pi_model, log_prob_pi_model)


TanhDiagonalGaussianSample = Lambda(
    lambda mean_logstd_tup: diagonal_gaussian_sample(mean=mean_logstd_tup[0],
                                                     log_std=mean_logstd_tup[1])
)

TanhDiagonalGaussianLogProb = Lambda(
    lambda pi_mean_logstd_tup: tanh_diagonal_gaussian_log_prob(tanh_gaussian_samples=pi_mean_logstd_tup[0],
                                                               mean=pi_mean_logstd_tup[1],
                                                               log_std=pi_mean_logstd_tup[2])
)


def tanh_diagonal_gaussian_sample(mean, log_std):
    return tf.tanh(diagonal_gaussian_sample(mean, log_std))


def tanh_diagonal_gaussian_log_prob(tanh_gaussian_samples, mean, log_std):
    assert len(tanh_gaussian_samples.shape) == 2
    gaussian_sample = tf.atanh(tanh_gaussian_samples)
    log_prob = diagonal_gaussian_log_prob(gaussian_sample, mean, log_std)
    log_prob -= tf.reduce_sum(tf.log(1 - tanh_gaussian_samples ** 2), axis=1, keepdims=True)
    return log_prob


def diagonal_gaussian_sample(mean, log_std):
    eps = tf.random.normal(tf.shape(mean))
    std = tf.exp(log_std)
    sample = mean + std * eps
    return sample


def diagonal_gaussian_log_prob(gaussian_samples, mean, log_std):
    assert len(gaussian_samples.shape) == 2
    n_dims = gaussian_samples.shape[1]
    assert gaussian_samples.shape.as_list() == [None, n_dims]
    eps = 1e-8
    std = tf.exp(log_std)
    log_probs_each_dim = -0.5 * np.log(2 * np.pi) - log_std - (gaussian_samples - mean) ** 2 / (2 * std ** 2 + eps)
    assert log_probs_each_dim.shape.as_list() == [None, n_dims]
    # For a diagonal Gaussian, the probability of the random vector is the product of the probabilities
    # of the individual random variables. We're operating in log-space, so we can just sum.
    log_prob = tf.reduce_sum(log_probs_each_dim, axis=1, keepdims=True)
    assert log_prob.shape.as_list() == [None, 1]
    return log_prob
