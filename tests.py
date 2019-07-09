import unittest

import numpy as np
import tensorflow as tf
from matplotlib.pyplot import hist, show, legend, plot, subplot, ylim, title

from model import SACModel
from policies import diagonal_gaussian_log_prob, tanh_diagonal_gaussian_log_prob, \
    get_diagonal_gaussian_model, diagonal_gaussian_sample
from utils import tf_disable_warnings, tf_disable_deprecation_warnings

tf_disable_warnings()
tf_disable_deprecation_warnings()


class UnitTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        tf.random.set_random_seed(0)

    def test_v_target_complete_update(self):
        obs_dim = 3
        obs = np.random.rand(obs_dim)
        model = self._get_model_with_polyak_coef(obs_dim, polyak_coef=0)

        v_main = self._get_v_main(model, obs)
        v_targ = self._get_v_targ(model, obs)
        with self.assertRaises(AssertionError):
            self.assertEqual(v_main, v_targ)

        model.sess.run(model.v_targ_polyak_update_op)

        v_main = self._get_v_main(model, obs)
        v_targ = self._get_v_targ(model, obs)
        self.assertEqual(v_main, v_targ)

    def test_v_target_no_update(self):
        obs_dim = 3
        obs = np.random.rand(obs_dim)
        model = self._get_model_with_polyak_coef(obs_dim, polyak_coef=1)

        v_targ_old = self._get_v_targ(model, obs)
        model.sess.run(model.v_targ_polyak_update_op)
        v_targ_new = self._get_v_targ(model, obs)
        self.assertEqual(v_targ_old, v_targ_new)

    def test_v_target_some_update(self):
        obs_dim = 3
        obs = np.random.rand(obs_dim)
        model = self._get_model_with_polyak_coef(obs_dim, polyak_coef=0.5, seed=2)

        v_main = self._get_v_main(model, obs)
        v_targ = self._get_v_targ(model, obs)
        delta = abs(v_main - v_targ)
        self.assertGreater(delta, 0.15)
        for _ in range(20):
            model.sess.run(model.v_targ_polyak_update_op)
        v_targ = self._get_v_targ(model, obs)
        np.testing.assert_approx_equal(v_targ, v_main, significant=5)

    def test_gaussian_log_prob(self):
        self._test_gaussian_log_prob_correct(mean=0, std=1)
        self._test_gaussian_log_prob_correct(mean=0, std=2)
        self._test_gaussian_log_prob_correct(mean=1, std=1)
        self._test_gaussian_log_prob_correct(mean=1, std=2)
        self._test_gaussian_log_prob_correct(mean=3, std=1)
        self._test_gaussian_log_prob_correct(mean=3, std=2)
        self._test_gaussian_log_prob_correct(mean=-1, std=1)

        self._test_gaussian_log_prob_finite(mean=2, std=0.1)
        self._test_gaussian_log_prob_finite(mean=2, std=0.01)
        self._test_gaussian_log_prob_finite(mean=2, std=0.001)
        self._test_gaussian_log_prob_finite(mean=2, std=1e-8)

    def manual_test_tanh_gaussian(self):
        mean, std, n_samples = 0.0, 1.0, 100000
        log_std = np.log(std).astype(np.float32)

        expected_samples = np.tanh(np.random.normal(mean, std, [n_samples, 1]))
        actual_samples = self._get_tanh_gaussian_samples(mean, log_std, n_samples)
        subplot(1, 2, 1)
        y_min, y_max = self._plot_samples(actual_samples, expected_samples)

        probs, values = self._get_tanh_gaussian_pdf(log_std)
        subplot(1, 2, 2)
        self._plot_pdf(values, probs, y_min, y_max)

        show()

    @staticmethod
    def _get_tanh_gaussian_samples(mean, log_std, n_samples):
        sample_op = tf.tanh(diagonal_gaussian_sample(mean=[mean] * n_samples,
                                                     log_std=[log_std] * n_samples))
        sess = tf.Session()
        samples = sess.run(sample_op)
        return samples

    @staticmethod
    def _get_tanh_gaussian_pdf(log_std):
        tanh_gaussian_sample_values = np.linspace([-0.99], [0.99], num=100, axis=0)
        ph = tf.placeholder(tf.float32, [None, 1])
        log_probs = tanh_diagonal_gaussian_log_prob(ph, mean=0, log_std=log_std)
        sess = tf.Session()
        calculated_log_probs = sess.run(log_probs, feed_dict={ph: tanh_gaussian_sample_values})
        calculated_probs = np.exp(calculated_log_probs)
        return calculated_probs, tanh_gaussian_sample_values

    @staticmethod
    def _plot_samples(actual_samples, expected_samples):
        title('Samples')
        hist(expected_samples, bins=100, alpha=0.5, density=True, label='Expected')
        hist(actual_samples, bins=100, alpha=0.5, density=True, label='Actual')
        legend()
        y_min, y_max = ylim()
        return y_min, y_max

    @staticmethod
    def _plot_pdf(values, probs, y_min, y_max):
        title('Calculated PDF')
        plot(values, probs)
        ylim([y_min, y_max])

    def test_act_limit(self):
        act_lim = np.array([3, 5])
        feed_dict, log_std_op, mean_op, pi_op, train_high, train_low = self._get_ops(act_lim)

        for train_op in [train_low, train_high]:
            means, log_stds, pis = self._train(train_op, mean_op, log_std_op, pi_op, feed_dict)

            if train_op == train_high:
                self._check_means_increase(means)
            elif train_op == train_low:
                self._check_means_decrease(means)

            self._check_actions_saturated(act_lim, means[-1], log_stds[-1], pis[-1],
                                          (train_op == train_low),
                                          (train_op == train_high))

    @staticmethod
    def _get_ops(act_lim):
        policy = get_diagonal_gaussian_model(obs_dim=1, n_actions=2, act_lim=act_lim)

        obs_ph = tf.placeholder(tf.float32, shape=[None, 1])
        mean_op = policy.mean(obs_ph)
        log_std_op = policy.log_std(obs_ph)
        pi_op = policy.pi(obs_ph)

        train_low = tf.train.AdamOptimizer().minimize(pi_op)
        train_high = tf.train.AdamOptimizer().minimize(-pi_op)

        obs = np.random.rand(1, 1)
        feed_dict = {obs_ph: obs}

        return feed_dict, log_std_op, mean_op, pi_op, train_high, train_low

    @staticmethod
    def _train(train_op, mean_op, log_std_op, pi_op, feed_dict):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        vals = []
        for _ in range(1000):
            _, mean, log_std, pi = sess.run([train_op, mean_op, log_std_op, pi_op], feed_dict)
            vals.append((mean, log_std, pi))

        return map(np.array, zip(*vals))

    def _check_means_decrease(self, means):
        self.assertTrue(np.all(means[1:] < means[:-1]))

    def _check_means_increase(self, means):
        self.assertTrue(np.all(means[1:] > means[:-1]))

    def _check_actions_saturated(self, act_lim, mean, log_std, pi, train_low, train_high):
        assert np.array(mean).shape == np.array(log_std).shape == np.array(pi).shape == (1, 2)
        for i in range(len(act_lim)):
            if train_high:
                self.assertAlmostEqual(pi[0][i], act_lim[i], places=2)
            elif train_low:
                self.assertAlmostEqual(pi[0][i], -act_lim[i], places=2)
            else:
                raise RuntimeError()

    @staticmethod
    def _test_gaussian_log_prob_correct(mean, std):
        for x in [-1, 0, 1, mean, np.random.rand(),
                  np.random.normal(loc=mean, scale=std, size=(10, 3))]:
            if isinstance(x, (int, float)):
                x = np.array([[x]])
                n_samples, n_dims = 1, 1
            else:
                n_samples, n_dims = x.shape

            prob_each_dimension = 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
            assert prob_each_dimension.shape == (n_samples, n_dims)
            prob = np.prod(prob_each_dimension, axis=1, keepdims=True)
            log_prob = np.log(prob)
            expected = log_prob

            sess = tf.Session()
            ph = tf.placeholder(tf.float32, [None, n_dims])
            actual = sess.run(diagonal_gaussian_log_prob(ph, mean=mean, log_std=np.log(std).astype(np.float32)),
                              feed_dict={ph: x})

            np.testing.assert_array_almost_equal(actual, expected)

    def _test_gaussian_log_prob_finite(self, mean, std):
        for x in [-1, 0, 1, mean, np.random.rand()]:
            if isinstance(x, (int, float)):
                x = np.array([x])

            sess = tf.Session()
            ph = tf.placeholder(tf.float32, [None, 1])
            log_prob = sess.run(diagonal_gaussian_log_prob(ph, mean=mean, log_std=np.log(std).astype(np.float32)),
                                feed_dict={ph: [x]})
            self.assertTrue(np.all(np.isfinite(log_prob)))

    @staticmethod
    def _get_v_targ(model: SACModel, obs):
        v_targ = model.sess.run(model.v_targ_obs1, feed_dict={model.obs1: [obs]})[0]
        return v_targ

    @staticmethod
    def _get_v_main(model: SACModel, obs):
        v_main = model.sess.run(model.v_main_obs1, feed_dict={model.obs1: [obs]})[0]
        return v_main

    @staticmethod
    def _get_model_with_polyak_coef(obs_dim, polyak_coef, seed=0):
        model = SACModel(obs_dim=obs_dim, n_actions=2, seed=seed, discount=0.99,
                         temperature=1e-3, polyak_coef=polyak_coef, lr=1e-3,
                         act_lim=np.array([1, 1]))
        return model


if __name__ == '__main__':
    unittest.main()
