import unittest

import numpy as np
import tensorflow as tf
from matplotlib.pyplot import hist, show, legend, plot, subplot, ylim, title

from model import SACModel
from policies import diagonal_gaussian_log_prob, tanh_diagonal_gaussian_sample, tanh_diagonal_gaussian_log_prob
from utils import tf_disable_warnings, tf_disable_deprecation_warnings

tf_disable_warnings()
tf_disable_deprecation_warnings()


class UnitTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_v_target_complete_update(self):
        obs_dim = 3
        obs = np.random.rand(obs_dim)
        model = self.get_model_with_polyak_coef(obs_dim, polyak_coef=0)

        v_main = self.get_v_main(model, obs)
        v_targ = self.get_v_targ(model, obs)
        with self.assertRaises(AssertionError):
            self.assertEqual(v_main, v_targ)

        model.sess.run(model.v_targ_polyak_update_op)

        v_main = self.get_v_main(model, obs)
        v_targ = self.get_v_targ(model, obs)
        self.assertEqual(v_main, v_targ)

    def test_v_target_no_update(self):
        obs_dim = 3
        obs = np.random.rand(obs_dim)
        model = self.get_model_with_polyak_coef(obs_dim, polyak_coef=1)

        v_targ_old = self.get_v_targ(model, obs)
        model.sess.run(model.v_targ_polyak_update_op)
        v_targ_new = self.get_v_targ(model, obs)
        self.assertEqual(v_targ_old, v_targ_new)

    def test_v_target_some_update(self):
        obs_dim = 3
        obs = np.random.rand(obs_dim)
        model = self.get_model_with_polyak_coef(obs_dim, polyak_coef=0.5, seed=2)

        v_main = self.get_v_main(model, obs)
        v_targ = self.get_v_targ(model, obs)
        prev_delta = abs(v_main - v_targ)
        self.assertGreater(prev_delta, 0.3)
        for _ in range(20):
            model.sess.run(model.v_targ_polyak_update_op)
            v_targ = self.get_v_targ(model, obs)
            new_delta = abs(v_main - v_targ)
            self.assertLess(new_delta, prev_delta)
            prev_delta = new_delta
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
        n_samples = 100000
        mean = 0.0
        std = 1.0
        log_std = np.log(std).astype(np.float32)

        expected_samples = np.tanh(np.random.normal(loc=mean, scale=std, size=[n_samples, 1]))

        sample = tanh_diagonal_gaussian_sample(mean=[0] * n_samples, log_std=[log_std] * n_samples)
        sess = tf.Session()
        actual_samples = sess.run(sample)

        subplot(1, 2, 1)
        title('Samples')
        hist(expected_samples, bins=100, alpha=0.5, density=True, label='Expected')
        hist(actual_samples, bins=100, alpha=0.5, density=True, label='Actual')
        legend()
        y_min, y_max = ylim()

        tanh_gaussian_sample_values = np.linspace([-0.99], [0.99], num=100, axis=0)
        ph = tf.placeholder(tf.float32, [None, 1])
        log_probs = tanh_diagonal_gaussian_log_prob(ph, mean=0, log_std=log_std)
        calculated_log_probs = sess.run(log_probs, feed_dict={ph: tanh_gaussian_sample_values})
        calculated_probs = np.exp(calculated_log_probs)

        subplot(1, 2, 2)
        title('Calculated PDF')
        plot(tanh_gaussian_sample_values, calculated_probs)
        ylim([y_min, y_max])
        show()

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
    def get_v_targ(model: SACModel, obs):
        v_targ = model.sess.run(model.v_targ_obs1, feed_dict={model.obs1: [obs]})[0]
        return v_targ

    @staticmethod
    def get_v_main(model: SACModel, obs):
        v_main = model.sess.run(model.v_main_obs1, feed_dict={model.obs1: [obs]})[0]
        return v_main

    @staticmethod
    def get_model_with_polyak_coef(obs_dim, polyak_coef, seed=0):
        model = SACModel(obs_dim=obs_dim, n_actions=5, seed=seed, discount=0.99,
                         temperature=1e-3, polyak_coef=polyak_coef, lr=1e-3)
        return model


if __name__ == '__main__':
    unittest.main()
