import os
import pickle
import time
from glob import glob

import easy_tf_log
import tensorflow as tf
from gym.utils.atomic_write import atomic_write

from sac.keras_utils import LinearOutputMLP, NamedInputsModel
from sac.policies import TanhDiagonalGaussianPolicy
from sac.replay_buffer import ReplayBatch
from sac.utils import LogMilliseconds


class Q(NamedInputsModel, LinearOutputMLP):
    # Wrapper enforcing consistent concatenation order of obs and act
    def call_named(self, obs, act):
        return LinearOutputMLP.call(self, [obs, act])


class SACModel:

    def __init__(self, obs_dim, n_actions, act_lim, seed, discount, temperature, polyak_coef, lr, std_min_max,
                 network, save_dir=None):
        self.args_copy = self._args_from_locals(locals())
        self.n_actions = n_actions
        self.save_dir = save_dir

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.random.set_random_seed(seed)

            obs1 = tf.placeholder(tf.float32, (None, obs_dim))
            obs2 = tf.placeholder(tf.float32, (None, obs_dim))
            acts = tf.placeholder(tf.float32, (None, n_actions))
            rews = tf.placeholder(tf.float32, [None, 1])
            done = tf.placeholder(tf.float32, [None, 1])

            policy_model = TanhDiagonalGaussianPolicy(network=network, n_actions=n_actions,
                                                      act_lim=act_lim, std_min_max=std_min_max)
            ops = policy_model(obs1)
            mean_obs1, pi_obs1, log_prob_pi_obs1 = ops.mean, ops.pi, ops.log_prob_pi

            q1_model = Q(network, n_outputs=1)
            q2_model = Q(network, n_outputs=1)
            v_main_model = LinearOutputMLP(network, n_outputs=1)
            v_targ_model = LinearOutputMLP(network, n_outputs=1)

            q1_obs1_acts = q1_model(obs=obs1, act=acts)
            q2_obs1_acts = q2_model(obs=obs1, act=acts)
            min_q12_obs1_pi = tf.minimum(q1_model(obs=obs1, act=pi_obs1),
                                         q2_model(obs=obs1, act=pi_obs1))
            v_main_obs1 = v_main_model(obs1)
            v_targ_obs2 = v_targ_model(obs2)

            # Equation 5 in the paper
            # Note that although we use states from the replay buffer,
            # we sample actions from the current policy.
            assert min_q12_obs1_pi.shape.as_list() == [None, 1]
            assert log_prob_pi_obs1.shape.as_list() == [None, 1]
            assert v_main_obs1.shape.as_list() == [None, 1]
            v_backup = min_q12_obs1_pi - temperature * log_prob_pi_obs1
            v_backup = tf.stop_gradient(v_backup)
            v_loss = (v_main_obs1 - v_backup) ** 2
            v_loss = tf.reduce_mean(v_loss)

            # Equations 8/7 in the paper
            assert rews.shape.as_list() == [None, 1]
            assert done.shape.as_list() == [None, 1]
            assert v_targ_obs2.shape.as_list() == [None, 1]
            q_backup = rews + discount * (1 - done) * v_targ_obs2
            q_backup = tf.stop_gradient(q_backup)
            q1_loss = (q1_obs1_acts - q_backup) ** 2
            q2_loss = (q2_obs1_acts - q_backup) ** 2
            q1_loss = tf.reduce_mean(q1_loss)
            q2_loss = tf.reduce_mean(q2_loss)

            # Equation 12 in the paper
            # Again, note that we don't use actions sampled from the replay buffer.
            assert min_q12_obs1_pi.shape.as_list() == [None, 1]
            assert log_prob_pi_obs1.shape.as_list() == [None, 1]
            pi_loss = -(min_q12_obs1_pi - temperature * log_prob_pi_obs1)
            pi_loss = tf.reduce_mean(pi_loss)

            # The paper isn't explicit about how many optimizers are used,
            # but this is what Spinning Up does.
            pi_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(pi_loss, var_list=policy_model.weights)

            v_q_loss = v_loss + q1_loss + q2_loss
            v_q_weights = q1_model.weights + q2_model.weights + v_main_model.weights
            v_q_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(v_q_loss, var_list=v_q_weights)

            self.train_ops = tf.group([pi_train, v_q_train])

            v_main_params = v_main_model.weights
            v_targ_params = v_targ_model.weights
            assert len(v_main_params) == len(v_targ_params)
            v_targ_polyak_update_ops = []
            for var_main, var_targ in zip(v_main_params, v_targ_params):
                update_op = var_targ.assign(polyak_coef * var_targ + (1 - polyak_coef) * var_main)
                v_targ_polyak_update_ops.append(update_op)
            v_targ_polyak_update_op = tf.group(v_targ_polyak_update_ops)

            restore_ops, restore_phs = self._get_restore_ops()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            self.v_main_obs1 = v_main_model(obs1)
            self.v_targ_obs1 = v_targ_model(obs1)

        self.obs1 = obs1
        self.obs2 = obs2
        self.acts = acts
        self.rews = rews
        self.done = done
        self.pi_obs1 = pi_obs1
        self.mu_obs1 = mean_obs1
        self.sess = sess
        self.obs_dim = obs_dim
        self.v_targ_polyak_update_op = v_targ_polyak_update_op
        self.q_loss = q1_loss + q2_loss

        self.restore_phs = restore_phs
        self.restore_ops = restore_ops

    def step(self, obs, deterministic):
        assert obs.shape == (self.obs_dim,)
        if deterministic:
            action_op = self.mu_obs1
        else:
            action_op = self.pi_obs1
        action = self.sess.run(action_op, feed_dict={self.obs1: [obs]})[0]
        return action

    def train(self, batch: ReplayBatch):
        if not hasattr(self, 'logger'):
            self.logger = easy_tf_log.Logger(os.path.join(self.save_dir, 'train_log'))
        with LogMilliseconds('train sess.run', self.logger):
            _, _, loss = self.sess.run([self.train_ops, self.v_targ_polyak_update_op, self.q_loss],
                                       feed_dict={self.obs1: batch.obs1,
                                                  self.acts: batch.acts,
                                                  self.rews: batch.rews,
                                                  self.obs2: batch.obs2,
                                                  self.done: batch.done})
        self.logger.logkv('n_ops', len(self.graph.get_operations()))
        return loss

    @staticmethod
    def _args_from_locals(locals_dict):
        locals_dict = dict(locals_dict)
        del locals_dict['self']
        locals_dict['save_dir'] = None
        return locals_dict

    @staticmethod
    def _get_restore_ops():
        restore_ops, restore_phs = [], {}
        for param in tf.trainable_variables():
            ph = tf.placeholder(tf.float32, param.shape)
            op = param.assign(ph)
            restore_ops.append(op)
            restore_phs[param.name] = ph
        return restore_ops, restore_phs

    def save(self):
        max_n_checkpoints = 2
        weights = glob(os.path.join(self.save_dir, 'weights-*.pkl'))
        models = glob(os.path.join(self.save_dir, 'model-*.pkl'))
        assert len(weights) == len(models)
        all_ckpts = weights + models
        all_ckpts.sort(key=lambda p: os.path.getmtime(p))
        while len(all_ckpts) // 2 >= max_n_checkpoints:
            for ckpt in all_ckpts[:2]:
                os.remove(ckpt)
            all_ckpts = all_ckpts[2:]

        save_id = int(time.time())
        with atomic_write(os.path.join(self.save_dir, f'model-{save_id}.pkl'), binary=True) as f:
            pickle.dump(self, f)
        with atomic_write(os.path.join(self.save_dir, f'weights-{save_id}.pkl'), binary=True) as f:
            pickle.dump(self._get_params(), f)

    def load(self, load_dir):
        ckpts = glob(os.path.join(load_dir, 'weights-*.pkl'))
        ckpts.sort(key=lambda p: os.path.getmtime(p))
        latest_ckpt = ckpts[-1]
        with open(latest_ckpt, 'rb') as f:
            params = pickle.load(f)
        self.restore_params(params)

    def restore_params(self, params):
        feed_dict = {self.restore_phs[param_name]: param_value
                     for param_name, param_value in params.items()}
        self.sess.run(self.restore_ops, feed_dict=feed_dict)

    def _get_params(self):
        with self.graph.as_default():
            params = tf.trainable_variables()
        names = [p.name for p in params]
        values = self.sess.run(params)
        params = {k: v for k, v in zip(names, values)}
        return params

    def __getstate__(self):
        params = self._get_params()
        state = self.args_copy, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        self.restore_params(params)
