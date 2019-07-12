import os
import time

import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Concatenate, Lambda

from policies import get_diagonal_gaussian_model
from replay_buffer import ReplayBatch
from utils import get_mlp_model


def get_q_model(obs_dim, n_actions):
    obs = Input([obs_dim])
    act = Input([n_actions])
    obs_act = Concatenate(axis=-1)([obs, act])
    assert obs_act.shape.as_list() == [None, obs_dim + n_actions]
    q = get_mlp_model(n_inputs=(obs_dim + n_actions), n_outputs=1)(obs_act)
    assert q.shape.as_list() == [None, 1]
    return Model(inputs=[obs, act], outputs=q)


def get_min_q12_model(obs_dim, n_actions, q1, q2):
    obs = Input([obs_dim])
    act = Input([n_actions])
    q12 = Concatenate(axis=1)([q1([obs, act]), q2([obs, act])])
    assert q12.shape.as_list() == [None, 2]
    min_q12 = Lambda(lambda x: tf.reduce_min(x, axis=-1, keepdims=True))(q12)
    assert min_q12.shape.as_list() == [None, 1]
    return Model(inputs=[obs, act], outputs=min_q12)


class SACModel:

    def __init__(self, obs_dim, n_actions, act_lim, seed, discount, temperature, polyak_coef, lr, save_dir=None):
        self.args_copy = self.args_from_locals(locals())
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

            policy = get_diagonal_gaussian_model(obs_dim=obs_dim, n_actions=n_actions, act_lim=act_lim)
            mu = policy.mean
            pi = policy.pi
            log_prob_pi = policy.log_prob_pi

            q1 = get_q_model(obs_dim, n_actions)
            q2 = get_q_model(obs_dim, n_actions)
            min_q12 = get_min_q12_model(obs_dim, n_actions, q1, q2)

            v_main = get_mlp_model(obs_dim, n_outputs=1)
            v_targ = get_mlp_model(obs_dim, n_outputs=1)

            # Equation 5 in the paper
            # Note that although we use states from the replay buffer,
            # we sample actions from the current policy.
            assert min_q12.output_shape == (None, 1)
            assert log_prob_pi.output_shape == (None, 1)
            assert v_main.output_shape == (None, 1)
            v_backup = min_q12([obs1, pi(obs1)]) - temperature * log_prob_pi(obs1)
            v_backup = tf.stop_gradient(v_backup)
            v_loss = (v_main(obs1) - v_backup) ** 2
            v_loss = tf.reduce_mean(v_loss)

            # Equations 8/7 in the paper
            assert rews.shape.as_list() == [None, 1]
            assert done.shape.as_list() == [None, 1]
            assert v_targ.output_shape == (None, 1)
            q_backup = rews + discount * (1 - done) * v_targ(obs2)
            q_backup = tf.stop_gradient(q_backup)
            q1_loss = (q1([obs1, acts]) - q_backup) ** 2
            q2_loss = (q2([obs2, acts]) - q_backup) ** 2
            q1_loss = tf.reduce_mean(q1_loss)
            q2_loss = tf.reduce_mean(q2_loss)

            # Equation 12 in the paper
            # Again, note that we don't use actions sampled from the replay buffer.
            assert min_q12.output_shape == (None, 1)
            assert log_prob_pi.output_shape == (None, 1)
            pi_loss = -(min_q12([obs1, pi(obs1)]) - temperature * log_prob_pi(obs1))
            pi_loss = tf.reduce_mean(pi_loss)

            # The paper isn't explicit about how many optimizers are used,
            # but this is what Spinning Up does.
            pi_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(pi_loss, var_list=pi.weights)
            v_q_loss = v_loss + q1_loss + q2_loss
            v_q_weights = q1.weights + q2.weights + v_main.weights
            v_q_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(v_q_loss, var_list=v_q_weights)
            self.train_ops = tf.group([pi_train, v_q_train])

            v_main_params = v_main.weights
            v_targ_params = v_targ.weights
            assert len(v_main_params) == len(v_targ_params)
            v_targ_polyak_update_ops = []
            for i in range(len(v_main_params)):
                var_main = v_main_params[i]
                var_targ = v_targ_params[i]
                update_op = var_targ.assign(polyak_coef * var_targ + (1 - polyak_coef) * var_main)
                v_targ_polyak_update_ops.append(update_op)
            v_targ_polyak_update_op = tf.group(v_targ_polyak_update_ops)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            self.v_main_obs1 = v_main(obs1)
            self.v_targ_obs1 = v_targ(obs1)
            self.pi_obs1 = pi(obs1)
            self.mu_obs1 = mu(obs1)

        self.obs1 = obs1
        self.obs2 = obs2
        self.acts = acts
        self.rews = rews
        self.done = done
        self.sess = sess
        self.saver = saver
        self.obs_dim = obs_dim
        self.v_targ_polyak_update_op = v_targ_polyak_update_op
        self.pi = pi
        self.q_loss = q1_loss + q2_loss

    @staticmethod
    def args_from_locals(locals_dict):
        locals_dict = dict(locals_dict)
        del locals_dict['self']
        locals_dict['save_dir'] = None
        return locals_dict

    def step(self, obs, deterministic):
        assert obs.shape == (self.obs_dim,)
        if deterministic:
            action_op = self.mu_obs1
        else:
            action_op = self.pi_obs1
        action = self.sess.run(action_op, feed_dict={self.obs1: [obs]})[0]
        return action

    def train(self, batch: ReplayBatch):
        _, _, loss = self.sess.run([self.train_ops, self.v_targ_polyak_update_op, self.q_loss],
                                   feed_dict={self.obs1: batch.obs1,
                                              self.acts: batch.acts,
                                              self.rews: batch.rews,
                                              self.obs2: batch.obs2,
                                              self.done: batch.done})
        return loss

    def save(self):
        save_id = int(time.time())
        self.saver.save(self.sess, os.path.join(self.save_dir, 'model'), save_id)

    def load(self, load_dir):
        ckpt = tf.train.latest_checkpoint(load_dir)
        self.saver.restore(self.sess, ckpt)

    def __getstate__(self):
        with self.graph.as_default():
            params = tf.trainable_variables()
        names = [p.name for p in params]
        values = self.sess.run(params)
        params = {k: v for k, v in zip(names, values)}
        state = self.args_copy, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        with self.graph.as_default():
            ops = []
            for param in tf.trainable_variables():
                assign = param.assign(params[param.name])
                ops.append(assign)
            self.sess.run(ops)
