import multiprocessing
import os

import easy_tf_log
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

import sac.config
from sac.env import make_env
from sac.model import SACModel
from sac.replay_buffer import ReplayBuffer
from sac.utils import tf_disable_warnings, tf_disable_deprecation_warnings, RateMeasure, LogMilliseconds

tf_disable_warnings()
tf_disable_deprecation_warnings()

ex = Experiment('sac')
observer = FileStorageObserver.create('runs')
ex.observers.append(observer)
ex.add_config(sac.config.default_config)


@ex.capture
def train_sac(buffer: ReplayBuffer, model: SACModel, train_env, test_env, n_start_env_steps, log_every_n_steps,
              checkpoint_every_n_steps, train_n_steps, test_every_n_steps, batch_size):
    obs1, done = train_env.reset(), False
    logger = easy_tf_log.Logger(os.path.join(observer.dir, 'sac'))
    n_steps = 0
    step_rate_measure = RateMeasure(n_steps)
    losses = []

    while n_steps < train_n_steps:
        if len(buffer) < n_start_env_steps:
            act = train_env.action_space.sample()
        else:
            with LogMilliseconds('model.step', logger):
                act = model.step(obs1, deterministic=False)

        with LogMilliseconds('train_env.step', logger):
            obs2, reward, done, _ = train_env.step(act)
        with LogMilliseconds('buffer.store', logger):
            buffer.store(obs1=obs1, acts=act, rews=reward, obs2=obs2, done=float(done))
        obs1 = obs2
        if done:
            obs1, done = train_env.reset(), False

        if len(buffer) < n_start_env_steps:
            continue

        with LogMilliseconds('buffer.sample', logger):
            batch = buffer.sample(batch_size=batch_size)
        with LogMilliseconds('model.train', logger):
            loss = model.train(batch)
        losses.append(loss)

        if n_steps % log_every_n_steps == 0:
            print(f"Ran {n_steps} steps")
            n_steps_per_second = step_rate_measure.measure(n_steps)
            logger.logkv('sac/buffer_size', len(buffer))
            logger.logkv('sac/n_steps', n_steps)
            logger.logkv('sac/n_steps_per_second', n_steps_per_second)
            logger.logkv('sac/loss', np.mean(losses))
            losses = []

        if n_steps % test_every_n_steps == 0:
            with LogMilliseconds('run_test_episodes', logger):
                run_test_episodes(model, test_env, n_episodes=10, render=False)

        if n_steps % checkpoint_every_n_steps == 0:
            with LogMilliseconds('model.save', logger):
                model.save()

        n_steps += 1


def run_test_episodes(model, env, n_episodes, render):
    for _ in range(n_episodes):
        obs, done, = env.reset(), False
        while not done:
            action = model.step(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if render:
                env.render()


def async_test_episodes_loop(model, model_load_dir, env_id, seed, log_dir):
    env = make_env(env_id, seed, log_dir, 'test_async')
    while True:
        model.load(model_load_dir)
        run_test_episodes(model, env, n_episodes=1, render=True)


@ex.automain
def main(gamma, buffer_size, lr, seed, env_id, polyak_coef, temperature, policy_std_min, policy_std_max,
         async_test, network):
    train_env = make_env(env_id, seed, observer.dir, 'train')
    test_env = make_env(env_id, seed, observer.dir, 'test')

    buffer = ReplayBuffer(train_env.observation_space.shape, train_env.action_space.shape, max_size=buffer_size)
    obs_dim = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.shape[0]
    act_lim = train_env.action_space.high
    ckpt_dir = os.path.join(observer.dir, 'checkpoints')
    os.makedirs(ckpt_dir)
    model = SACModel(obs_dim=obs_dim, n_actions=n_actions, act_lim=act_lim, save_dir=ckpt_dir,
                     discount=gamma, lr=lr, seed=seed, polyak_coef=polyak_coef, temperature=temperature,
                     std_min_max=(policy_std_min, policy_std_max), network=network)
    model.save()

    if async_test:
        ctx = multiprocessing.get_context('spawn')
        test_env_proc = ctx.Process(target=async_test_episodes_loop, daemon=True,
                                    args=(model, ckpt_dir, env_id, seed, observer.dir))
        test_env_proc.start()
    else:
        test_env_proc = None

    train_sac(buffer, model, train_env, test_env)

    if test_env_proc is not None:
        test_env_proc.terminate()
    return model
