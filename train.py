import multiprocessing
import os

import easy_tf_log
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

import config
from env import make_env
from model import SACModel
from replay_buffer import ReplayBuffer
from utils import tf_disable_warnings, tf_disable_deprecation_warnings, RateMeasure

tf_disable_warnings()
tf_disable_deprecation_warnings()

ex = Experiment('sac')
observer = FileStorageObserver.create('runs')
ex.observers.append(observer)
ex.add_config(config.default_config)


@ex.capture
def train_sac(buffer: ReplayBuffer, model: SACModel, train_env,
              batch_size, n_start_env_steps, log_every_n_steps, checkpoint_every_n_steps,
              train_n_steps):
    n_steps = 0
    obs1, done = train_env.reset(), False
    logger = easy_tf_log.Logger(os.path.join(observer.dir, 'sac'))
    step_rate_measure = RateMeasure(n_steps)
    losses = []

    while n_steps < train_n_steps:
        if len(buffer) < n_start_env_steps:
            act = train_env.action_space.sample()
        else:
            act = model.step(obs1, deterministic=False)

        obs2, reward, done, _ = train_env.step(act)
        buffer.store(obs1=obs1, acts=act, rews=reward, obs2=obs2, done=float(done))
        obs1 = obs2
        if done:
            obs1, done = train_env.reset(), False

        if len(buffer) < n_start_env_steps:
            continue

        batch = buffer.sample(batch_size=batch_size)
        loss = model.train(batch)
        losses.append(loss)

        if n_steps % checkpoint_every_n_steps == 0:
            model.save()

        if n_steps % log_every_n_steps == 0:
            print(f"Trained {n_steps} steps")
            n_steps_per_second = step_rate_measure.measure(n_steps)
            logger.logkv('dqn/buffer_size', len(buffer))
            logger.logkv('dqn/n_steps', n_steps)
            logger.logkv('dqn/n_steps_per_second', n_steps_per_second)
            logger.logkv('dqn/loss', np.mean(losses))
            losses = []

        n_steps += 1


def run_test_env(model, model_load_dir, render, env_id, seed, log_dir):
    env = make_env(env_id, seed, log_dir, 'test')
    while True:
        model.load(model_load_dir)
        obs, done, rewards = env.reset(), False, []
        while not done:
            action = model.step(obs, deterministic=True)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)


@ex.automain
def main(gamma, buffer_size, lr, render, seed, env_id, polyak_coef, temperature, policy_std_min, policy_std_max):
    env = make_env(env_id, seed, observer.dir, 'train')

    buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape, max_size=buffer_size)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    act_lim = env.action_space.high
    ckpt_dir = os.path.join(observer.dir, 'checkpoints')
    model = SACModel(obs_dim=obs_dim, n_actions=n_actions, act_lim=act_lim, save_dir=ckpt_dir,
                     discount=gamma, lr=lr, seed=seed, polyak_coef=polyak_coef, temperature=temperature,
                     std_min_max=(policy_std_min, policy_std_max))
    model.save()

    ctx = multiprocessing.get_context('spawn')
    test_env_proc = ctx.Process(target=run_test_env, daemon=True,
                                args=(model, ckpt_dir, render, env_id, seed, observer.dir))
    test_env_proc.start()

    train_sac(buffer, model, env)

    test_env_proc.terminate()
    return model
