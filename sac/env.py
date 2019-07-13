import os
import sys

import easy_tf_log
import gym
from gym.core import Wrapper
from gym.wrappers import Monitor


class LogRewards(Wrapper):
    def __init__(self, env, logger, suffix):
        super().__init__(env)
        self.logger = logger
        self.suffix = suffix
        self.episode_reward = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_reward = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        if done:
            if self.suffix == 'test':
                print(f"{self.suffix.capitalize()} episode done; reward {self.episode_reward}")
                sys.stdout.flush()
            self.logger.logkv(f'env_{self.suffix}/episode_reward', self.episode_reward)
        return obs, reward, done, info


def make_env(env_id, seed, log_dir, suffix, record_videos):
    log_dir = os.path.join(log_dir, f'env_{suffix}')

    env = gym.make(env_id)
    env.seed(seed)

    env = Monitor(env, directory=log_dir, video_callable=lambda n: record_videos)
    logger = easy_tf_log.Logger(log_dir)
    env = LogRewards(env, logger, suffix)

    return env
