import os
import sys

import easy_tf_log
import gym
from gym.core import Wrapper


class LogRewards(Wrapper):
    def __init__(self, env, logger, test_or_train):
        super().__init__(env)
        self.logger = logger
        self.test_or_train = test_or_train
        self.episode_reward = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.episode_reward = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        if done:
            if self.test_or_train == 'test':
                print(f"{self.test_or_train.capitalize()} episode done; reward {self.episode_reward}")
                sys.stdout.flush()
            self.logger.logkv(f'env_{self.test_or_train}/episode_reward', self.episode_reward)
        return obs, reward, done, info


def make_env(env_id, seed, log_dir, test_or_train):
    env = gym.make(env_id)
    env.seed(seed)
    logger = easy_tf_log.Logger(os.path.join(log_dir, f'env_{test_or_train}'))
    env = LogRewards(env, logger, test_or_train)
    return env
