import random

import numpy as np


class ReplayBatch:
    def __init__(self, obs1, acts, rews, obs2, done):
        self.n = len(obs1)
        self.obs1 = obs1
        self.acts = acts
        self.rews = rews
        self.obs2 = obs2
        self.done = done

    def __len__(self):
        return self.n


class ReplayBuffer:
    def __init__(self, obs_shape, act_shape, max_size):
        obs_shape, act_shape = list(obs_shape), list(act_shape)
        self.obs1_buf = np.zeros([max_size] + obs_shape)
        self.obs2_buf = np.zeros([max_size] + obs_shape)
        self.acts_buf = np.zeros([max_size] + act_shape)
        self.rews_buf = np.zeros([max_size, 1])
        self.done_buf = np.zeros([max_size, 1])
        self.idx = 0
        self.len = 0
        self.max_size = max_size

    def __len__(self):
        return self.len

    def store(self, obs1, acts, rews, obs2, done):
        self.obs1_buf[self.idx] = obs1
        self.acts_buf[self.idx] = acts
        self.rews_buf[self.idx] = rews
        self.obs2_buf[self.idx] = obs2
        self.done_buf[self.idx] = done

        self.idx = (self.idx + 1) % self.max_size
        if self.len < self.max_size:
            self.len += 1

    def sample(self, batch_size) -> ReplayBatch:
        idxs = random.sample(range(self.len), batch_size)
        return ReplayBatch(
            obs1=[self.obs1_buf[i] for i in idxs],
            acts=[self.acts_buf[i] for i in idxs],
            rews=[self.rews_buf[i] for i in idxs],
            obs2=[self.obs2_buf[i] for i in idxs],
            done=[self.done_buf[i] for i in idxs]
        )
