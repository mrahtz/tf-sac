import argparse
import pickle

import gym

parser = argparse.ArgumentParser()
parser.add_argument('env_id')
parser.add_argument('ckpt')
args = parser.parse_args()

with open(args.ckpt, 'rb') as f:
    model = pickle.load(f)

env = gym.make(args.env_id)
while True:
    obs, done = env.reset(), False
    while not done:
        action = model.step(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
