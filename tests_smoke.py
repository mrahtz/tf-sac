import unittest

import train


class SmokeTests(unittest.TestCase):
    @staticmethod
    def test():
        train.ex.run(config_updates={'train_n_steps': 10})
