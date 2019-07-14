import os
import time
from collections import defaultdict

import easy_tf_log
from tensorflow.python.util import deprecation


def tf_disable_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tf_disable_deprecation_warnings():
    deprecation._PRINT_DEPRECATION_WARNINGS = False


class RateMeasure:
    def __init__(self, val):
        self.prev_t = self.prev_value = None
        self.reset(val)

    def reset(self, val):
        self.prev_value = val
        self.prev_t = time.time()

    def measure(self, val):
        val_change = val - self.prev_value
        cur_t = time.time()
        interval = cur_t - self.prev_t
        rate = val_change / interval

        self.prev_t = cur_t
        self.prev_value = val

        return rate


class LogMilliseconds:
    _values = defaultdict(list)

    def __init__(self, name, logger: easy_tf_log.Logger, log_every=1000):
        self.name = name
        self.logger = logger
        self.log_every = log_every

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, type, value, traceback):
        t_end = time.time()
        duration_seconds = t_end - self.t_start
        duration_ms = duration_seconds * 1000
        if self.log_every == 1:
            self.logger.logkv(self.name, duration_ms)
        else:
            LogMilliseconds._values[self.name].append(duration_ms)
            if len(LogMilliseconds._values[self.name]) == self.log_every:
                self.logger.log_list_stats(self.name, self._values[self.name])
                LogMilliseconds._values[self.name] = []
