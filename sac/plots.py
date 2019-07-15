#!/usr/bin/env python3

import argparse
import fnmatch
import glob
import json
import locale
import os
from collections import defaultdict
from functools import partial

import numpy as np
import tensorflow as tf
from matplotlib.pyplot import close, title, fill_between, figure
from pylab import plot, xlabel, ylabel, savefig, grid, xlim, ticklabel_format

from sac.utils import tf_disable_warnings, tf_disable_deprecation_warnings

tf_disable_warnings()
tf_disable_deprecation_warnings()
# Get thousands separated by commas
locale.format_string = partial(locale.format_string, grouping=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs_dir', nargs='?')
    args = parser.parse_args()

    for f in glob.glob('*.png'):
        os.remove(f)

    print(f"Loading runs...")
    events_by_env_name_by_seed = defaultdict(dict)
    for run_dir in [path for path in os.scandir(args.runs_dir) if os.path.isdir(path.path)]:
        events = read_all_events(run_dir.path)
        make_timestamps_relative_hours(events)
        with open(os.path.join(run_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        events_by_env_name_by_seed[config['env_id']][config['seed']] = events

    for env_name, events_by_seed in events_by_env_name_by_seed.items():
        plot_env(env_name, events_by_seed)


def read_all_events(directory):
    events_files = find_files_matching_pattern('events.out.tfevents*', directory)
    events = defaultdict(list)
    for events_file in events_files:
        for event in tf.train.summary_iterator(events_file):
            for value in event.summary.value:
                events[value.tag].append((event.wall_time, value.simple_value))
    return events


def find_files_matching_pattern(pattern, path):
    result = []
    for root, dirs, files in os.walk(path, followlinks=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def make_timestamps_relative_hours(events):
    for timestamp_value_tuples in events.values():
        first_timestamp = timestamp_value_tuples[0][0]
        for n, (timestamp, value) in enumerate(timestamp_value_tuples):
            timestamp_value_tuples[n] = ((timestamp - first_timestamp) / 3600, value)


def plot_env(env_id, events_by_seed):
    print(f"Plotting {env_id}...")

    figure(figsize=(4, 3))

    xs_list = []
    ys_list = []
    for events in events_by_seed.values():
        xs, ys = get_rewards_by_step(events)
        xs_list.append(xs)
        ys_list.append(ys)

    for n in range(1, len(xs_list)):
        np.testing.assert_array_equal(xs_list[n], xs_list[0])
    xs = xs_list[0]

    plot_averaged(xs, ys_list, env_id)

    escaped_env_name = escape_env_name(env_id)
    fig_filename = '{}.png'.format(escaped_env_name)
    savefig(fig_filename, dpi=300, bbox_inches='tight')

    close('all')


def get_rewards_by_step(events):
    step_timestamps, steps = zip(*events['sac/n_steps'])
    step_timestamps = np.array(step_timestamps)
    rewards_by_step = []
    for timestamp, reward in events['env_test/episode_reward']:
        step_idx = np.argwhere(step_timestamps > timestamp)[0, 0] - 1
        step = steps[step_idx]
        # Hack: round to nearest thousand
        step = int(np.round(step / 1000) * 1000)
        rewards_by_step.append((step, reward))
    return zip(*rewards_by_step)


def plot_averaged(xs, ys_list, env_id):
    mean_ys = np.mean(ys_list, axis=0)  # Average across seeds
    smoothed_mean_ys = moving_average(mean_ys, window_size=1)
    plot(xs, smoothed_mean_ys, alpha=0.9)

    std = np.std(ys_list, axis=0)
    smoothed_std = np.array(moving_average(std, window_size=1))
    lower = smoothed_mean_ys - smoothed_std
    upper = smoothed_mean_ys + smoothed_std
    fill_between(xs, lower, upper, alpha=0.2)

    grid(True)
    for axis in ['x', 'y']:
        ticklabel_format(axis=axis, style='scientific', scilimits=(0, 5), useLocale=True)
    xlabel('Steps')
    ylabel('Episode reward')
    xlim([0, 1e6])
    title(env_id)


def moving_average(values, window_size):
    window = np.ones(window_size)
    actual_window_sizes = np.convolve(np.ones(len(values)), window, 'same')
    smoothed_values = np.convolve(values, window, 'same') / actual_window_sizes
    return smoothed_values


def exponential_smoothing(values, smoothing):
    smoothed_values = [values[0]]
    for v in values[1:]:
        smoothed_values.append(smoothing * smoothed_values[-1] + (1 - smoothing) * v)
    return smoothed_values


def escape_env_name(name):
    return name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').lower()


if __name__ == '__main__':
    main()
