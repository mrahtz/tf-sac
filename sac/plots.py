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
from matplotlib.pyplot import close, title, fill_between
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

    xs_list = []
    ys_list = []
    for events in events_by_seed.values():
        xs, ys = interpolate_to_common_xs(events['env_test/episode_reward'], events['dqn/n_steps'])
        xs_list.append(xs)
        ys_list.append(ys)

    plot_averaged(xs_list, ys_list, env_id)

    escaped_env_name = escape_env_name(env_id)
    fig_filename = '{}.png'.format(escaped_env_name)
    savefig(fig_filename, dpi=300, bbox_inches='tight')

    close('all')


def interpolate_to_common_xs(timestamp_y_tuples, timestamp_x_tuples):
    x_timestamps, xs = zip(*timestamp_x_tuples)
    y_timestamps, ys = zip(*timestamp_y_tuples)

    if len(timestamp_x_tuples) < len(timestamp_y_tuples):
        # Use x timestamps for interpolation
        xs = xs
        ys = interpolate_values(timestamp_y_tuples, x_timestamps)
    elif len(timestamp_y_tuples) < len(timestamp_x_tuples):
        # Use y timestamps for interpolation
        xs = interpolate_values(timestamp_x_tuples, y_timestamps)
        ys = ys

    return xs, ys


def interpolate_values(x_y_tuples, new_xs):
    xs, ys = zip(*x_y_tuples)
    if new_xs[-1] < xs[0]:
        raise Exception("New x values end before old x values begin")
    if new_xs[0] > xs[-1]:
        raise Exception("New x values start after old x values end")

    new_ys = np.interp(new_xs, xs, ys, left=np.nan, right=np.nan)  # use NaN if we don't have data
    assert np.nan not in new_ys
    return new_ys


def plot_averaged(xs_list, ys_list, env_id):
    # Interpolate all data to have common x values
    all_xs = set([x for xs in xs_list for x in xs])
    all_xs = sorted(list(all_xs))
    for n in range(len(xs_list)):
        ys_list[n] = interpolate_values(x_y_tuples=list(zip(xs_list[n], ys_list[n])),
                                        new_xs=all_xs)

    mean_ys = np.mean(ys_list, axis=0)  # Average across seeds
    smoothed_mean_ys = moving_average(mean_ys, window_size=100)
    plot(all_xs, smoothed_mean_ys, alpha=0.9)

    std = np.std(ys_list, axis=0)
    smoothed_std = np.array(moving_average(std, window_size=100))
    lower = smoothed_mean_ys - smoothed_std
    upper = smoothed_mean_ys + smoothed_std
    fill_between(all_xs, lower, upper, alpha=0.2)

    grid(True)
    for axis in ['x', 'y']:
        ticklabel_format(axis=axis, style='scientific', scilimits=(0, 5), useLocale=True)
    xlabel('Steps')
    ylabel('Episode reward')
    xlim(left=0)
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
