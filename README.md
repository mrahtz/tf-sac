[![Build Status](https://travis-ci.com/mrahtz/tf-sac.svg?branch=master)](https://travis-ci.com/mrahtz/tf-sac)

## Requirements

Install MuJoco separately.

****
Lessons learned:
* Gaussian likelihood
* Keras really nice for making equations look good
* Check shapes
* Tanh - really easy to mess up (e.g. do you tanh the mean, too?)
* Tanh maxes out surprisingly soon; be careful about atanh
* Don't forget to reduce_mean losses
* Don't forget to limit which vars you optimize for
* stop_backup
* Nice Pycharm shortcuts

## Tests

To run tests:
```bash
$ python -m unittest discover
```