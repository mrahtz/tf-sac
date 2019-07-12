Lessons learned:
* Gaussian likelihood
* Keras really nice for making equations look good
* Check shapes
* Tanh - really easy to mess up (e.g. do you tanh the mean, too?)
* Tanh maxes out surprisingly soon; be careful about atanh
* Don't forget to reduce_mean losses
* Don't forget to limit which vars you optimize for
* stop_backup