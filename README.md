[![Build Status](https://travis-ci.com/mrahtz/tf-sac.svg?branch=master)](https://travis-ci.com/mrahtz/tf-sac)

# TensorFlow Soft Actor-Critic

TensorFlow implementation of Haarnoja et al.'s [Soft Actor-Critic](https://arxiv.org/abs/1801.01290).

## Usage

### Setup

To set up a virtual environment and install requirements, install [Pipenv](https://github.com/pypa/pipenv) then just do:

```bash
$ pipenv sync
```

If you want to run MuJoCo environments, you'll also need to install `mujoco-py` version 1.50.1.68.

### Training

Basic usage is

```bash
$ pipenv run python -m sac.train
```

We use [Sacred](https://github.com/IDSIA/sacred) for configuration management.
See [config.py](sac/config.py) for available parameters, and set parameters using e.g.:

```bash
$ pipenv run python -m sac.train with env_id=HalfCheetah-v2 render=True
```

A run directory will be created in `runs/` containing TensorBoard metrics.

To view a trained agent:

```bash
$ pipenv run python -m sac.play env_id runs/x/checkpoints/<model-yyy.pkl
```

### Tests

Unit tests cover:
*   Target network update
*   Gaussian policy log prob calculation
*   Action limit

To run tests:

```bash
$ pipenv run python -m unittest discover
```

To run only smoke tests, confirming that `train.py` runs without crashing:

```bash
$ pipenv run python -m tests.tests_smoke
```

## Lessons learned

### Stochastic policies

The main thing which surprised me with soft-actor critic was how hard it was to
implement the stochastic tanh-Gaussian policy correctly. There are a lot of fiddly
details (e.g. do you tanh the mean before or after you sample?) and it's tricky
to figure out how the tanh modifies the PDF (see <https://math.stackexchange.com/a/3283855/468726>).

### tanh precision

tanh reaches the limit of float32 precision surprisingly quickly:

```python
>>> np.tanh(19)
0.99999999999999989
>>> np.tanh(20)
1.0
```

Be really careful if you need to tanh something and then later arctanh it,
or you'll get arithmetic errors.

### Keras

Since `tf.layers.dense` is now deprecated in favour of Keras layers,
this was one of my first projects using Keras in anger.

I love how easy Keras's model paradigm makes it to reuse layers.
For example, to use the same set of policy weights to calculate
an action for two different observations:

```python
pi = PolicyModel()
action1 = pi(obs1)
action2 = pi(obs2)
```

`PolicyModel()` instatiates a set of weights which are then held in `pi`.
The resulting transformation can then be applied by just calling
`pi` on other tensors.

I preferred subclassing `Model` than calling `Model` on a bunch of Keras layers.
You don't have worry about input shape or much around the `Lambda` layers as much. For example:

```python
def get_pi_model(obs_dim):
    obs = Input(shape=(obs_dim,))
    h = Dense(16)(obs)
    act = Dense(1)(h)
    tanh_act = Lambda(lambda x: tf.tanh(x))(act)
    return Model(inputs=obs, outputs=tanh_act)

obs = tf.placeholder(tf.float32, [None, obs_dim])
pi = get_pi_model(obs_dim)
pi(obs)
```

vs.

```python
class Pi(Model):
    def __init__(self):
        super().__init__()
        self.h = Dense(16)
        self.act = Dense(1)

    def call(self, inputs, **kwargs):
        x = self.h(inputs)
        x = self.act(x)
        x = tf.tanh(x)
        return x

obs = tf.placeholder(tf.float32, [None, obs_dim])
pi = Pi()
pi(obs)
```

### Graph mistakes

Take a look at this code fragment.

```python
q_backup = rews + discount * (1 - done) * v_targ_obs2
q_loss = (q - q_backup) ** 2
```

Does it give you the heebie-jeebies? Well it _should_!

<details>
<summary>Why?</summary>
Don't forget to `tf.stop_gradient` your Bellman backups!
</details>

What about this one?

```python
q_loss = (q_obs1 - q_backup) ** 2
train_op = tf.train.AdamOptimizer().minimize(q_loss)
```

<details>
<summary>What's wrong?</summary>
Don't forget to `tf.reduce_mean` your losses!
</details>

Finally - say you're implementing a DDPG-style graph, where the policy loss
is based on the Q output. What about this one?

```python
pi_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(pi_loss)
```

<details>
<summary>Why be nervous?</summary>
Your optimizer will try and modify Q parameters, too! Don't forget to limit optimizers to only the variables you care about!
</details>

### PyCharm debugging templates

This implementation required a lot of inspection of intermediate values in the graph,
but `tf.print` needs so much boilerplate:

```python
with tf.control_dependencies([tf.print(x, summarize=999)]):
    x = tf.identity(x)
```

Amazingly, it turns out PyCharm has a
[templating system that supports substitutions](https://www.jetbrains.com/help/pycharm/using-live-templates.html)!

With an appropriate template
```text
with tf.control_dependencies([tf.print('$SELECTION$', $SELECTION$, summarize=999)]):
    $SELECTION$ = tf.identity($SELECTION$)
```
adding a print op takes only a few seconds:

![](images/templates.gif)
