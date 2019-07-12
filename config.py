default_config = dict(
    batch_size=100,
    gamma=0.99,
    temperature=0.2,
    polyak_coef=0.995,
    n_start_env_steps=10_000,
    log_every_n_steps=100,
    checkpoint_every_n_steps=1000,
    buffer_size=int(1e6),
    lr=1e-3,
    render=False,
    env_id='InvertedPendulum-v2',
    train_n_steps=1e5,
    seed=0,
    policy_std_min=1e-4,
    policy_std_max=4
)
