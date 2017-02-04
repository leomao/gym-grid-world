from gym.envs.registration import register

register(
    id='pickput2d-v0',
    entry_point='gym_grid_world.envs:PickputEnv',
    tags={
        'wrapper_config.TimeLimit.max_episode_steps': 200
    },
)

register(
    id='eatbullet2d-v0',
    entry_point='gym_grid_world.envs:EatBulletEnv',
    tags={
        'wrapper_config.TimeLimit.max_episode_steps': 100
    },
)


register(
    id='pushblock2d-v0',
    entry_point='gym_grid_world.envs:PushBlockEnv',
    tags={
        'wrapper_config.TimeLimit.max_episode_steps': 200
    },
)
