from gym.envs.registration import register

register(
    id='pickput2d-v0',
    entry_point='gym_grid_world.envs:PickputEnv',
)

register(
    id='eatbullet2d-v0',
    entry_point='gym_grid_world.envs:EatBulletEnv',
)

register(
    id='pushblock2d-v0',
    entry_point='gym_grid_world.envs:PushBlockEnv',
)

register(
    id='eatbulletmem2d-v0',
    entry_point='gym_grid_world.envs:EatBulletMemEnv',
)

register(
    id='eatbulletpair2d-v0',
    entry_point='gym_grid_world.envs:EatBulletPairEnv',
)
