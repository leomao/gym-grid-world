from gym.envs.registration import register

register(
    id='pickput2d-v0',
    entry_point='gym_grid_world.envs:PickputEnv',
    timestep_limit=500,
)
