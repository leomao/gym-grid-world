import time
import numpy as np
import gym
import gym_grid_world
from gym_grid_world.envs.pickput import TaskType

env = gym.make('pickput2d-v0')

env.configure(view_name='pickput', task_type=TaskType.both)
# you will need to run env_viewer server separately
env.render()
env.reset()

while True:
    try:
        time.sleep(0.05)
        env.step(np.random.randint(env.action_space.n))
    except KeyboardInterrupt:
        break

env.render(close=True)
