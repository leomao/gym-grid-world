import time
import numpy as np
import gym
import gym_grid_world
from gym_grid_world.envs.pickput import TaskType

env0 = gym.make('pushblock2d-v0')
env0.configure(view_name='push', obj_n=2)
# you will need to run env_viewer server separately
env0.render()
env0.reset()

env1 = gym.make('pickput2d-v0')
env1.configure(view_name='pickput', task_type=TaskType.both)
# you will need to run env_viewer server separately
env1.render()
env1.reset()

env2 = gym.make('eatbullet2d-v0')
env2.configure(view_name='bullet', food_n=10)
# you will need to run env_viewer server separately
env2.render()
env2.reset()

cnt = 0
while True:
    cnt += 1
    try:
        time.sleep(0.02)
        _, _, done0, _ = env0.step(np.random.randint(env0.action_space.n))
        _, _, done1, _ = env1.step(np.random.randint(env1.action_space.n))
        _, _, done2, _ = env2.step(np.random.randint(env2.action_space.n))
        print(done0, done1, done2, cnt)
        if done0 and done1 and done2:
            break
    except KeyboardInterrupt:
        break

env0.close()
env1.close()
env2.close()
