import time
import numpy as np
import gym
import gym_grid_world
from gym_grid_world.envs.pickput import TaskType

env = gym.make('eatbullet2d-v0')
env.configure(center=True, view_radius=(5, 5))

act_map = { 81: 3, 82: 1, 83: 4, 84: 2 }

cnt = 0
total_rew = 0
obs = env.reset()

avg = 0

while True:
    try:
        act = env.action_space.sample()
        st = time.time()
        obs, rew, done, _ = env.step(act)
        delta = time.time() - st
        total_rew += rew
        print(total_rew, cnt, done, delta)
        avg += delta
        cnt += 1

        if done:
            break
    except KeyboardInterrupt:
        break

avg /= cnt
print(avg)
