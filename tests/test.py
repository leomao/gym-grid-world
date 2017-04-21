import time
import numpy as np
import gym
import gym_grid_world
from gym_grid_world.envs.pickput import TaskType
# import cv2

env0 = gym.make('pushblock2d-v0')
env0.configure(obj_n=2)
env0.reset()

env1 = gym.make('pickput2d-v0')
env1.configure(task_type=TaskType.both)
env1.reset()

env2 = gym.make('eatbullet2d-v0')
env2.configure(food_n=10, center=True, view_radius=(2, 2))
env2.reset()

cnt = 0
while True:
    cnt += 1
    try:
        time.sleep(0.02)
        _, _, done0, _ = env0.step(env0.action_space.sample())
        _, _, done1, _ = env1.step(env1.action_space.sample())
        _, _, done2, _ = env2.step(env2.action_space.sample())
        print(done0, done1, done2, cnt)
        # ss = cv2.resize(obs[:,:,::-1].astype('uint8'), (400, 400))
        # cv2.imshow('obs', ss)
        # cv2.waitKey(100)

        if done0 and done1 and done2:
            break
    except KeyboardInterrupt:
        break
