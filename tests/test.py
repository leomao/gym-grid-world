import time
import numpy as np
import gym
import gym_grid_world
from gym_grid_world.envs.pickput import TaskType
import cv2

env = gym.make('eatbullet2d-v0')
env.configure(center=True, view_radius=(5, 5))

act_map = { 81: 3, 82: 1, 83: 4, 84: 2 }

cnt = 0
total_rew = 0
obs = env.reset()

while True:
    cnt += 1
    try:
        ss = cv2.resize(obs[:,:,::-1].astype('uint8'), (400, 400))
        cv2.imshow('obs', ss)
        k = cv2.waitKey()

        act = act_map.get(k, 0)

        if np.random.random() < 0.05:
            act = env.action_space.sample()

        obs, rew, done, _ = env.step(act)
        total_rew += rew
        print(total_rew, cnt, done)

        if done:
            ss = cv2.resize(obs[:,:,::-1].astype('uint8'), (400, 400))
            cv2.imshow('obs', ss)
            k = cv2.waitKey()
            break
    except KeyboardInterrupt:
        break
