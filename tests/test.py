import time
import numpy as np
import gym
import gym_grid_world
from gym_grid_world.envs.pickput import TaskType
import cv2

# env = gym.make('eatbullet2d-v0')
env = gym.make('pickput2d-v0')
env.configure(center=True, view_radius=(5, 5), task_type=TaskType.both,
              block_size=5,
              # raw_array=True,
              )

act_map = { 81: 3, 82: 1, 83: 4, 84: 2 }
# pickput action
act_map.update({ 120: 5, 122: 6})

cnt = 0
total_rew = 0
obs = env.reset()

OUT_SIZE = (400, 400)

def resize(img):
    return cv2.resize(img[:,:,::-1].astype('uint8'), OUT_SIZE,
                      interpolation=cv2.INTER_NEAREST)

def show(obs):
    ss = resize(obs)
    cv2.imshow('obs', ss)

while True:
    cnt += 1
    try:
        show(obs)
        k = cv2.waitKey()

        act = act_map.get(k, 0)

        if np.random.random() < 0.05:
            act = env.action_space.sample()

        obs, rew, done, _ = env.step(act)
        total_rew += rew
        print(total_rew, cnt, done)

        if done:
            show(obs)
            k = cv2.waitKey()
            break
    except KeyboardInterrupt:
        break
