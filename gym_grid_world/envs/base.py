import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from PIL import Image, ImageDraw


class BaseEnv(gym.Env):
    '''
    Abstract class for visual environments rendered by PIL
    '''
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.seed()
        self.__configured = False

    def configure(self, actions, frame_size, *, raw_array=False, max_step=-1):
        '''
        Usage:
            self.super()._configure(actions, frame_size)
        '''
        self.frame_size = frame_size
        self.raw_array = raw_array

        self.image = Image.new('RGB', self.frame_size, 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.max_step = max_step
        self.step_cnt = 0

        self.actions = actions
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(0., 255., (*self.frame_size, 3))
        self.__configured = True

    def init(self):
        self._init()
        if not self.__configured:
            raise NotImplementedError

    # should be implemented
    def _init(self):
        raise NotImplementedError

    def _step_env(self, action):
        raise NotImplementedError

    def _render_env(self):
        raise NotImplementedError

    def get_info(self):
        return {}

    def _get_raw_array(self):
        raise NotImplementedError

    # gym.Env functions
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_cnt = 0
        self.init()
        return self.get_obs()

    def step(self, action):
        action = int(action)
        assert 0 <= action < self.action_space.n

        rew, done = self._step_env(action)
        self.step_cnt += 1
        if self.max_step > 0 and self.step_cnt > self.max_step:
            done = True
        obs = self.get_obs()
        info = None

        return obs, rew, done, info

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return self.get_bitmap()

    # utils functions
    def get_obs(self):
        if self.raw_array:
            return self._get_raw_array()
        else:
            self._render_env()
            return self.get_bitmap()

    def get_bitmap(self):
        arr = np.array(self.image).reshape((*self.frame_size, 3))
        return arr.astype('float32')
