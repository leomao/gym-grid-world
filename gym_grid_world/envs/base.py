import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from PIL import Image, ImageDraw

class BaseEnv(gym.Env):
    '''
    Abstract class for visual environments
    '''
    metatdata = {'render.modes': ['human']}

    def __init__(self):
        self._seed()
        self.__is_configured = False

    def _configure(self, actions, frame_size, *, max_step=-1):
        '''
        Usage:
        self.super()._configure(actions, frame_size)
        '''
        self.frame_size = frame_size

        self.image = Image.new('RGB', self.frame_size, 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.max_step = max_step
        self.step_cnt = 0

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(0., 255., (*self.frame_size, 3))
        self.__is_configured = True

    def init(self):
        self._init()
        if not self.__is_configured:
            raise NotImplementedError

    # should be implemented
    def _init(self):
        raise NotImplementedError

    def _step_env(self, action):
        raise NotImplementedError

    def _render_env(self):
        raise NotImplementedError

    # gym.Env functions
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.step_cnt = 0
        self.init()
        self._render_env()
        return self.get_bitmap()

    def _step(self, action):
        print(action, int(action))
        action = int(action)
        assert 0 <= action < self.action_space.n
        
        rew, done = self._step_env(action)
        self.step_cnt += 1
        print(self.step_cnt)
        if self.max_step > 0 and self.step_cnt > self.max_step:
            done = True
        self._render_env()
        obs = self.get_bitmap()
        info = None

        return obs, rew, done, info

    def _render(self, mode='human', close=False):
        pass

    # utils functions
    def get_bitmap(self):
        arr = np.array(self.image.getdata()).reshape((*self.frame_size, 3))
        return arr.astype('float32')
