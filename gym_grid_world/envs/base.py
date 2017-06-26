import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

from PIL import Image, ImageDraw
import pyglet

class BaseEnv(gym.Env):
    '''
    Abstract class for visual environments rendered by PIL
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._seed()
        self.__configured = False

        self.viewer = None

    def configure(self, actions, frame_size, *, max_step=-1):
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
        action = int(action)
        assert 0 <= action < self.action_space.n

        rew, done = self._step_env(action)
        self.step_cnt += 1
        if self.max_step > 0 and self.step_cnt > self.max_step:
            done = True
        self._render_env()
        obs = self.get_bitmap()
        info = None

        return obs, rew, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = TKViewer(100, 100)

        self.viewer.add_image(50, 50, self.image.resize((100, 100)))

    # utils functions
    def get_bitmap(self):
        arr = np.array(self.image).reshape((*self.frame_size, 3))
        return arr.astype('float32')

import tkinter
import threading
from PIL import Image, ImageTk

class TKViewer(threading.Thread):

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.root = None
        self.start()

    def run(self):
        self.root = tkinter.Tk()
        self.canvas = tkinter.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()
        self.root.mainloop()

    def close(self):
        self.root.destroy()
        self.root = None

    def add_image(self, x, y, image):
        if self.root is None:
            return
        try:
            self.current_image = ImageTk.PhotoImage(image)
            self.canvas.create_image(x, y, image=self.current_image)
        except:
            pass

def numpy_to_PIL(rgb_array):
    return Image.fromarray(rgb_array.astype(np.uint8), 'RGB')

def numpy_to_pyglet(rgb_array):
    width, height = rgb_array.shape[:2]
    data = rgb_array.flatten()
    return pyglet.image.ImageData(
        width, height, 'RGB',
        (pyglet.gl.GLubyte * data.size)(*data.astype('uint8')))

class PygletImage(rendering.Geom):
    def __init__(self, img):
        rendering.Geom.__init__(self)
        self.img = img
        self.width = img.width
        self.height = img.height
        self.flip = False
    def render(self):
        x, y = 0, 0
        w, h = self.width, self.height
        for attr in self.attrs:
            if type(attr) is rendering.Transform:
                x += attr.translation[0]
                y += attr.translation[1]
                w *= attr.scale[0]
                h *= attr.scale[1]
        self.img.blit(-w/2+x, -h/2+y, width=w, height=h)
