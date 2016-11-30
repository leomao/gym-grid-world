import time
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from PIL import Image, ImageDraw, ImageTk
from tkinter import Tk, Canvas, NW, NE

class GridEnv(gym.Env):
    '''
    Abstract class for grid environments
    '''
    metatdata = {'render.modes': ['human']}

    def __init__(self, actions, grid_size, block_size, *,
                 gui_amp=4, FPS=20, max_step=-1):
        '''
        Usage:
        self.super().__init__(actions, grid_size, block_size,
                              gui_amp, FPS)
        '''
        self.grid_size = grid_size
        self.block_size = block_size
        self.FPS = FPS
        self.frame_size = tuple(x * y for x, y in zip(grid_size, block_size))
        self.gui_size = tuple(gui_amp * x for x in self.frame_size)

        self.image = Image.new('RGB', self.frame_size, 'black')
        self.draw = ImageDraw.Draw(self.image)

        self.max_step = max_step
        self.step_cnt = 0

        self._seed()

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(0., 255., (*self.frame_size, 3))

        self.spf = 0.01
        self.show_gui = False

        # gui related
        self.key_map = {}
        self.last_action = None
        self.tk = None

    def __del__(self):
        pass

    def init(self, *, show_gui=False):
        self.show_gui = show_gui
        self._init()

    # should be implemented
    def _init(self):
        pass

    def _step_env(self, action):
        return 0, True

    def _render_env(self):
        pass

    def _gui_onkey(self, event):
        pass

    # gym.Env functions
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.last_action = None
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

        if self.tk is not None and not self.show_gui:
            time.sleep(self.spf)
            self.tk.update()

        return obs, rew, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.tk is not None:
                self.tk.quit()
            self.tk = None
        else:
            if self.tk is None:
                self.tk = Tk()
                self.canvas = Canvas(self.tk, 
                                     width=self.gui_size[0], 
                                     height=self.gui_size[1])
                self.canvas.pack()

    # utils functions
    def randpos(self, skip=set()):
        pos = tuple(self.np_random.randint(x) for x in self.grid_size)
        while pos in skip:
            pos = tuple(self.np_random.randint(x) for x in self.grid_size)
        return pos

    def get_frame_pos(self, pos):
        return tuple(p * bs + bs // 2 for p, bs in zip(pos, self.block_size))

    def get_bitmap(self):
        arr = np.array(self.image.getdata()).reshape((*self.frame_size, 3))
        return arr.astype('float32')

    def set_key_action(self, key, action):
        self.key_map[key] = action

    # Tk GUI setup
    def gui_start(self):
        self._render()
        self.init(show_gui=True)
        self.canvas.bind("<Key>", self.gui_onkey)
        self.canvas.focus_set()
        self.tk.after(1000//self.FPS, self.gui_step)
        self.tk.mainloop()

    def gui_step(self):
        if self.last_action is not None:
            print(self.last_action)
            obs, rew, done, info = self._step(self.last_action)
            self.last_action = None
            if abs(rew) > 1e-9:
                print('Get reward = %.2f' % rew)
        self._render_env()

        self.gui_photo = ImageTk.PhotoImage(self.image.resize(self.gui_size))
        canvas_img = self.canvas.create_image(0, 0, anchor=NW,
                                              image=self.gui_photo)
        self.tk.update()

        self.tk.after(1000//self.FPS, self.gui_step)

    def gui_onkey(self, event):
        if event.keysym in self.key_map:
            self.last_action = self.key_map[event.keysym]
        self._gui_onkey(event)
