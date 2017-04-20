from functools import total_ordering
import collections
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw

from .base import BaseEnv

@total_ordering
class Point:
    def __init__(self, x=0, y=0) -> None:
        if isinstance(x, collections.Iterable):
            self.x, self.y = x
        else:
            self.x = x
            self.y = y

    def __add__(self, he: 'Point'):
        return Point(self.x + he.x, self.y + he.y)

    def __sub__(self, he: 'Point'):
        return Point(self.x - he.x, self.y - he.y)

    def __mul__(self, v):
        return Point(v * self.x, v * self.y)

    def __rmul__(self, v):
        return self.__mul__(v)

    def to_tuple(self):
        return (self.x, self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return '(%s, %s)' % (self.x, self.y)

    def __lt__(self, he: 'Point'):
        return self.to_tuple() < he.to_tuple()

    def __eq__(self, he: 'Point'):
        return self.to_tuple() == he.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

class GridEnv(BaseEnv):
    '''
    Abstract class for grid environments
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def configure(self, actions, grid_size, block_size, center=False,
                  view_radius=(0, 0), **kwargs):
        self.grid_size = grid_size
        self.block_size = block_size
        self.center = center
        if center:
            self.view_radius = Point(view_radius)
            self.frame_size = tuple((2*x+1)*y for x, y in zip(view_radius, block_size))
            super().configure(actions, self.frame_size, **kwargs)

            self.whole_size = tuple(x*y for x, y in zip(grid_size, block_size))
            self.whole_image = Image.new('RGB', self.whole_size, 'black')
            self.view_draw = self.draw
            self.draw = ImageDraw.Draw(self.whole_image)
        else:
            self.frame_size = tuple(x*y for x, y in zip(grid_size, block_size))
            super().configure(actions, self.frame_size, **kwargs)
            self.whole_size = self.frame_size

    # utils functions
    def rand_pos(self, size=None, skip=set(), replace=False):
        skip_n = len(skip)
        all_n = self.grid_size[0] * self.grid_size[1]
        valid_n = all_n - skip_n

        def n_to_pos(pos_n):
            return Point(pos_n // self.grid_size[1], pos_n % self.grid_size[1])

        reverse_map = {pos: n_to_pos(i)
                       for pos, i in zip(skip, range(valid_n, all_n))}

        if size is None:
            pos = n_to_pos(self.np_random.randint(valid_n))
            if pos in skip:
                pos = reverse_map[pos]
            return pos
        else:
            pos_n_list = self.np_random.choice(valid_n, size, replace=replace)
            pos_list = [n_to_pos(x) for x in pos_n_list]
            pos_list = [reverse_map[pos] if pos in skip else pos
                        for pos in pos_list]
            return pos_list

    def is_in_map(self, pos):
        return (pos.x >= 0 and pos.x < self.grid_size[0] and
                pos.y >= 0 and pos.y < self.grid_size[1])

    def get_frame_rect(self, pt: Point) -> Tuple[int, int, int, int]:
        '''
        Return (xmin, ymin, xmax, ymax)
        '''
        w, h = self.block_size
        return (pt.x * w, pt.y * h, (pt.x+1) * w - 1, (pt.y+1) * h - 1)

    def _render_env(self):
        self._render_grid()
        if self.center:
            self.view_draw.rectangle((0, 0, *self.frame_size), fill='black')
            pos = self.get_center()
            left, top, _, _ = self.get_frame_rect(pos - self.view_radius)
            _, _, right, bot = self.get_frame_rect(pos + self.view_radius)
            t_left = max(left, 0)
            t_top = max(top, 0)
            right = min(right, self.whole_size[0])
            bot = min(bot, self.whole_size[1])
            crop = self.whole_image.crop((t_left, t_top, right, bot))
            self.image.paste(crop, box=(t_left - left, t_top - top))

    def get_info(self):
        obs = np.array(self.image.getdata()).reshape((*self.frame_size, 3))
        mmap = np.array(self.whole_image.getdata()).reshape((*self.whole_size, 3))
        return {
            'obs': obs,
            'map': mmap,
        }
