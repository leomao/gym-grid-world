from functools import total_ordering
import collections
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw
from gym import spaces

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
        he = tuple(he)
        return Point(self.x + he[0], self.y + he[1])

    def __sub__(self, he: 'Point'):
        he = tuple(he)
        return Point(self.x - he[0], self.y - he[1])

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
        return self.to_tuple() < tuple(he)

    def __eq__(self, he: 'Point'):
        return self.to_tuple() == tuple(he)

    def __hash__(self):
        return hash(self.to_tuple())

    def abs(self):
        return abs(self.x) + abs(self.y)


class GridEnv(BaseEnv):
    '''
    Abstract class for grid environments
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def configure(self,
                  actions,
                  grid_size: Tuple[int, int],
                  block_size: int,
                  n_features: int,
                  center=False, view_radius=(0, 0),
                  **kwargs):
        self.grid_size = grid_size
        self.obs_size = grid_size
        self.block_size = block_size
        self.center = center
        self.n_features = n_features
        self.feature_map = np.zeros((*grid_size, n_features))
        if center:
            self.obs_size = tuple(2*x+1 for x in view_radius)
            self.obs_map = np.zeros((*self.obs_size, n_features))
            self.view_radius = Point(view_radius)
            self.frame_size = tuple(x*block_size for x in self.obs_size)
            super().configure(actions, self.frame_size, **kwargs)

            self.whole_size = tuple(x*block_size for x in grid_size)
            self.whole_image = Image.new('RGB', self.whole_size, 'black')
            self.view_draw = self.draw
            self.draw = ImageDraw.Draw(self.whole_image)
        else:
            self.obs_map = self.feature_map
            self.frame_size = tuple(x*block_size for x in grid_size)
            super().configure(actions, self.frame_size, **kwargs)
            self.whole_size = self.frame_size

        if self.raw_array:
            self.observation_space = spaces.Box(0., 1., self.obs_map.shape)

    # utils functions
    def rand_pos(self, size=None, skip=set(), replace=False):
        skip_n = len(skip)
        all_n = self.grid_size[0] * self.grid_size[1]
        valid_n = all_n - skip_n

        def n_to_pos(pos_n):
            return Point(pos_n // self.grid_size[1], pos_n % self.grid_size[1])

        rev_pos = set(n_to_pos(i) for i in range(valid_n, all_n))
        reverse_map = {p: q for p, q in zip(skip - rev_pos, rev_pos - skip)}

        if size is None:
            pos = n_to_pos(self.np_random.randint(valid_n))
            if pos in reverse_map:
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
        s = self.block_size
        return (pt.x * s, pt.y * s, (pt.x+1) * s - 1, (pt.y+1) * s - 1)

    def _get_raw_array(self):
        self._render_feature_map()
        if self.center:
            pos = self._get_center()
            # left, top, right, bottom
            l, t = pos - self.view_radius
            r, b = pos + self.view_radius
            r += 1
            b += 1
            # clipped value
            lc = max(l, 0)
            tc = max(t, 0)
            rc = min(r, self.grid_size[0])
            bc = min(b, self.grid_size[1])
            # offset
            lo = lc - l
            to = tc - t
            ro = self.obs_size[0] - r + rc
            bo = self.obs_size[1] - b + bc
            self.obs_map.fill(0)
            self.obs_map[lo:ro, to:bo] = self.feature_map[lc:rc, tc:bc]
        return self.obs_map

    def _render_env(self):
        self._render_grid()
        if self.center:
            pos = self._get_center()
            l, t, _, _ = self.get_frame_rect(pos - self.view_radius)
            _, _, r, b = self.get_frame_rect(pos + self.view_radius)
            # clipped value
            lc = max(l, 0)
            tc = max(t, 0)
            r = min(r, self.whole_size[0])
            b = min(b, self.whole_size[1])
            self.view_draw.rectangle((0, 0, *self.frame_size), fill='black')
            crop = self.whole_image.crop((lc, tc, r + 1, b + 1))
            self.image.paste(crop, box=(lc - l, tc - t))

    def get_info(self):
        obs = np.array(self.image.getdata()).reshape((*self.frame_size, 3))
        if self.center:
            mmap = np.array(self.whole_image.getdata())
            mmap = mmap.reshape((*self.frame_size, 3))
        else:
            mmap = obs
        return {
            'obs': obs,
            'map': mmap,
        }

    def _get_center(self):
        raise NotImplementedError

    def _render_feature_map(self):
        raise NotImplementedError

    def _render_grid(self):
        raise NotImplementedError
