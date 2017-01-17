from functools import total_ordering
import collections
import numpy as np
from typing import Tuple

from .sendenv import SendEnv

@total_ordering
class Point:
    def __init__(self, x=0, y=0) -> None:
        if isinstance(x, collections.Iterable):
            self.x, self.y = x
            return
        self.x = x
        self.y = y

    def __add__(self, he: 'Point'):
        return Point(self.x + he.x, self.y + he.y)

    def __minus__(self, he: 'Point'):
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

class GridEnv(SendEnv):
    '''
    Abstract class for grid environments
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def _configure(self, actions, grid_size, block_size,
                   **kwargs):
        self.grid_size = grid_size
        self.block_size = block_size
        self.frame_size = tuple(x * y for x, y in zip(grid_size, block_size))

        super()._configure(actions, self.frame_size, **kwargs)

    # utils functions
    def rand_pos(self, skip=set()):
        pos = Point(self.np_random.randint(x) for x in self.grid_size)
        while pos in skip:
            pos = Point(self.np_random.randint(x) for x in self.grid_size)
        return pos

    def get_frame_rect(self, pt: Point) -> Tuple[int, int, int, int]:
        '''
        Return (xmin, ymin, xmax, ymax)
        '''
        w, h = self.block_size
        return (pt.x * w, pt.y * h, (pt.x+1) * w - 1, (pt.y+1) * h - 1)

