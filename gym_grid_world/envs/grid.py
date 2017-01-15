import numpy as np

from .sendenv import SendEnv

class GridEnv(SendEnv):
    '''
    Abstract class for grid environments
    '''
    metatdata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def _configure(self, actions, grid_size, block_size,
                   **kwargs):
        self.grid_size = grid_size
        self.block_size = block_size
        self.frame_size = tuple(x * y for x, y in zip(grid_size, block_size))

        super()._configure(actions, self.frame_size, **kwargs)

    # utils functions
    def randpos(self, skip=set()):
        pos = tuple(self.np_random.randint(x) for x in self.grid_size)
        while pos in skip:
            pos = tuple(self.np_random.randint(x) for x in self.grid_size)
        return pos

    def get_frame_pos(self, pos):
        return tuple(p * bs + bs // 2 for p, bs in zip(pos, self.block_size))
