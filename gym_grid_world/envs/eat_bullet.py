import numpy as np
from enum import IntEnum
from typing import Tuple

from .grid import GridEnv, Point

class EatBulletEnv(GridEnv):

    metadata = {'render.modes': ['human']}

    ActionNames = [ 'stay', 'up', 'down', 'left', 'right', ]
    Action = IntEnum('Action', ActionNames, start=0)
    Movesets = {
        Action.up: Point(0, -1),
        Action.down: Point(0, 1),
        Action.right: Point(1, 0),
        Action.left: Point(-1, 0),
    }

    def __init__(self) -> None:
        super().__init__();
        self._is_configured = False

    def _configure(self, grid_size=(10, 10), block_size=(5, 5),
                   food_n: int = 3,
                   max_step: int = 500,
                   **kwargs):
        super()._configure(self.ActionNames, grid_size, block_size,
                           max_step=max_step, **kwargs)
        self.player_pos = None # type: Point
        self.food_n = food_n
        self.foods_pos = [] # type: List[Point]
        self._is_configured = True

    def _init(self):
        if not self._is_configured:
            self._configure()

        self.foods_pos = []
        for i in range(4):
            self.foods_pos.append(self.rand_pos(set(self.foods_pos)))

        self.player_pos = self.rand_pos(set(self.foods_pos))

    def _step_env(self, act) -> Tuple[float, bool]:
        if act is None:
            return 0., False
        rew = 0.
        if act == self.Action.stay:
            return 0., False

        if act in self.Movesets:
            self.player_pos += self.Movesets[act]

            x, y = self.player_pos
            self.player_pos.x = np.clip(x, 0, self.grid_size[0]-1)
            self.player_pos.y = np.clip(y, 0, self.grid_size[1]-1)

        rew += self._check_eaten()

        return rew, False

    def _check_eaten(self) -> float:
        if self.player_pos not in self.foods_pos:
            return 0.

        self.foods_pos.remove(self.player_pos)
        self.foods_pos.append(self.rand_pos(self.foods_pos))

        return 1.

    def _render_env(self):
        # clear canvas
        self.draw.rectangle((0, 0, *self.frame_size), fill='black')

        # draw player
        loc = self.get_frame_rect(self.player_pos)
        self.draw.ellipse(loc, fill='blue')

        # draw foods
        for pos in self.foods_pos:
            loc = self.get_frame_rect(pos)
            self.draw.rectangle(loc, fill='green')
