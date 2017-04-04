import numpy as np
from enum import IntEnum
from typing import Tuple

from .grid import GridEnv, Point

class EatBulletEnv(GridEnv):

    metadata = {'render.modes': ['human']}
    reward_range = (0., 1.)

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

    def configure(self, grid_size=(10, 10), block_size=(5, 5),
                  food_n: int = 3,
                  max_step: int = 500,
                  **kwargs):
        super().configure(self.ActionNames, grid_size, block_size,
                          max_step=max_step, **kwargs)
        self.player_pos = None # type: Point
        self.food_n = food_n
        self.foods_pos = [] # type: List[Point]
        self._is_configured = True

    def _init(self):
        if not self._is_configured:
            self._configure()

        pos_cnt = 1 + self.food_n
        pos_list = self.rand_pos(size=pos_cnt)
        self.player_pos = pos_list.pop()
        self.foods_pos = set(pos_list)

    def _step_env(self, act) -> Tuple[float, bool]:
        if act is None:
            return 0., False
        rew = 0.
        if act == self.Action.stay:
            return 0., False

        if act in self.Movesets:
            prev_pos = self.player_pos
            self.player_pos += self.Movesets[act]

            if not self.is_in_map(self.player_pos):
                self.player_pos = prev_pos

        rew += self._check_eaten()

        return rew, False

    def _check_eaten(self) -> float:
        if self.player_pos in self.foods_pos:
            new_food_pos = self.rand_pos(skip=set(self.foods_pos))
            self.foods_pos.remove(self.player_pos)
            self.foods_pos.add(new_food_pos)
            return 1.
        else:
            return 0.

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
