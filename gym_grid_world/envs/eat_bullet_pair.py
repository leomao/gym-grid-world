import numpy as np
from enum import IntEnum
from typing import Tuple

from .grid import GridEnv, Point

class EatBulletPairEnv(GridEnv):

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
        super().__init__()
        self._is_configured = False

    def configure(self, grid_size=(10, 10), block_size=5,
                  food_n: int = 3,
                  max_step: int = 500,
                  **kwargs):
        super().configure(self.ActionNames, grid_size, block_size,
                          n_features=4,
                          max_step=max_step, **kwargs)
        self.player_pos = None # type: Point
        self.food_n = food_n
        self.foods = {}
        self._is_configured = True

    def _init(self):
        if not self._is_configured:
            self.configure()

        pos_cnt = 1 + self.food_n*2
        pos_list = self.rand_pos(size=pos_cnt)
        self.player_pos = pos_list.pop()
        self.foods = {
            pos: 0 if idx < self.food_n else 1
            for (idx, pos) in enumerate(pos_list)
        }
        self.last_eaten_type = None

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
        if self.player_pos in self.foods:
            pos = self.player_pos
            typ = self.foods[pos]
            new_food_pos = self.rand_pos(skip=self.foods.keys())
            self.foods.pop(pos)
            self.foods[new_food_pos] = typ
            if self.last_eaten_type is None:
                self.last_eaten_type = typ
                return 0.
            else:
                reward = -1. if self.last_eaten_type == typ else 1.
                self.last_eaten_type = None
                return reward
        else:
            return 0.

    def _get_center(self):
        return self.player_pos

    def _render_feature_map(self):
        self.feature_map.fill(0)
        feat_cnt = 0

        loc = tuple(self.player_pos)
        self.feature_map[loc][feat_cnt] = 1
        feat_cnt += 1

        # draw foods
        for pos, typ in self.foods.items():
            loc = tuple(pos)
            feat_idx = feat_cnt + int(typ == 0)
            self.feature_map[loc][feat_idx] = 1
        feat_cnt += 2

        self.feature_map[:,:,feat_cnt] = 1

    def _render_grid(self):
        # clear canvas
        self.draw.rectangle((0, 0, *self.whole_size), fill='#333')

        # draw player
        loc = self.get_frame_rect(self.player_pos)
        self.draw.ellipse(loc, fill='blue')

        # draw foods
        for pos, typ in self.foods.items():
            loc = self.get_frame_rect(pos)
            self.draw.rectangle(loc, fill='green' if typ == 0 else 'red')
