import numpy as np
from enum import IntEnum
from typing import Tuple

from .grid import GridEnv, Point
from .eat_bullet import EatBulletEnv

class EatBulletMemEnv(EatBulletEnv):

    def __init__(self) -> None:
        super().__init__()

    def configure(self,
                  disappear_dist=3,
                  **kwargs):
        super().configure(**kwargs)
        self.disappear_dist = disappear_dist

    def _render_feature_map(self):
        self.feature_map.fill(0)
        feat_cnt = 0

        loc = tuple(self.player_pos)
        self.feature_map[loc][feat_cnt] = 1
        feat_cnt += 1

        # draw foods
        for pos in self.foods_pos:
            dis = (self.player_pos - pos).abs()
            if dis >= self.disappear_dist:
                loc = tuple(pos)
                self.feature_map[loc][feat_cnt] = 1
        feat_cnt += 1

        self.feature_map[:,:,feat_cnt] = 1

    def _render_grid(self):
        # clear canvas
        self.draw.rectangle((0, 0, *self.whole_size), fill='#333')

        # draw player
        loc = self.get_frame_rect(self.player_pos)
        self.draw.ellipse(loc, fill='blue')

        # draw foods
        for pos in self.foods_pos:
            dis = (self.player_pos - pos).abs()
            if dis >= self.disappear_dist:
                loc = self.get_frame_rect(pos)
                self.draw.rectangle(loc, fill='green')
