import numpy as np
from enum import IntEnum

from .grid import GridEnv, Point


State = IntEnum('State', [
    'start',
    'end' # this must exists
])


action_types = [ 'stay', 'up', 'down', 'left', 'right', ]
Action = IntEnum('Action', action_types, start=0)
Movesets = {
    Action.up: Point(0, -1),
    Action.down: Point(0, 1),
    Action.right: Point(1, 0),
    Action.left: Point(-1, 0),
}


class PushBlockEnv(GridEnv):

    metadata = {'render.modes': ['human']}
    reward_range = (-1., 5.)

    def __init__(self):
        super().__init__()
        self._is_configured = False

    def configure(self, grid_size=(10, 10), block_size=5,
                  max_step=200, obj_n=1, **kwargs):
        super().configure(action_types, grid_size, block_size,
                          n_features=4,
                          max_step=max_step, **kwargs)
        self.state = None

        self.player_pos = None
        self.obj_n = obj_n
        self.obj_set = None
        self.mark_set = None
        self._is_configured = True

    def _init(self):
        if not self._is_configured:
            self.configure()

        self.state = State.start

        w, h = self.grid_size
        lb = set((0, x) for x in range(h))
        tb = set((x, 0) for x in range(w))
        rb = set((w-1, x) for x in range(h))
        bb = set((x, h-1) for x in range(w))
        edges = lb | tb | rb | bb

        self.obj_set = set(self.rand_pos(size=self.obj_n, skip=edges))

        pos_cnt = 1 + self.obj_n
        pos_list = self.rand_pos(size=pos_cnt, skip=self.obj_set)

        self.player_pos = pos_list.pop()
        self.mark_set = set(pos_list)

    def _step_env(self, act):
        if act is None:
            return 0, False
        rew = 0
        if act == Action.stay:
            return 0, False
        elif act in Movesets:
            prev_pos = self.player_pos
            self.player_pos += Movesets[act]

            x, y = self.player_pos
            if not self.is_in_map(self.player_pos):
                self.player_pos = prev_pos

            if self.player_pos in self.obj_set:
                new_obj_pos = self.player_pos + Movesets[act]
                if (self.is_in_map(new_obj_pos) and
                        not new_obj_pos in self.obj_set):
                    self.obj_set.remove(self.player_pos)
                    self.obj_set.add(new_obj_pos)
                else:
                    self.player_pos = prev_pos

            if self.obj_set == self.mark_set:
                self.state = State.end

            # penalty
            if self.player_pos == prev_pos:
                rew -= 1

        done = False
        if self.state == State.end:
            rew += 5
            done = True

        return rew, done

    def _get_center(self):
        return self.player_pos

    def _render_feature_map(self):
        self.feature_map.fill(0)
        feat_cnt = 0

        loc = tuple(self.player_pos)
        self.feature_map[loc][feat_cnt] = 1
        feat_cnt += 1

        # draw obj
        for obj_pos in self.obj_set:
            loc = tuple(obj_pos)
            self.feature_map[loc][feat_cnt] = 1
        feat_cnt += 1

        # draw mark
        for mark_pos in self.mark_set:
            loc = tuple(mark_pos)
            self.feature_map[loc][feat_cnt] = 1
        feat_cnt += 1

        self.feature_map[:, :, feat_cnt] = 1

    def _render_grid(self):
        # clear canvas
        self.draw.rectangle((0, 0, *self.frame_size), fill='#333')

        # draw player
        loc = self.get_frame_rect(self.player_pos)
        self.draw.ellipse(loc, fill='blue')

        # draw obj
        for obj_pos in self.obj_set:
            loc = self.get_frame_rect(obj_pos)
            self.draw.rectangle(loc, fill='green')

        # draw mark
        for mark_pos in self.mark_set:
            loc = self.get_frame_rect(mark_pos)
            if self.block_size == 1:
                self.draw.rectangle(loc, fill='white')
            else:
                self.draw.rectangle(loc, outline='white')

