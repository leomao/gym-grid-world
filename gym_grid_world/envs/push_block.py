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
        super().__init__();
        self._is_configured = False

    def __del__(self):
        super().__del__()

    def _configure(self, grid_size=(10, 10), block_size=(5, 5),
                   max_step=200, obj_n=1, **kwargs):
        super()._configure(action_types, grid_size, block_size,
                           max_step=max_step, **kwargs)
        self.state = None

        self.player_pos = None
        self.obj_n = obj_n
        self.obj_set = None
        self.mark_set = None
        self._is_configured = True

    def _init(self):
        if not self._is_configured:
            self._configure()

        self.state = State.start

        pos_cnt = 1 + self.obj_n * 2
        pos_list = self.rand_pos(size=pos_cnt)

        self.player_pos = pos_list.pop()
        self.obj_set = set(pos_list[:self.obj_n])
        self.mark_set = set(pos_list[self.obj_n:])

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

    def _render_env(self):
        # clear canvas
        self.draw.rectangle((0, 0, *self.frame_size), fill='black')

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
            self.draw.rectangle(loc, outline='white')

