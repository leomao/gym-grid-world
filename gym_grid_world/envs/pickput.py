import numpy as np
from enum import IntEnum

from .grid import GridEnv, Point

class TaskType(IntEnum):
    pick = 0b1
    put = 0b10
    both = 0b11

State = IntEnum('State', [
    'start',
    'picked',
    'end' # this must exists
])

action_types = [ 'stay', 'up', 'down', 'left', 'right', 'pick', 'put', ]
Action = IntEnum('Action', action_types, start=0)

Movesets = {
    Action.up: Point(0, -1),
    Action.down: Point(0, 1),
    Action.right: Point(1, 0),
    Action.left: Point(-1, 0),
}

class PickputEnv(GridEnv):

    metadata = {'render.modes': ['human']}
    reward_range = (-1., 6.)

    def __init__(self):
        super().__init__();
        self._is_configured = False

    def __del__(self):
        super().__del__()

    def configure(self, grid_size=(10, 10), block_size=(5, 5),
                  task_type=TaskType.pick, max_step=500, **kwargs):
        super().configure(action_types, grid_size, block_size,
                          max_step=max_step, **kwargs)
        self.state = None
        self.first_pick = True
        self.task_type = task_type

        self.player_pos = None
        self.obj_pos = None
        self.mark_pos = None
        self._is_configured = True

    def _init(self):
        if not self._is_configured:
            self._configure()

        self.player_pos = self.rand_pos()

        self.first_pick = self.task_type != TaskType.put
        self.state = State.start
        if self.task_type & TaskType.pick:
            self.obj_pos = self.rand_pos()
        else:
            self.state = State.picked
        if self.task_type & TaskType.put:
            self.mark_pos = self.rand_pos(skip=set([self.obj_pos]))

    def _step_env(self, act):
        if act is None:
            return 0, False
        rew = 0
        if act == Action.stay:
            return 0, False
        if act in Movesets:
            prev_pos = self.player_pos
            self.player_pos += Movesets[act]

            if not self.is_in_map(self.player_pos):
                self.player_pos = prev_pos

            # penalty
            if self.player_pos == prev_pos:
                rew -= 1
            
        elif act == Action.pick and self.state == State.start:
            if self.obj_pos == self.player_pos:
                self.state = State.picked
                if self.first_pick:
                    self.first_pick = False
                    rew += 1
                if (not self.task_type & TaskType.put):
                    self.state = State.end
        elif act == Action.put and self.state == State.picked:
            if self.mark_pos == self.player_pos:
                self.state = State.end
                rew += 1
            else:
                self.state = State.start
                self.obj_pos = self.player_pos
                rew -= 1

        done = False
        if self.state == State.end:
            rew += 5
            done = True

        return rew, done

    def _render_env(self):
        # clear canvas
        self.draw.rectangle((0, 0, *self.frame_size), fill='black')

        # draw obj
        if self.obj_pos and self.state == State.start:
            loc = self.get_frame_rect(self.obj_pos)
            self.draw.rectangle(loc, fill='green')
            
        # draw mark
        if self.mark_pos:
            loc = self.get_frame_rect(self.mark_pos)
            self.draw.rectangle(loc, outline='white')

        # draw player
        loc = self.get_frame_rect(self.player_pos)
        if self.state == State.picked:
            self.draw.ellipse(loc, fill=(0, 255, 255, 0))
        else:
            self.draw.ellipse(loc, fill='blue')
