import numpy as np
from enum import IntEnum

from .grid import GridEnv

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

Movesets = [Action.up, Action.down, Action.left, Action.right]

class PickputEnv(GridEnv):

    metatdata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__();
        self._is_configured = False

    def __del__(self):
        super().__del__()

    def _configure(self, grid_size=(10, 10), block_size=(4, 4),
                   task_type=TaskType.pick, max_step=500, **kwargs):
        super()._configure(action_types, (10, 10), (6, 6),
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

        self.player_pos = self.randpos()

        self.first_pick = self.task_type != TaskType.put
        self.state = State.start
        if self.task_type & TaskType.pick:
            self.obj_pos = self.randpos()
        else:
            self.state = State.picked
        if self.task_type & TaskType.put:
            self.mark_pos = self.randpos([self.obj_pos])

    def _step_env(self, act):
        if act is None:
            return 0, False
        rew = 0
        if act == Action.stay:
            return 0, False
        if act in Movesets:
            x, y = self.player_pos
            nx, ny = self.player_pos
            if act == Action.up:
                (nx, ny) = (x, y-1)
            elif act == Action.down:
                (nx, ny) = (x, y+1)
            elif act == Action.left:
                (nx, ny) = (x-1, y)
            elif act == Action.right:
                (nx, ny) = (x+1, y)
            real_pos = (
                max(0, min(self.grid_size[0]-1, nx)),
                max(0, min(self.grid_size[1]-1, ny)),
            )
            
            self.player_pos = real_pos
            pen = 0 if real_pos == (nx, ny) else -1
            rew += pen
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
            px, py = self.get_frame_pos(self.obj_pos)
            self.draw.rectangle((px - 2, py - 2, px + 2, py + 2), fill='green')
            
        # draw mark
        if self.mark_pos:
            px, py = self.get_frame_pos(self.mark_pos)
            self.draw.rectangle((px - 2, py - 2, px + 2, py + 2), outline='white')

        # draw player
        px, py = self.get_frame_pos(self.player_pos)
        loc = (px - 2, py - 2, px + 2, py + 2)
        if self.state == State.picked:
            self.draw.ellipse(loc, fill=(0, 255, 255, 0))
        else:
            self.draw.ellipse(loc, fill='blue')
