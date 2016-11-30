import numpy as np
from enum import IntEnum

if __name__ == '__main__':
    from grid import GridEnv
else:
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

    def __init__(self, *, task_type=TaskType.pick, max_step=500):
        super().__init__(action_types, (10, 10), (6, 6), max_step=max_step)
        self.state = None
        self.first_pick = True
        self.task_type = task_type

        self.player_pos = None
        self.obj_pos = None
        self.mark_pos = None

        self.last_act = None

        self.set_key_action('Up', Action.up)
        self.set_key_action('Down', Action.down)
        self.set_key_action('Left', Action.left)
        self.set_key_action('Right', Action.right)
        self.set_key_action('x', Action.pick)
        self.set_key_action('z', Action.put)

    def _init(self):
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

if __name__ == '__main__':
    import sys
    task = TaskType.pick
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
        if task_name == 'put':
            task = TaskType.put
        elif task_name == 'both':
            task = TaskType.both
    env = PickputEnv(task_type=task)
    try:
        env.gui_start()
    except KeyboardInterrupt:
        pass

