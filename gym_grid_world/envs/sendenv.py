import uuid
import numpy as np

from .base import BaseEnv
from .sender import start_send_thread

class SendEnv(BaseEnv):
    '''
    Abstract class for environments sending observation to env viewer
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.__stop_fn = None
        self.view_name = uuid.uuid4().hex[:6]

    def _close(self):
        self.__stop_sender()

    def _configure(self, actions, frame_size, *, max_step=-1, view_name='',
                   FPS=20, viewer_host='localhost', viewer_port=12345):
        '''
        Usage:
        '''
        super()._configure(actions, frame_size)
        self.FPS = FPS
        self.view_name = view_name
        self.__viewer_host = viewer_host
        self.__viewer_port = viewer_port
        
        self.__stop_sender()

    def _render(self, mode='human', close=False):
        if close:
            self.__stop_sender()
            return

        if mode == 'human':
            if self.__stop_fn:
                return
            if self.view_name:
                self.__stop_fn = start_send_thread(
                    self.view_name, self.frame_size, self.get_bitmap, self.FPS,
                    self.__viewer_host, self.__viewer_port
                )

    def __stop_sender(self):
        if self.__stop_fn:
            self.__stop_fn()
            self.__stop_fn = None
