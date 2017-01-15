import numpy as np

from .base import BaseEnv
from .sender import start_send_thread

class SendEnv(BaseEnv):
    '''
    Abstract class for environments sending observation to env viewer
    '''
    metatdata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.__stop_fn = None

    def __del__(self):
        self.stop_sender()
        super().__del__()

    def _configure(self, actions, frame_size, *, max_step=-1, name='', FPS=20,
                   viewer_host='localhost', viewer_port=12345):
        '''
        Usage:
        '''
        super()._configure(actions, frame_size)
        self.FPS = FPS
        self.name = name
        self.__viewer_host = viewer_host
        self.__viewer_port = viewer_port
        
        self.stop_sender()

        if self.name:
            self.__stop_fn = start_send_thread(
                self.name, self.frame_size, self.get_bitmap, self.FPS,
                self.__viewer_host, self.__viewer_port
            )

    def stop_sender(self):
        if self.__stop_fn:
            self.__stop_fn()
