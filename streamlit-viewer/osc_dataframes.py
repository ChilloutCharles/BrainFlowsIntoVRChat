from collections import deque
import numpy as np
from itertools import islice   
NUM_STORED_TIMESTEPS = 1024

class OSCDataFrame:
    def __init__(self):
        self.data = {}
        self.secondsSinceLastUpdate = 0.0

class OSCFrameDeque:
    
    def __init__(self, max_frames=NUM_STORED_TIMESTEPS):
        self.deque = deque(maxlen=max_frames)

    def add_frame(self, frame: OSCDataFrame):
        self.deque.append(frame)

    def get_frames(self) -> deque:
        return self.deque
    
    def get_latest_frames(self, n) :

        #copy the deque
        copy = deque(self.deque)

        if n > len(self.deque):
            return self.get_latest_frames(len(self.deque))
        return islice(copy, 0, n)
    
    def get_frames_count(self):
        return len(self.deque)
    

class OSCFrameCollector:

    def __init__(self):
        self.currentFrame = OSCDataFrame()
        self.osc_framedeque = OSCFrameDeque()

    def process_osc_deltatime (self, seconds_since_last_update: float):
        self.currentFrame.secondsSinceLastUpdate = seconds_since_last_update
        self.osc_framedeque.add_frame(self.currentFrame)
        self.currentFrame = OSCDataFrame()

    def process_osc_message(self, path: str, value: float):
        if self.currentFrame is None:
            self.currentFrame = OSCDataFrame()
        self.currentFrame.data[path] = value

    def get_osc_dataframes(self) -> OSCFrameDeque:
        return self.osc_framedeque
    
