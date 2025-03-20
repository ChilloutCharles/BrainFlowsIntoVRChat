import threading
from collections import deque

class ProtectedOSCBuffer:
    def __init__(self, max_len = 50):
        self.lock = threading.Lock()
        self.deque = deque(maxlen=max_len)
        self.deque.append((0.0, 0.0))