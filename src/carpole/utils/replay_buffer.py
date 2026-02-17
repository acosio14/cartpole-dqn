from collections import deque
import random

class ReplayBuffer(deque):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)
        self.maxlen = maxlen
        self._is_full = False

    def sample(self, batch_size):
        if len(self) == 0:
            return ValueError("Replay Buffer is empty")
        
        return random.sample(self, min(batch_size, len(self)))
    
    def _is_maxlen(self):
        if len(self) == self.maxlen:
            self._is_full = True