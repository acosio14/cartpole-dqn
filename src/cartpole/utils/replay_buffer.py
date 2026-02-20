from collections import deque
import random

class ReplayBuffer(deque):
    def __init__(self, buffer_length):
        super().__init__(maxlen=buffer_length)
        self.buffer_length = buffer_length
        self.is_full = False

    def sample(self, batch_size):
        if len(self) == 0:
            return ValueError("Replay Buffer is empty")
        
        return random.sample(self, min(batch_size, len(self)))
    
    def _is_maxlen(self):
        if len(self) == self.buffer_length:
            self.is_full = True