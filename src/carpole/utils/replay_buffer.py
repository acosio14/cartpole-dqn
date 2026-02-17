from collections import deque
import random

class ReplayBuffer(deque):
    def __init__(self, max_size):
        super().__init__(maxlen=max_size)

    def sample(self, batch_size):
        return random.sample(self, min(batch_size, len(self)))