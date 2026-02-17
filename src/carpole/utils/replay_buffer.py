from collections import deque
import random

class ReplayBuffer(deque):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)

    def sample(self, batch_size):
        return random.sample(self, min(batch_size, len(self)))