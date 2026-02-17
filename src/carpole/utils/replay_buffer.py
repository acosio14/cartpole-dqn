

class ReplayBuffer():
    def __init__(self, max_size):
        self.memory = []
        self.max_size = max_size

    def add(self, data: tuple):
        if len(self.memory) < self.max_size:
            # just add
            self.memory.extend(data)
        else:
            ...
            # remove oldest, add newest
    
    def sample(self, batch_size):
        ...
        # shuffle list and return a batch