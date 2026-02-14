import numpy as np
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)
    
    def forward(self, states):
        x = nn.ReLU(self.fc1(states))
        x = nn.ReLU(self.fc2(x))

        return self.fc3(x)