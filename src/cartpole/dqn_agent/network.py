import numpy as np
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,12)
        self.fc2 = nn.Linear(12,8)
        self.fc3 = nn.Linear(8,output_dim)
    
    def forward(self, states):
        x = self.fc1(states)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)

        return self.fc3(x)