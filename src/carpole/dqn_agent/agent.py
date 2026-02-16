import numpy as np
import torch
from network import DQN
from envs.cartpole_env import CartPoleEnv

class CartpoleAgent():
    def __init__(self, replay_memory, policy_network, target_network):
        self.replay_memory = replay_memory
        self.policy_network = policy_network
        self.target_network = target_network
        
    def calculate_q_value(self, state):
        return DQN(state).to(device="mps")

    def select_action(self, action_space, epsilon, q_values):
        """ Selection action based on epsilon and Q values."""
        
        if np.random.random() < epsilon:
            action = np.random.choice(action_space) # action_space = [0, 1, 2, 3, 4] mapped to [-10, -5, 0, 5, 10]
        else:
            action = torch.argmax(q_values).item()

        return action