import numpy as np
import torch
from network import DQN
import gymnasium as gym
from carpole.dqn_agent.network import DQN

class CartPoleAgent():
    def __init__(self, env: gym.Env, policy_network: DQN, target_network: DQN, epsilon: float):
        self.env = env
        self.policy_network = policy_network
        self.target_network = target_network
        self.epsilon = epsilon # epsilon = 1 / sqrt(n + 1)
    
    def select_action(self, state):
        """ Selection action based on epsilon and Q values."""
        
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample() # action_space = [0, 1, 2, 3, 4] mapped to [-10, -5, 0, 5, 10]
        else:
            q_values = self.policy_network(state)
            action = torch.argmax(q_values).item()

        return action
    
    def replay_buffer(self, state, action, reward, next_state, done):
        # store (state, action, reward, next state, done)
        # support random batching
        ...

    def learn(self, batch_size, gamma):
        # Sample a batch from replay_buffer
        # Compute predicted Q-values for the states and actions in the batch using policy_network
        # Compute Bellmans targets using target_network
        # Compute Loss (prediceted Q and target diff)
        # Backpropogate to update polic_network weights
        ...

    def update_target_network(self):
        # should copy policy network weights to target_network
        # need to decide how frequent (every N steps)
        ...