import numpy as np
import torch

import gymnasium as gym
from dqn_agent.network import DQN


class CartPoleAgent():
    def __init__(
            self,
            env: gym.Env,
            policy_network: DQN,
            target_network: DQN,
            start_epsilon: float,
            epsilon_min: float,
            epsilon_decay_rate: float,
            discount_factor: float,
        ):
        self.env = env
        self.policy_network = policy_network
        self.target_network = target_network
        self.epsilon = start_epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
    
    def select_action(self, state):
        """ Selection action based on epsilon and Q values."""
        
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample() # action_space = [0, 1, 2, 3, 4] mapped to [-10, -5, 0, 5, 10]
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.policy_network(state) # returns (values, indicies)
            action = torch.argmax(q_values).item() # return the best value's indicies (only 1)

        return action

    def update_q_values(self, batch):

        state = torch.tensor(np.array(batch[0]), dtype=torch.float32)
        action = torch.tensor(batch[1]).long().unsqueeze(1)
        reward = torch.tensor(batch[2], dtype=torch.float32)
        next_state = torch.tensor(np.array(batch[3]), dtype=torch.float32)
        terminated = torch.tensor(batch[4], dtype=torch.float32)

        with torch.no_grad():
            max_q_values = self.target_network(next_state).max(1)[0] # A batch of values
            target_q_values = reward + self.discount_factor * max_q_values * (1 - terminated)
        
        q_values =  self.policy_network(state).gather(1,action).squeeze(1)

        return q_values, target_q_values

    def update_target_network(self, steps, update_frequency):
        if steps % update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def decay_epsilon(self):
        return max(self.decay_rate * self.epsilon, self.epsilon_min)