import numpy as np
import torch
from torch.optim import Optimizer
import gymnasium as gym
from dqn_agent.network import DQN
from utils.replay_buffer import ReplayBuffer
import torch.nn as nn
from torch.nn import functional as F

class CartPoleAgent():
    def __init__(
            self,
            env: gym.Env,
            policy_network: DQN,
            target_network: DQN,
            optimizer: Optimizer,
            learning_rate: float,
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
        self.learning_rate = learning_rate

        self.optimizer = optimizer(
            params=self.policy_network.parameters(), 
            lr=self.learning_rate,
        )
    
    def select_action(self, state):
        """ Selection action based on epsilon and Q values."""
        
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample() # action_space = [0, 1, 2, 3, 4] mapped to [-10, -5, 0, 5, 10]
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.policy_network(state) # returns (values, indicies)
            action = torch.argmax(q_values).item() # return the best value's indicies (only 1)

        return action

    def update_q_values(self, state, action, reward, next_state, terminated):
        # Sample a batch from replay_buffer
        # Compute predicted Q-values for the states and actions in the batch using policy_network
        # Compute Bellmans targets using target_network
        # Compute Loss (prediceted Q and target diff)
        # Backpropogate to update polic_network weights
        # optimizer step
        
        # target -> shape(B,A) B=batch_size A=actions
        # max(dimension = 1) - take max along dimension 1 (actions)
        # (value, indicies) -> [0] = only values
        with torch.no_grad():
            max_q_values = self.target_network(next_state).max(1)[0] # A batch of values
            target_q_values = reward + self.discount_factor * max_q_values * (1 - terminated)
        
        q_values =  self.policy_network(state).gather(1,action).squeeze(1)
        # print()
        # print(f'target: {target_q_values}')
        # print(f'q value: {q_values}')

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


    def update_target_network(self, steps, update_frequency):
        # should copy policy network weights to target_network
        # need to decide how frequent (every N steps)
        if steps % update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def decay_epsilon(self):
        return max(self.decay_rate * self.epsilon, self.epsilon_min)