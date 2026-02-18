import numpy as np
import torch
import gymnasium as gym
from carpole.dqn_agent.network import DQN
from utils.replay_buffer import ReplayBuffer
import torch.nn as nn
from torch.nn import functional as F

class CartPoleAgent():
    def __init__(
            self,
            env: gym.Env,
            policy_network: DQN,
            target_network: DQN, 
            epsilon: float,
            discount_factor: float,
            maxlen: int = 10000,
        ):
        self.env = env
        self.policy_network = policy_network
        self.target_network = target_network
        self.epsilon = epsilon # epsilon = 1 / sqrt(n + 1)
        self.discount_factor = discount_factor
        self.replay_buffer = ReplayBuffer(maxlen)
        self.iteration = 0
        self.learning_rate = 0.001
    
    def select_action(self, state):
        """ Selection action based on epsilon and Q values."""
        
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample() # action_space = [0, 1, 2, 3, 4] mapped to [-10, -5, 0, 5, 10]
        else:
            q_values = self.policy_network(state)
            action = torch.argmax(q_values).item() # return the best value's indicies (only 1)

        return action

    def learn(self, state, action, reward, next_state, terminated, batch_size):
        # Sample a batch from replay_buffer
        # Compute predicted Q-values for the states and actions in the batch using policy_network
        # Compute Bellmans targets using target_network
        # Compute Loss (prediceted Q and target diff)
        # Backpropogate to update polic_network weights
        # optimizer step
        self.replay_buffer.append(state, action, reward, next_state, terminated)
        # DON"T KNOW IF THIS SHOULD BE OUTSIDE IN TRAINING LOOP
        ''' Only bring in one batch?'''
        
        if self.replay_buffer.is_full:
            mini_batch = self.replay_buffer.sample(batch_size)
            state, action, reward, next_state, terminated = mini_batch
            # target -> shape(B,A) B=batch_size A=actions
            # max(dimension = 1) - take max along dimension 1 (actions)
            # (value, indicies) -> [0] = only values
            max_q_values = self.target_network(next_state).max(1)[0] # A batch of values
            target_q_values = reward + self.discount_factor * max_q_values * (1 - terminated)
            q_values =  self.policy_network(state)

            loss = F.mse_loss(q_values, target_q_values)
            loss.backward()
            torch.optim.Adam(
                params=self.policy_network.parameters(), 
                rl=self.learning_rate,
            ).step()
        
        self.update_target_network(update_frequency=60)


    def update_target_network(self, update_frequency):
        # should copy policy network weights to target_network
        # need to decide how frequent (every N steps)
        if self.iteration % update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.iteration = 0

    def decay_epsilon(self, episode):
        self.epsilon = (1 / np.sqrt(episode + 1)) * self.epsilon