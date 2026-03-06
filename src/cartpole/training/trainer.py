from dqn_agent.network import DQN
from envs.cartpole_env import CartPoleEnv
from dqn_agent.agent import CartPoleAgent
from utils.replay_buffer import ReplayBuffer
import torch
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

@dataclass
class TrainingArgs:
    episodes: int
    time_step: float
    batch_size: int
    target_update_freq: int
    replay_buffer_size: int

class Trainer():
    def __init__(
        self,
        environment: CartPoleEnv, 
        agent: CartPoleAgent,
        training_args: TrainingArgs,
    ):
        self.environment = environment
        self.agent = agent

        self.episodes = training_args.episodes
        self.time_step = np.array([training_args.time_step])
        self.batch_size = training_args.batch_size
        self.target_update_freq = training_args.target_update_freq
        self.replay_buffer_size = training_args.replay_buffer_size

        self.reward_per_episode = []
        self.epsilon_per_episode = []
        self.loss_per_episode = []
        self.steps_per_episode = []

    
    def train(self):

        memory = ReplayBuffer(self.replay_buffer_size)
        total_steps = 0

        for episode in tqdm(range(self.episodes),ncols=100,desc="Episodes"):
            state = self.environment.reset()
            episode_reward = 0
            steps_per_episode = 0
            total_loss = 0
            time = 0

            terminated = False

            while not terminated:
                # Select action
                action = self.agent.select_action(state)

                #step takes (action,time,timestep) - What is time and timestep?
                next_state, reward, terminated, *_ = (
                    self.environment.step(state, action, time, self.time_step)
                )

                # Store transition in memory
                memory.append((state, action, reward, next_state, terminated))

                # Update state
                state = next_state
                episode_reward += reward

                # Optimize model
                if len(memory) > 1000:

                    batch = list(zip(*memory.sample(self.batch_size)))

                    loss = self.agent.learn(batch)

                    total_loss += loss
                steps_per_episode += 1
                # Update target network periodically
                total_steps += 1
                time += self.time_step
                self.agent.update_target_network(total_steps, self.target_update_freq)
                self.agent.epsilon = self.agent.decay_epsilon()

            self.steps_per_episode.append(steps_per_episode)
            self.loss_per_episode.append(total_loss/steps_per_episode)
            self.reward_per_episode.append(episode_reward)
            self.epsilon_per_episode.append(self.agent.epsilon)
