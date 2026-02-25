from dqn_agent.network import DQN
from envs.cartpole_env import CartPoleEnv
from dqn_agent.agent import CartPoleAgent
from utils.replay_buffer import ReplayBuffer
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np


@dataclass
class TrainingArgs:
    episodes: int
    time_step: float
    batch_size: int
    target_update_freq: int
    replay_buffer_size: int
    output_dir: str


class Trainer():
    def __init__(
        self,
        model: DQN,
        environment: CartPoleEnv, 
        agent: CartPoleAgent,
        training_args: TrainingArgs,
    ):
        self.model = model
        self.environment = environment
        self.agent = agent

        self.episodes = training_args.episodes
        self.time_step = np.array([training_args.time_step])
        self.batch_size = training_args.batch_size
        self.target_update_freq = training_args.target_update_freq
        self.output_dir = training_args.output_dir
        self.replay_buffer_size = training_args.replay_buffer_size
        
        self.reward_per_episode = []
        self.epsilon_per_episode = []
        self.loss_per_episode = []
        self.steps_per_episode = []

    
    def train(self):

        memory = ReplayBuffer(self.replay_buffer_size)


        for episode in range(self.episodes):
            state = self.environment.reset()
            episode_reward = 0
            steps_per_episode = 0
            total_loss = 0
            time = 0
            total_steps = 0
            terminated = False

            while not terminated:
                # Select action
                action = self.agent.select_action(state)

                #step takes (action,time,timestep) - What is time and timestep?
                next_state, reward, terminated, *_ = (
                    self.environment.step(action, time, self.time_step)
                )

                # Store transition in memory
                memory.append((state, action, reward, next_state, terminated))

                # Update state
                state = next_state
                episode_reward += reward

                # Optimize model
                # state, action, reward, next_state, terminated = mini_batches
                mini_batch = list(zip(*memory.sample(self.batch_size)))
                # I was taking out the 0 index of deque not a batch of 5
                # Basically never reorganized sample to separate buckets (state, action, etc)

                state_batch = torch.tensor(mini_batch[0], dtype=torch.float32)
                action_batch = torch.tensor(mini_batch[1]).long().unsqueeze(1)
                reward_batch = torch.tensor(mini_batch[2], dtype=torch.int32)
                nstate_batch = torch.tensor(mini_batch[3], dtype=torch.float32)
                terminated_batch = torch.tensor(mini_batch[4], dtype=torch.int32)
                loss = self.agent.update_q_values(
                    state_batch,
                    action_batch,
                    reward_batch,
                    nstate_batch,
                    terminated_batch,
                )
                
                total_loss += loss.item()
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

    
    def save_model(self, name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = Path('')
        filename = f'{name}_{timestamp}.pt'

        torch.save(
            self.model.state_dict(),
            dir_path / filename
        )