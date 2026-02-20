from dqn_agent.network import DQN
from envs.cartpole_env import CartPoleEnv
from dqn_agent.agent import CartPoleAgent
from utils.replay_buffer import ReplayBuffer
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TrainingArgs:
    learning_rate: float
    epsilon: float
    steps: int
    batch_size: int
    frequency_rate: int
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

        self.learning_rate = training_args.learning_rate
        self.epsilon = training_args.epsilon
        self.steps = training_args.steps
        self.batch_size = training_args.batch_size
        self.frequency_rate = training_args.frequency_rate
        self.output_dir = training_args.output_dir
        self.rewards = []

    
    def train(self):

        memory = ReplayBuffer()
        total_steps = 0

        for step in self.steps:
            state = self.environment.reset()
            episode_reward = 0
            terminated = False

            while not terminated:
                # Select action
                action = self.agent.select_action(state, self.epsilon)

                #step takes (action,time,timestep) - What is time and timestep?
                next_state, reward, terminated, _, _ = self.environment.step(action)

                # Store transition in memory
                memory.append(state, action, reward, next_state, terminated)

                # Update state
                state = next_state
                episode_reward =+ reward

                # Optimize model
                # state, action, reward, next_state, terminated = mini_batches
                mini_batches = memory.sample(self.batch_size)
                self.agent.update_q_values(*mini_batches)
                
                # Update target network periodically
                total_steps =+ step
                self.agent.update_target_network(total_steps, self.frequency_rate)
            
            self.agent.epsilon = self.agent.decay_epsilon(step)
            self.rewards.append(episode_reward)
    
    def save_model(self, name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = Path('')
        filename = f'{name}_{timestamp}.pt'

        torch.save(
            self.model.state_dict(),
            dir_path / filename
        )