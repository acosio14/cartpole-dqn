from carpole.dqn_agent.network import DQN
from carpole.envs.cartpole_env import CartPoleEnv
from carpole.dqn_agent.agent import CartPoleAgent
from utils.replay_buffer import ReplayBuffer
import torch
from dataclasses import dataclass


@dataclass
class TrainingArgs:
    learning_rate: float
    epsilon: float
    steps: int
    batch_size: int
    frequency_rate: int


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

    
    def train(self):

        rewards = []
        memory = ReplayBuffer()
        total_steps = 0

        for step in self.steps:
            state = self.environment.reset()
            episode_reward = 0
            terminated = False

            while not terminated:
                # Select action
                action = self.agent.select_action(state, self.epsilon)
                state, reward, terminated, _, _ = self.environment.step()

                # Store transition in memory
                memory.append(
                    state, action, reward, next_state, terminated,
                )

                # Update state
                state = next_state
                episode_reward =+ reward

                # Optimize model
                state, action, reward, next_state, terminated = (
                    memory.sample(self.batch_size)
                )
                self.agent.update(
                    state, action, reward, next_state, terminated,
                )
                
                # Update target network periodically
                total_steps =+ step
                self.agent.update_target_network(total_steps, self.frequency_rate)
            
            self.agent.epsilon = self.agent.decay_epsilon(step)

            rewards.append(episode_reward)