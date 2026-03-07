from envs.cartpole_env import CartPoleEnv
from dqn_agent.agent import CartPoleAgent
from dqn_agent.network import DQN
import torch
from tqdm import tqdm
from typing import List

# class Evaluator():
#     def __init__(
#         self,
#         environment: CartPoleEnv, 
#         agent: CartPoleAgent,
#         episodes: int,
#         time_step: float,
#     ):
#         self.environment = environment
#         self.agent = agent
#         self.episodes = episodes
#         self.time_step = time_step
#         self.loaded_policy: DQN = None
#         self.reward_per_episode = []

def load(self, model_file: str):
    self.loaded_policy.load_state_dict(
        torch.load(model_file, weights_only=True)
    )
    self.loaded_policy.eval()

def evaluate(self, seed: int):
    self.environment.reset(seed)
    torch.manual_seed(seed)
    
    for episode in tqdm(range(self.episodes),ncols=100,desc="Episodes"):
        state = self.environment.reset()
        episode_reward = 0
        time = 0

        terminated = False

        while not terminated:
            # Select action
            action = self.agent.select_action(state)

            #step takes (action,time,timestep)
            next_state, reward, terminated, *_ = (
                self.environment.step(state, action, time, self.time_step)
            )

            # Update state
            state = next_state
            episode_reward += reward

            # Update target network periodically
            total_steps += 1
            time += self.time_step

        self.reward_per_episode.append(episode_reward)

def metrics(self):
    metric = 0
    return metric
