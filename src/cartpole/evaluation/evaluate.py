from envs.cartpole_env import CartPoleEnv
from dqn_agent.agent import CartPoleAgent
from dqn_agent.network import DQN
import torch
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

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

def load(dqn_network: DQN, model_file: str):
    
    state_dict = torch.load(model_file, weights_only=True)
    dqn_network.load_state_dict(state_dict)

    return dqn_network.eval()

def evaluate(
    agent: CartPoleAgent,
    environment: CartPoleEnv,
    episodes: int,
    time_step: float,
    seed: int
) -> Tuple[np.float64, np.float64]:
    
    environment.reset(seed)
    torch.manual_seed(seed)

    reward_per_episode = []
    for episode in tqdm(range(episodes),ncols=100,desc="Episodes"):
        state = environment.reset()
        total_reward = 0
        time = 0

        terminated = False

        while not terminated:
            # Select action
            action = agent.select_action(state)

            # Take Step
            next_state, reward, terminated, *_ = (
                environment.step(state, action, time, time_step)
            )

            # Update state
            state = next_state
            total_reward += reward

            # Update target network periodically
            total_steps += 1
            time += time_step

        reward_per_episode.append(total_reward)
    
    return np.mean(reward_per_episode), np.std(reward_per_episode)

def metrics(all_seeds):
    means, stds = zip(*all_seeds)

    return np.mean(means), np.mean(stds)


