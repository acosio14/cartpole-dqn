import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np

@dataclass
class RLPlots:
    rewards: float
    steps: float
    losses: float
    epsilons: float

    def plot_learning_curve(self):
        plt.plot(self.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Rewards Per Episode")
    
    def plot_learning_curve_moving_avg(self, window_size):
        avg_rewards = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(avg_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward")
        plt.title("Avg Rewards Per Episode")

    # Steps
    def plot_steps_per_episode(self):
        plt.plot(self.steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Steps Per Episode")

    def plot_step_moving_avg(self, window_size):
        avg_steps = np.convolve(self.steps, np.ones(window_size)/window_size, mode='valid')
        plt.plot(avg_steps)
        plt.xlabel("Episode")
        plt.ylabel("Avg Steps")
        plt.title("Avg Steps Per Episode")

    # Losses
    def plot_mse_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Loss Per Episode")

    def plot_loss_moving_avg(self, window_size):
        avg_loss = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(avg_loss)
        plt.xlabel("Episode")
        plt.ylabel("Avg Loss")
        plt.title("Avg Loss Per Episode")

    def plot_epsilon(self):
        plt.plot(self.epsilons)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Per Episode")

    def plot(self, data):
        plt.plot(data)
        plt.xlabel("Epsidoe")
        plt.ylabel(f'{data}')
        plt.title(f"{data} Per Episode")
