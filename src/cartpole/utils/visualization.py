import matplotlib.pyplot as plt
from dataclasses import dataclass

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

    def plot_steps_per_episode(self):
        plt.plot(self.steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Steps Per Episode")

    def plot_mse_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Loss Per Episode")

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
