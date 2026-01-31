# Make my own cartpole environment

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CartPoleEnv(gym.Env):

    def __init__(self, size: int = 5):
        self.size = size 

        self.observation_space = spaces.Dict(
            {
                "cart_position": spaces.Box(-size, size, dtype=np.float32),
                "cart_velocity": spaces.Box(-np.inf, np.inf, dtype=np.float32),
                "pole_angle": spaces.Box(-np.deg2rad(30), np.deg2rad(30), dtype=np.float32), 
                "pole_angular_velocity": spaces.Box(-np.inf, np.inf, dtype=np.float32),
            }
        )
        self._cart_position = np.array([0], dtype=np.float32)
        self._cart_velocity = np.array([0], dtype=np.float32)
        self._pole_angle = np.array([0], dtype=np.float32)
        self._pole_angular_velocity = np.array([0], dtype=np.float32)

        self.action_space = spaces.Discrete(2)

        self._force_applied = {
            0: np.array([-1]),
            1: np.array([1]),
        }

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with cart and poles states (position, velocity, angle, ang_rate)
        """
        return {
            "cart_position": self._cart_position,
            "cart_velocity": self._cart_velocity,
            "pole_angle": self._pole_angle,
            "pole_angular_velocity": self._pole_angular_velocity,
        }
    
    def reset(self, seed: int = None, options: dict = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """

        super().reset(seed = seed)

        self._pole_angle = self.np_random.uniform(-np.deg2rad(15), np.deg2rad(15))

        return self._get_obs()
