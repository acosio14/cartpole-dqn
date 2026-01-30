# Make my own cartpole environment

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MyGridWorld(gym.Env):

    def __init__(self, size: int = 5):
        self.size = size # grid size 5 x 5

        self.observation_space = spaces.Dict(
            {
                "cart_position": spaces.Box(0, size, dtype=float),
                "cart_velocity": spaces.Box(0, size, dtype=float),
                "pole_position": spaces.Box(0, size, dtype=float),
                "pole_angle": spaces.Box(0, size, dtype=float),
                "pole_angular_velocity": spaces.Box(0, size, dtype=float),
            }
        )
        self._cart_position
        self._cart_velocity
        self._pole_position
        self._pole_angle
        self._pole_angular_velocity

        self.action_space = spaces.Discrete(2) # Left or right
