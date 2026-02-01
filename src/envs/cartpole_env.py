# Make my own cartpole environment

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CartPoleEnv(gym.Env):

    def __init__(self):
        
        # physical constants
        self.gravity = 9.8 # m/s/s
        self.cart_mass = 10 # kg
        self.pole_mass = 5
        self.pole_length = 3

        # initalized variables
        self._cart_position = 0.0
        self._cart_velocity = 0.0
        self._cart_acceleration = 0.0
        self._pole_angle = 0.0
        self._pole_angular_velocity = 0.0
        self._pole_angular_acceleration = 0.0
        self._force = 0.0
        
        # observation and action space
        self.observation_space = spaces.Dict(
            {
                "cart_position": spaces.Box(-5.0, 5.0, shape=(1,), dtype=np.float32),
                "cart_velocity": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "cart_acceleration": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),

                "pole_angle": spaces.Box(-np.deg2rad(30), np.deg2rad(30), shape=(1,), dtype=np.float32), 
                "pole_angular_velocity": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "pole_angular_accelerate": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Dict(
            {
                "force": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            }
        )

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with cart and poles states (position, velocity, angle, angular velocity)
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
    
    def step(self):

        x1 = self._get_obs()["cart_position"]
        x2 = self._get_obs()["cart_velocity"]
        x3 = self._get_obs()["pole_angle"] # theta
        x4 = self._get_obs()["pole_angular_velocity"] # theta_dot
        
        x1_dot= x2
        x2_dot = -mp * L * np.sin(x3) * np.square(x4) + mp * g * np.cos(x3) * np.sin(x4) + Force / (M + mp * np.square(np.sin(x3)))
        x3_dot = x4
        x4_dot = -(M + mp) * g * np.sin(x3) - mp * L * np.sin(x3) * np.cos(x3) * np.square(x4) - Force * np.cos(x3) / (L * (M + mp * np.square(np.sin(x3))) )