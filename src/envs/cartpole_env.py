# Make my own cartpole environment

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import odeint


class CartPoleEnv(gym.Env):

    def __init__(self):
        
        # physical constants
        self.gravity = 9.8 # m/s/s
        self.cart_mass = 10 # kg
        self.pole_mass = 5
        self.pole_length = 3

        # initalized state variables
        self._cart_position = 0.0
        self._cart_velocity = 0.0
        self._pole_angle = 0.0
        self._pole_angular_velocity = 0.0

        # self.numerical_integrator ("original CartPole used: euler")
        
        # observation and action space
        self.observation_space = spaces.Dict(
            {
                "cart_position": spaces.Box(-5.0, 5.0, shape=(1,), dtype=np.float32),
                "cart_velocity": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),

                "pole_angle": spaces.Box(-np.deg2rad(30), np.deg2rad(30), shape=(1,), dtype=np.float32), 
                "pole_angular_velocity": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)

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
    
    def step(self, action):

        force = action # action should be a force applied to cart: -F (left), +F (right), 0 (no force)
        
        mp = self.pole_mass
        M = self.cart_mass
        L = self.pole_length
        g = self.gravity
        
        # Grab States
        x1 = self._cart_position # x
        x2 = self._cart_velocity # x_dot
        x3 = self._pole_angle # theta
        x4 = self._pole_angular_velocity #theta_dot
        
        # Get Derivatives
        x1_dot= x2 # cart vel
        x2_dot = -mp * L * np.sin(x3) * np.square(x4) + mp * g * np.cos(x3) * np.sin(x4) + force / (M + mp * np.square(np.sin(x3))) # cart acceleration
        x3_dot = x4 # angular velocity
        x4_dot = -(M + mp) * g * np.sin(x3) - mp * L * np.sin(x3) * np.cos(x3) * np.square(x4) - force * np.cos(x3) / (L * (M + mp * np.square(np.sin(x3))) ) # angular acceleration
        
        # Use integrator to get next state
        state = self.rk4_integrator([x1,x2,x3,x4], [x1_dot, x2_dot, x3_dot, x4_dot], time, delta_time)

        # reward = 1 if self._pole_angle equals 0
        reward = 1
        terminated = False # if pole falls (>= 30 deg), time duration (10 sec, <=30 deg)
        truncated = False
        info = None

        observations = state

        return observations, reward, terminated, truncated, info