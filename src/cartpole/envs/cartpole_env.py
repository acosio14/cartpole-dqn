# Make my own cartpole environment

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import odeint
from utils.numerical_integrators import runge_kutta_fourth_order


class CartPoleEnv(gym.Env):

    def __init__(self, gravity: float, cart_mass: float, pole_mass: float, pole_length: float):
        
        # physical constants
        self.constants = [gravity, cart_mass, pole_mass, pole_length]
        
        # initalized state variables
        self._cart_position = 0.0
        self._cart_velocity = 0.0
        self._pole_angle = 0.1
        self._pole_angular_velocity = 0.0

        self.state = np.array(
            [
                self._cart_position,
                self._cart_velocity,
                self._pole_angle,
                self._pole_angular_velocity,
            ]
        )
        
        # observation and action space
        self.observation_space = spaces.Dict(
            {
                "cart_position": spaces.Box(-5.0, 5.0, shape=(1,), dtype=np.float32),
                "cart_velocity": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),

                "pole_angle": spaces.Box(-np.deg2rad(30), np.deg2rad(30), shape=(1,), dtype=np.float32), 
                "pole_angular_velocity": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(4) # Discrete(5) -> [0,1,2,3,4]
        self.force = [-10, -2, 2, 10]

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with cart and poles states (position, velocity, angle, angular velocity)
        """
        return np.array(
            [
                self._cart_position,
                self._cart_velocity,
                self._pole_angle,
                self._pole_angular_velocity,
            ]
        )
    
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

    def step(self, state, action, time, timestep):

        next_state= (
            runge_kutta_fourth_order(state, self.force[action], timestep, self.constants)
        )

        pole_angle = abs(next_state[2])
        if pole_angle <= np.deg2rad(10):
            reward = 1.0
        elif pole_angle >= np.deg2rad(10) and pole_angle <= np.deg2rad(30):
            reward = 0.5
        else:
            reward = 0

        if pole_angle >= np.deg2rad(30) or time >= 20:
            terminated = True 
        else:
            terminated = False
        
        truncated = False
        info = None

        return next_state, reward, terminated, truncated, info