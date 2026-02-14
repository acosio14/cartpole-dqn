# Make my own cartpole environment

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import odeint
from utils.numerical_integrators import runge_kutta_fourth_order


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

        self.state = [
            self._cart_position,
            self._cart_velocity,
            self._pole_angle,
            self._pole_angular_velocity,
        ]
        
        # observation and action space
        self.observation_space = spaces.Dict(
            {
                "cart_position": spaces.Box(-5.0, 5.0, shape=(1,), dtype=np.float32),
                "cart_velocity": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),

                "pole_angle": spaces.Box(-np.deg2rad(30), np.deg2rad(30), shape=(1,), dtype=np.float32), 
                "pole_angular_velocity": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(5)
        self._action_to_forces = {
            "left_hard": -10, 
            "left": -5, 
            "stop": 0, 
            "right": 5, 
            "right_hard": 10
        }

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
    
    def ordinay_differantion_equations(self, state, force):
        
        mp = self.pole_mass
        M = self.cart_mass
        L = self.pole_length
        g = self.gravity
        
        # Get Derivatives
        x_dot= state[1] # cart velocity
        x_ddot = (
            -mp * L * np.sin(state[2]) * np.square(state[3])
            + mp * g * np.cos(state[2]) * np.sin(state[3])
            + force / (M + mp * np.square(np.sin(state[2]))) 
        ) # cart acceleration
        theta_dot = state[3] # angular velocity
        theta_ddot = (
            -(M + mp) * g * np.sin(state[2]) 
            - mp * L * np.sin(state[2]) * np.cos(state[2]) * np.square(state[3]) 
            - force * np.cos(state[2]) / (L * (M + mp * np.square(np.sin(state[2]))) ) 
        )# angular acceleration

        return [x_dot, x_ddot, theta_dot, theta_ddot]
    

    def step(self, action, time, timestep):

        x = [
            self._cart_position,
            self._cart_velocity,
            self._pole_angle,
            self._pole_angular_velocity,
        ]      

        xdot = self.ordinay_differantion_equations()
        state = runge_kutta_fourth_order(xdot, x, action, timestep) # basically: state = state + state_dot * dt
        observations = state

        # reward = 1 if self._pole_angle equals 0
        if state[2] == 0:
            reward = 1.0
        else:
            reward = -0.01

        if state[2] >= 30 or time >= 10:
            terminated = True # if pole falls (>= 30 deg), time duration (10 sec, <=30 deg)
        
        truncated = False
        info = None

        return observations, reward, terminated, truncated, info