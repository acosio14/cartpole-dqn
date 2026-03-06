import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
import gymnasium as gym
from dqn_agent.network import DQN
from typing import Callable, Sequence, Tuple, Any
from datetime import datetime

class CartPoleAgent():
    def __init__(
            self,
            policy_network: DQN,
            target_network: DQN = None,
            start_epsilon: float = None,
            epsilon_min: float = None,
            epsilon_decay_rate: float = None,
            discount_factor: float = None,
            optimizer: Optimizer = None,
            loss_function: Callable[[Tensor, Tensor], Tensor] = None,
            evaluate: bool = False,
    ) -> None:
        self.policy_network = policy_network
        self.target_network = target_network
        self.epsilon = start_epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.evaluate = evaluate
    
    def select_action(self, state: np.ndarray, env: gym.Env) -> int:
        """ Selection action using epsilon-greedy policy."""
        
        if np.random.random() < self.epsilon and self.evaluate is False:
            action = env.action_space.sample() # action_space = [0, 1, 2, 3, 4] mapped to [-10, -5, 0, 5, 10]
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.policy_network(state) # returns (values, indicies)
            action = torch.argmax(q_values).item() # return the best value's indicies (only 1)

        return action

    def learn(self, batch: Sequence[Tuple[Any, Any, Any, Any, Any]]) -> float:
        """Perform learning step."""
        state = torch.tensor(np.array(batch[0]), dtype=torch.float32)
        action = torch.tensor(batch[1]).long().unsqueeze(1)
        reward = torch.tensor(batch[2], dtype=torch.float32)
        next_state = torch.tensor(np.array(batch[3]), dtype=torch.float32)
        terminated = torch.tensor(batch[4], dtype=torch.float32)

        with torch.no_grad():
            max_q_values = torch.max(self.target_network(next_state), dim=1)[0] # A batch of values
            target_q_values = reward + self.discount_factor * max_q_values * (1 - terminated)
        
        q_values =  torch.gather(self.policy_network(state), 1, action).squeeze(1)

        loss = self.loss_function(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self, steps: int, update_frequency: int) -> None:
        if steps % update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def decay_epsilon(self) -> float:
        return max(self.decay_rate * self.epsilon, self.epsilon_min)

    def save_model(self, full_path: str, name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{name}_{timestamp}.pt'

        torch.save(
            self.policy_network.state_dict(),
            full_path / filename
        )

        print(f'Model Saved: "{filename}"') 