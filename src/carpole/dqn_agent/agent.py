import numpy as np

class CartpoleAgent():
    def __init__(self, replay_memory, policy_network, target_network):
        self.replay_memory = replay_memory
        self.policy_network = policy_network
        self.target_network = target_network
        
    def calculate_q_value(state):
        q_value = DQN(state).to(device="mps")

    def select_action(epsilon, q_values):
        """ Selection action based on epsilon and Q values."""
        
        if np.random.random() < epsilon:
            action = np.random.randint(low=0, high=1) # low, high placeholders
        else:
            action = np.argmax(q_values)

        return action