from dqn_agent.agent import CartPoleAgent
from envs.cartpole_env import CartPoleEnv
from dqn_agent.network import DQN
from training.trainer import TrainingArgs, Trainer
import matplotlib.pyplot as plt

def main():

    #Cart Variables
    gravity = 9.8
    cart_mass_kg = 10
    pole_mass_kg = 5
    pole_len_m = 3
    
    cartpole_env = CartPoleEnv(gravity, cart_mass_kg, pole_mass_kg, pole_len_m)

    network_input_dim = len(cartpole_env.observation_space.spaces) # Discrete has no len()
    network_output_dim = cartpole_env.action_space.n
    
    policy = DQN(network_input_dim, network_output_dim)
    target = DQN(network_input_dim, network_output_dim)

    cartpole_agent = CartPoleAgent(
        cartpole_env,
        policy, 
        target, 
        learning_rate=0.001,
        start_epsilon=1,
        epsilon_min = 0.001,
        epsilon_decay_rate = 0.995,
        discount_factor=0.99, 
    )

    training_args = TrainingArgs(
                        episodes=100,
                        time_step=0.1,
                        batch_size=5,
                        target_update_freq=10,
                        replay_buffer_size=10000,
                        output_dir=(
                            '/Users/adriancosio/Projects/cartpole-dqn/results',
                        ),
                    )
    
    my_trainer = Trainer(policy, cartpole_env, cartpole_agent, training_args)

    my_trainer.train()
    print(my_trainer.reward_per_episode)
    plt.plot(my_trainer.reward_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards Per Episode")
    plt.show()

if __name__ == "__main__":
    main()