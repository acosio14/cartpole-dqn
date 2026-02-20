from dqn_agent.agent import CartPoleAgent
from envs.cartpole_env import CartPoleEnv
from dqn_agent.network import DQN
from training.trainer import TrainingArgs, Trainer

def main():

    #Cart Variables
    gravity = 9.8
    cart_mass_kg = 10
    pole_mass_kg = 5
    pole_len_m = 3

    learning_rate = 0.001
    epsilon = 0.5
    discount_factor = 1
    replay_buffer_size = 10000
    
    cartpole_env = CartPoleEnv(gravity, cart_mass_kg, pole_mass_kg, pole_len_m)

    network_input_dim = len(cartpole_env.observation_space.spaces) # Discrete has no len()
    network_output_dim = cartpole_env.action_space.n
    
    policy = DQN(network_input_dim, network_output_dim)
    target = DQN(network_input_dim, network_output_dim)

    cartpole_agent = CartPoleAgent(
        cartpole_env,
        policy, 
        target, 
        learning_rate,
        epsilon,
        discount_factor, 
    )

    training_args = TrainingArgs(
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        steps=10,
                        batch_size=5,
                        frequency_rate=10,
                        replay_buffer_size=replay_buffer_size,
                        output_dir=(
                            '/Users/adriancosio/Projects/cartpole-dqn/results',
                        ),
                    )
    
    my_trainer = Trainer(policy, cartpole_env, cartpole_agent, training_args)

    my_trainer.train()
    my_trainer.rewards

if __name__ == "__main__":
    main()