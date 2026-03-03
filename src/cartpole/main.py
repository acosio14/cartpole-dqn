from dqn_agent.agent import CartPoleAgent
from envs.cartpole_env import CartPoleEnv
from dqn_agent.network import DQN
from training.trainer import TrainingArgs, Trainer
from utils.visualization import RLPlots as plots
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Non-Linear Cart Pole RL Problem.")
    parser.add_argument("--train", help="Train Cart Pole model.")
    parser.add_argument("--plot", nargs='+', help="Plot RL figures.")
    parser.add_argument("--output_dir", type=str, help="Directory to save trained models.")
    parser.add_argument("--animation", help="Show animation of Cart Pole.")
    parser.add_argument("--load", help="Load a model.")
    parser.add_argument("--run", help="Run inferance model.")

    args = parser.parse_args()
    
    if args.train:
        config_file = Path(args.train).name

        config_path = Path(__file__).parent.parent.parent / 'config' / config_file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        cartpole_env = CartPoleEnv(
            config['gravity'],
            config['cart_mass_kg'],
            config['pole_mass_kg'],
            config['pole_len_m'],
        )

        network_input_dim = len(cartpole_env.observation_space.spaces)
        network_output_dim = cartpole_env.action_space.n
        
        policy = DQN(network_input_dim, network_output_dim)
        target = DQN(network_input_dim, network_output_dim)

        cartpole_agent = CartPoleAgent(
            cartpole_env,
            policy, 
            target,
            optimizer = torch.optim.Adam,
            learning_rate = config['learning_rate'],
            start_epsilon = config['start_epsilon'],
            epsilon_min = config['epsilon_min'],
            epsilon_decay_rate = config['epsilon_decay_rate'],
            discount_factor = config['discount_factor'], 
        )

        training_args = TrainingArgs(
                            episodes = config['episodes'],
                            time_step = config['time_step'],
                            batch_size = config['batch_size'],
                            target_update_freq = config['target_update_freq'],
                            replay_buffer_size = config['replay_buffer_size'],
                        )
        
        my_trainer = Trainer(policy, cartpole_env, cartpole_agent, training_args)

        my_trainer.train()
        
        if args.output_dir:
            results_dir = Path(args.output_dir)
            my_trainer.save_model(results_dir.resolve(), 'cartpole_rk4')
    
    if args.plot:

        cartpole_plots = plots(
            my_trainer.reward_per_episode,
            my_trainer.steps_per_episode,
            my_trainer.loss_per_episode,
            my_trainer.epsilon_per_episode
        )

        # make this into a plot arg with option: reward, loss, epsilon, step (default: reward)
        # plt.figure(1)
        # cartpole_plots.plot_epsilon()
        # plt.figure(2)
        # cartpole_plots.plot_learning_curve_moving_avg(window_size=50)
        # plt.figure(3)
        # cartpole_plots.plot_loss_moving_avg(window_size=50)
        # plt.figure(4)
        # cartpole_plots.plot_step_moving_avg(window_size=50)
        
        # plt.show()


if __name__ == "__main__":
    main()