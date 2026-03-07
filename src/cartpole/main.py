from dqn_agent.agent import CartPoleAgent
from envs.cartpole_env import CartPoleEnv
from dqn_agent.network import DQN
from torch.nn import functional as F
import torch.nn as nn
from training.trainer import TrainingArgs, Trainer
from utils.visualization import RLPlots as plots
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch
import argparse
from evaluation import evaluate
from typing import List

def main():
    parser = argparse.ArgumentParser(description="Non-Linear Cart Pole RL Problem.")
    parser.add_argument("--train", help="Train Cart Pole model (input: config YAML file).")
    parser.add_argument("--train_seeds", nargs="+", type=int, help="Set seed(s) for training runs.")
    parser.add_argument("--plot", action="store_true", help="Plot RL figures.")
    parser.add_argument("--output_dir", type=str, help="Directory to save trained models.")
    parser.add_argument("--animation", action="store_true", help="Show animation of Cart Pole.")
    parser.add_argument("--evaluate", nargs="+", help="Evaluate model(s).")
    parser.add_argument("--eval_seeds", nargs="+", type=int, help="Set seed(s) for evaluation runs.")
    
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
        
        policy_net = DQN(network_input_dim, network_output_dim)
        target_net = DQN(network_input_dim, network_output_dim)

        optimizer = torch.optim.Adam(
                policy_net.parameters(),
                config['learning_rate'],
        )
        loss_function = nn.MSELoss()

        # agent_args = 
        
        cartpole_agent = CartPoleAgent(
            policy_net, 
            target_net,
            start_epsilon = config['start_epsilon'],
            epsilon_min = config['epsilon_min'],
            epsilon_decay_rate = config['epsilon_decay_rate'],
            discount_factor = config['discount_factor'], 
            optimizer = optimizer,
            loss_function = loss_function,  
        )

        training_args = TrainingArgs(
                            episodes = config['episodes'],
                            time_step = config['time_step'],
                            batch_size = config['batch_size'],
                            target_update_freq = config['target_update_freq'],
                            replay_buffer_size = config['replay_buffer_size'],
                        )
        
        my_trainer = Trainer(cartpole_env, cartpole_agent, training_args)

        if not isinstance(args.train_seeds, list):
            raise ValueError("Missing Training Seed(s).")

        for seed in args.train_seeds:
            my_trainer.train(seed)
        
            if args.output_dir:
                results_dir = Path(args.output_dir)
                cartpole_agent.save_model(results_dir.resolve(), f'cartpole_rk4_seed_{seed}')

    if not isinstance(args.eval_seeds, list):
        raise ValueError("Missing Evaluation Seed(s).")
    
    for file, seed in zip(args.evaluate, args.eval_seeds):
        
        model_file = Path(file).resolve()
        eval_network = DQN(network_input_dim, network_output_dim)
        eval_agent = CartPoleAgent(eval_network, evaluate=True)

        evaluate.load(model_file)
        evaluate.evaluate(eval_agent, cartpole_env, episode=10, time_step=0.01, seed=seed)
        evaluate.metrics()

        # Should also have rendering / animation flag
    
    if args.plot:

        cartpole_plots = plots( # this need to be for train and eval
            my_trainer.reward_per_episode,
            my_trainer.steps_per_episode,
            my_trainer.loss_per_episode,
            my_trainer.epsilon_per_episode
        )

        # make this into a plot arg with option: reward, loss, epsilon, step (default: reward)
        plt.figure(1)
        cartpole_plots.plot_epsilon()
        plt.figure(2)
        cartpole_plots.plot_learning_curve_moving_avg(window_size=50)
        plt.figure(3)
        cartpole_plots.plot_loss_moving_avg(window_size=50)
        plt.figure(4)
        cartpole_plots.plot_step_moving_avg(window_size=50)
        
        plt.show()
if __name__ == "__main__":
    main()