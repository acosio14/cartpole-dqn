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
import numpy as np

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

        if args.train_seeds is None:
            args.train_seeds = [42]
            print(f"No train seed. Defaulting to {args.train_seeds[0]}.")

        for seed in args.train_seeds:
            my_trainer.train(seed)
        
            if args.output_dir:
                results_dir = Path(args.output_dir)
                cartpole_agent.save_model(results_dir.resolve(), f'cartpole_rk4_seed_{seed}')
    
    if args.evaluate:
        if args.eval_seeds is None:
            args.eval_seeds = [43, 44, 45]
            print(f"No evaluation seeds. Defaulting to {args.eval_seeds}.")
        
        model_metrics = []
        for file in args.evaluate:
            cartpole_env = CartPoleEnv(
                gravity=9.8,
                cart_mass=10,
                pole_mass=5,
                pole_length=3,
            )

            network_input_dim = len(cartpole_env.observation_space.spaces)
            network_output_dim = cartpole_env.action_space.n

            eval_network = DQN(network_input_dim, network_output_dim)
            model_file = Path(file).resolve()
            evaluation_policy = evaluate.load(eval_network, model_file)

            eval_agent = CartPoleAgent(
                evaluation_policy, 
                evaluate=True,
            )
            all_seeds = []
            for seed in args.eval_seeds:
                seed_mean, seed_std = (
                    evaluate.evaluate(
                        eval_agent,
                        cartpole_env,
                        episodes=10,
                        time_step=0.01,
                        seed=seed
                    )
                )
                all_seeds.append((seed_mean,seed_std))
            
            model_means, model_stds = zip(*all_seeds)
            model_mean = np.mean(model_means)
            model_std = np.std(model_means)
            
            model_metrics.append((model_mean,model_std))

        eval_means, eval_stds = zip(*model_metrics)
        overall_mean = np.mean(eval_means)
        overall_std = np.std(eval_means)
        
        if len(args.evaluate) < 2:
            overall_mean = float(*eval_means)
            overall_std = float(*eval_stds)
            eval_means = model_means
            eval_stds = model_stds

        print("\nCart Pole Performance")

        print(f"Model means: {[float(seed_mean) for seed_mean in eval_means]}") # list of means for each model
        print(f"Model stds: {[round(float(seed_std),2) for seed_std in eval_stds]}")

        print(f"Overall mean: {round(overall_mean,2)}, Overall std: {round(overall_std,2)}")
        
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