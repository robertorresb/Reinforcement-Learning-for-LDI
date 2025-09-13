import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import the NEW hedging environment
from src.environment.hedging_env import PortfolioHedgingEnv

def train_hedging_model(
    data_path="data/processed/NVDA_hedging_features.csv",
    log_dir="logs/tensorboard",
    checkpoints_dir="checkpoints",
    best_model_dir="models/best_model",
    eval_log_dir="logs/evaluation",
    total_timesteps=5_000_000,
    episode_months=6,
    window_size=5,
    dead_zone=0.01,  
    initial_capital=2_000_000,
    commission=0.00125,
    action_change_penalty_threshold=0.2,
    max_shares_per_trade=0.5,
    algorithm="PPO",
    verbose=True,
    n_envs=4
):
    """
    Train a reinforcement learning agent for portfolio hedging.
    
    Args:
        data_path: Path to the processed dataset
        log_dir: Directory for TensorBoard logs
        checkpoints_dir: Directory for model checkpoints
        best_model_dir: Directory to save the best model
        eval_log_dir: Directory for evaluation logs
        total_timesteps: Total number of timesteps to train for
        episode_length_months: Length of each episode in months
        window_size: Observation window size
        dead_zone: Dead zone around action changes
        initial_capital: Initial capital for portfolio 
        commission: Commission rate for trades
        action_change_penalty_threshold: Threshold for penalizing large action changes
        max_shares_per_trade: Maximum proportion of portfolio value per trade
        algorithm: RL algorithm to use ("PPO" or "DDPG")
        n_envs: Number of parallel environments for training
        verbose: Whether to print progress messages
    
    Returns:
        model: Trained RL model
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Load data
    if verbose:
        print("Loading data from {}...".format(data_path))
    df = pd.read_csv(data_path)
    
    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'])
    dates = df['Date'].values
    
    # Extract prices and features
    prices = df['Close'].astype(np.float32).values
    if np.any(prices <= 0):
        raise ValueError("Prices contain non-positive values")
    
    # Remove Date and Close from features
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close']]
    features = df[feature_columns].astype(np.float32).values
    
    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Price data points: {len(prices)}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Dead zone: ±{dead_zone*100}%")
    
    def make_env():
        return PortfolioHedgingEnv(
            features=features,
            prices=prices,
            dates=dates,
            episode_months=episode_months,
            window_size=window_size,
            dead_zone=dead_zone,
            initial_capital=initial_capital,
            commission=commission,
            action_change_penalty_threshold=action_change_penalty_threshold,
            max_shares_per_trade=max_shares_per_trade
        )
    
    train_env = make_vec_env(lambda: make_env(), n_envs=n_envs, seed=0)
    
    eval_env = PortfolioHedgingEnv(
        features=features,
        prices=prices,
        dates=dates,
        episode_months=episode_months,
        window_size=window_size,
        dead_zone=dead_zone,
        initial_capital=initial_capital,
        commission=commission,
        action_change_penalty_threshold=action_change_penalty_threshold,
        max_shares_per_trade=max_shares_per_trade
    )
    
    
    # Configure logging
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    
    if algorithm.upper() == "DDPG":
        
        if verbose:
            print("Using DDPG (Deep Deterministic Policy Gradient) for continuous hedging control...")
        
        n_actions = train_env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.2 * np.ones(n_actions)  
        )
        
        model = DDPG(
            "MlpPolicy",
            train_env,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=1,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        )
        model_prefix = "ddpg_hedging"
        
    else:  # PPO
        if verbose:
            print("Using PPO (Proximal Policy Optimization) for hedging...")
        
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-4,
            n_steps=2048 // n_envs,
            batch_size=64,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
        )
        model_prefix = "ppo_hedging"
    
    model.set_logger(new_logger)
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1),
        save_path=checkpoints_dir,
        name_prefix=model_prefix
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=max(5_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    
    # Train model
    if verbose:
        print(f"Training {algorithm} for {total_timesteps} timesteps...")
        print(f"Episode length: {episode_months} months")
        print(f"Portfolio value: ${initial_capital:,}")
        print(f"Dead zone: ±{dead_zone*100:.1f}%")
        print(f"Action change penalty threshold: {action_change_penalty_threshold:.2f}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=f"{model_prefix}_run"
    )
    
    # Save final model
    final_model_path = os.path.join("models", f"{model_prefix}_final")
    model.save(final_model_path)
    if verbose:
        print("Final model saved to {}".format(final_model_path))
    
    return model



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train portfolio hedging model')
    parser.add_argument('--data_path', default="data/processed/NVDA_hedging_features.csv",
                       help='Path to processed dataset')
    parser.add_argument('--algorithm', choices=['PPO', 'DDPG'], default='PPO',
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                       help='Total training timesteps')
    parser.add_argument('--episode_months', type=int, default=6,
                       help='Episode length in months')
    parser.add_argument('--model_path', default="models/best_model/best_model",
                        help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    print("Training model...")
    trained_model = train_hedging_model(
        data_path=args.data_path,
        total_timesteps=args.timesteps,
        episode_months=args.episode_months,
        algorithm=args.algorithm
    )