import logging
import argparse
import torch
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
import os
import multiprocessing as mp

# It's crucial to have the project root in the path for imports to work correctly
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.trading_environment import TradingEnvironment
from src.features.feature_engineering import FeatureEngineer
from data_loader import CSVDataLoader
from config.config import drl_config, env_config, system_config, data_config

def precompute_features_parallel(data: pd.DataFrame, num_workers: int, logger) -> pd.DataFrame:
    """
    Pre-computes features for a dataset using multiple threads.
    """
    logger.info(f"Starting parallel feature computation with {num_workers} workers for {len(data)} records...")
    feature_engineer = FeatureEngineer({})
    features = feature_engineer.calculate_features(data, num_workers=num_workers)
    logger.info("Parallel feature computation complete.")
    return features

def make_env(rank: int, features_df: pd.DataFrame, price_data: pd.DataFrame, num_envs: int, seed: int = 0):
    """
    Utility function for multiprocessing to create a single environment instance.
    """
    def _init():
        # Each environment gets a chunk of the data
        total_len = len(features_df)
        chunk_size = total_len // num_envs
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size if rank != num_envs - 1 else total_len
        
        env_features = features_df.iloc[start_idx:end_idx]
        env_prices = price_data.iloc[start_idx:end_idx]

        env = TradingEnvironment(
            data=env_prices,
            features_df=env_features,
            config=env_config.__dict__
        )
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init()

def train(args, logger):
    """Main training function"""
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    logger.info(f"Using {device.upper()} for training")

    logger.info("Loading and processing data...")
    loader = CSVDataLoader(data_config.__dict__)
    data_dict = loader.load_splits(args.data)
    train_df = data_dict['train']
    val_df = data_dict['validation']
    logger.info(f"Data loaded: Train={len(train_df)}, Validation={len(val_df)}")

    # Convert timestamp column to datetime and set as index
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    train_df.set_index('timestamp', inplace=True)
    val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
    val_df.set_index('timestamp', inplace=True)

    # Use parallel feature pre-computation
    train_features = precompute_features_parallel(train_df, args.num_envs, logger)
    val_features = precompute_features_parallel(val_df, args.num_envs, logger)

    if train_features.empty or val_features.empty:
        logger.error("Feature computation failed, exiting.")
        return

    logger.info("Aligning original price data with feature sets...")
    train_price_data_aligned = train_df.loc[train_features.index]
    val_price_data_aligned = val_df.loc[val_features.index]
    
    assert len(train_features) == len(train_price_data_aligned), "Train data mismatch"
    assert len(val_features) == len(val_price_data_aligned), "Validation data mismatch"
    logger.info("Data alignment successful.")

    # Set an environment variable for make_env to know the number of envs
    os.environ['NUM_ENVS'] = str(args.num_envs)

    logger.info(f"Creating {args.num_envs} parallel environments...")
    
    # Use SubprocVecEnv for multiple cores, DummyVecEnv for a single core
    VecEnv = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv

    train_env_fns = [lambda i=i: make_env(i, train_features, train_price_data_aligned, num_envs=args.num_envs, seed=i) for i in range(args.num_envs)]
    vec_env = VecEnv(train_env_fns)

    eval_env_fns = [lambda i=i: make_env(i, val_features, val_price_data_aligned, num_envs=args.num_envs, seed=i + args.num_envs) for i in range(args.num_envs)]
    eval_vec_env = VecEnv(eval_env_fns)

    eval_freq = max(drl_config.eval_freq // args.num_envs, 1)
    eval_callback = EvalCallback(eval_vec_env, best_model_save_path=system_config.model_save_path,
                                 log_path=system_config.model_save_path, eval_freq=eval_freq,
                                 deterministic=True, render=False, n_eval_episodes=10)
    
    save_freq = max(drl_config.eval_freq // args.num_envs, 1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=system_config.model_save_path, name_prefix="drl_model")

    callback = CallbackList([eval_callback, checkpoint_callback])
    
    ppo_params = drl_config.__dict__.copy()
    ppo_params.pop('model_type', None)
    ppo_params.pop('buffer_size', None)
    ppo_params.pop('tau', None)

    model = PPO("MlpPolicy", vec_env, tensorboard_log=system_config.tensorboard_log, device=device, verbose=0, **ppo_params)
    
    logger.info("Starting model training...")
    logger.info(f"Tensorboard logs available at: {system_config.tensorboard_log}")
    
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(os.path.join(system_config.model_save_path, "final_model.zip"))
        logger.info("Training finished. Final model saved.")
    except KeyboardInterrupt:
        model.save(os.path.join(system_config.model_save_path, "interrupted_model.zip"))
        logger.warning("Training interrupted by user. Model saved.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
    finally:
        vec_env.close()
        eval_vec_env.close()

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(system_config.model_save_path, "training.log"), mode='w'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description="Train a DRL trading model.")
    parser.add_argument('--data', type=str, required=True, help='Name of the processed data file.')
    parser.add_argument('--num-envs', type=int, default=32, help='Number of parallel environments.')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total timesteps for training.')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA for training.')
    args = parser.parse_args()

    # Add a new 'workers' argument for clarity, even if it's not used in this stable version
    # This keeps the interface consistent for when we re-add parallel feature generation
    setattr(args, 'workers', args.num_envs) 

    os.makedirs(system_config.model_save_path, exist_ok=True)
    os.makedirs(system_config.tensorboard_log, exist_ok=True)
        
    train(args, logger)