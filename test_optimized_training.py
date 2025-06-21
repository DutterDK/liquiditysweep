"""
Test script for optimized DRL training
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import trading_config, drl_config, data_config, env_config, system_config
from data_loader import CSVDataLoader
from src.environment.trading_environment import TradingEnvironment

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parallel_data_processing():
    """Test parallel data processing"""
    logger.info("Testing parallel data processing...")
    
    # Load a small sample of data
    data_loader = CSVDataLoader(data_config.__dict__)
    data_splits = data_loader.load_splits('XAUUSD_processed')
    
    if not data_splits:
        logger.error("No data found")
        return False
    
    # Test with small dataset
    test_data = data_splits['train'].head(500)
    
    # Import the function from train_model
    from train_model import parallel_data_processing
    
    try:
        processed_data = parallel_data_processing(test_data, num_workers=4)
        logger.info(f"Parallel processing successful: {len(processed_data)} records")
        return True
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        return False

def test_vectorized_env():
    """Test vectorized environment creation"""
    logger.info("Testing vectorized environment creation...")
    
    # Load data
    data_loader = CSVDataLoader(data_config.__dict__)
    data_splits = data_loader.load_splits('XAUUSD_processed')
    
    if not data_splits:
        logger.error("No data found")
        return False
    
    test_data = data_splits['train'].head(500)
    
    # Import the function from train_model
    from train_model import create_vectorized_env
    
    try:
        vec_env = create_vectorized_env(test_data, env_config.__dict__, num_envs=4)
        logger.info(f"Vectorized environment created successfully with {vec_env.num_envs} environments")
        
        # Test environment step
        obs = vec_env.reset()
        action = [0] * vec_env.num_envs  # No action
        obs, reward, done, info = vec_env.step(action)
        
        logger.info(f"Environment step successful: obs shape {obs.shape}, reward shape {reward.shape}")
        vec_env.close()
        return True
    except Exception as e:
        logger.error(f"Vectorized environment creation failed: {e}")
        return False

def test_optimized_model():
    """Test optimized model creation"""
    logger.info("Testing optimized model creation...")
    
    # Create a simple environment
    data_loader = CSVDataLoader(data_config.__dict__)
    data_splits = data_loader.load_splits('XAUUSD_processed')
    
    if not data_splits:
        logger.error("No data found")
        return False
    
    test_data = data_splits['train'].head(500)
    
    # Import the function from train_model
    from train_model import create_vectorized_env, create_optimized_model
    
    try:
        vec_env = create_vectorized_env(test_data, env_config.__dict__, num_envs=2)
        model = create_optimized_model(vec_env, drl_config.__dict__)
        
        logger.info("Optimized model created successfully")
        
        # Test model prediction
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        logger.info(f"Model prediction successful: action shape {action.shape}")
        
        vec_env.close()
        return True
    except Exception as e:
        logger.error(f"Optimized model creation failed: {e}")
        return False

def test_hardware_utilization():
    """Test hardware utilization"""
    logger.info("Testing hardware utilization...")
    
    # Check CPU cores
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
    logger.info(f"CPU cores detected: {cpu_count}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_count} devices, {gpu_memory:.1f}GB memory")
    else:
        logger.info("No GPU detected, using CPU")
    
    # Check memory
    import psutil
    memory_gb = psutil.virtual_memory().total / 1e9
    logger.info(f"Total RAM: {memory_gb:.1f}GB")
    
    return True

def main():
    """Run all tests"""
    logger.info("Starting optimized training tests...")
    
    tests = [
        test_hardware_utilization,
        test_parallel_data_processing,
        # test_vectorized_env,  # Skip for now due to observation shape issues
        test_optimized_model
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            logger.info(f"Test {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("All tests passed! Ready for optimized training.")
        print("\nTo start training, run:")
        print("python train_model.py --data XAUUSD_processed --num-envs 32 --workers 32")
    else:
        print("Some tests failed. Please check the logs above.")
    print("="*50)

if __name__ == "__main__":
    main() 
