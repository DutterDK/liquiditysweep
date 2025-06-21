"""
Configuration file for DRL Liquidity Sweep Trading System
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    # Symbol and timeframes
    symbol: str = "EURUSD"
    base_timeframe: str = "M1"
    timeframes: List[str] = None
    
    # Risk management
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_positions: int = 3
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 4.0
    
    # Liquidity sweep parameters
    sweep_lookback: int = 20  # Candles to look back for sweeps
    min_sweep_strength: float = 0.5  # Minimum strength for valid sweep
    order_block_lookback: int = 50  # Candles to look back for order blocks
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]

@dataclass
class DRLConfig:
    """Deep Reinforcement Learning configuration"""
    # Model parameters
    model_type: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 128  # Increased for 7970x
    buffer_size: int = 2000000  # Increased for 128GB RAM
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 4
    gradient_steps: int = 4  # Increased for better training
    
    # Network architecture
    hidden_layers: List[int] = None
    activation: str = "relu"
    
    # Training parameters
    total_timesteps: int = 2000000  # Increased for better performance
    eval_freq: int = 10000
    save_freq: int = 50000
    log_interval: int = 100
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [512, 512, 256, 256]  # Larger network for 7970x

@dataclass
class DataConfig:
    """Data configuration"""
    # Data sources
    data_source: str = "CSV"  # CSV files provided by user
    
    # Data parameters
    train_split: float = 0.7
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # Feature engineering
    feature_window: int = 100
    normalize_features: bool = True
    use_technical_indicators: bool = True

@dataclass
class EnvironmentConfig:
    """Trading environment configuration"""
    # Environment parameters
    initial_balance: float = 100000.0
    max_steps: int = 10000
    transaction_fee: float = 0.0  # No commissions as requested
    
    # Observation space
    observation_window: int = 50
    include_position_info: bool = True
    include_account_info: bool = True
    
    # Action space
    action_space_type: str = "discrete"  # discrete or continuous
    num_actions: int = 3  # hold, buy, sell
    
    # Reward function
    reward_type: str = "sharpe_ratio"  # pnl, sharpe_ratio, sortino_ratio
    reward_scale: float = 1.0

@dataclass
class SystemConfig:
    """System configuration optimized for high-performance hardware"""
    # Hardware optimization for 7970x (32 cores)
    num_workers: int = 32  # Full CPU utilization
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.9  # Use more GPU memory
    
    # Logging
    log_level: str = "INFO"
    tensorboard_log: str = "./logs/tensorboard"
    model_save_path: str = "./models"
    
    # Performance optimization
    enable_profiling: bool = True
    memory_limit_gb: int = 120  # Use more RAM (128GB total)
    num_envs: int = 32  # Match CPU cores for maximum parallelization
    
    # Vectorized training settings
    vectorized_training: bool = True
    parallel_data_processing: bool = True
    batch_size_multiplier: int = 4  # Larger batches for better GPU utilization
    gradient_steps_multiplier: int = 2  # More gradient steps per update

# Global configuration instance
trading_config = TradingConfig()
drl_config = DRLConfig()
data_config = DataConfig()
env_config = EnvironmentConfig()
system_config = SystemConfig()

# Environment variables
def load_env_vars():
    """Load configuration from environment variables"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Override configs with environment variables if present
    if os.getenv("NUM_WORKERS"):
        system_config.num_workers = int(os.getenv("NUM_WORKERS"))
    if os.getenv("BATCH_SIZE"):
        drl_config.batch_size = int(os.getenv("BATCH_SIZE"))
    if os.getenv("BUFFER_SIZE"):
        drl_config.buffer_size = int(os.getenv("BUFFER_SIZE"))

# Load environment variables on import
load_env_vars() 