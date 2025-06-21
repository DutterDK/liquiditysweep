# DRL Liquidity Sweep Trading System

A state-of-the-art Deep Reinforcement Learning (DRL) system that learns to trade using the liquidity sweep strategy. This system is specifically designed to achieve approximately **85% win rate** by identifying and trading liquidity sweeps in the market.

## 🎯 Strategy Overview

The liquidity sweep strategy identifies areas where stop losses are likely to be placed and trades the reversal that occurs after these stops are triggered. Key components:

- **Liquidity Sweeps**: Areas where price moves beyond previous highs/lows to trigger stop losses
- **Order Blocks**: Key supply/demand zones that act as reversal points
- **Market Structure**: Identification of swing highs/lows and market structure shifts
- **Multi-timeframe Analysis**: Confluence across M1, M5, M15, H1, H4, D1 timeframes

## 🚀 Features

- **Advanced DRL Model**: Uses PPO (Proximal Policy Optimization) with custom neural architecture
- **High Performance**: Optimized for 7970x CPU (32 cores) and 128GB RAM
- **Comprehensive Feature Engineering**: 50+ features including technical indicators and strategy signals
- **Real-time Performance Tracking**: Win rate, Sharpe ratio, drawdown, and other key metrics
- **Multi-environment Training**: Parallel training environments for faster convergence
- **No Commissions**: Focus on strategy profitability without transaction costs
- **Extensive Logging**: TensorBoard integration and detailed performance reports

## 📊 Performance Targets

- **Win Rate**: ~85%
- **Risk Management**: 1:2 risk-reward ratio minimum
- **Drawdown**: <10%
- **Sharpe Ratio**: >2.0

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd DRL-liquidity-sweep
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup directories**:
```bash
mkdir -p data models logs
```

## 📁 Project Structure

```
├── config/
│   └── config.py              # Configuration settings
├── src/
│   ├── strategy/
│   │   └── liquidity_sweep.py # Core strategy implementation
│   ├── features/
│   │   └── feature_engineering.py # Feature engineering
│   └── environment/
│       └── trading_environment.py # DRL trading environment
├── data/                      # Data storage
├── models/                    # Trained models
├── logs/                      # Training logs
├── data_loader.py            # CSV data loader
├── train_model.py            # Main training script
└── requirements.txt          # Dependencies
```

## 📈 Usage

### 1. Prepare Your Data

Place your CSV files in the `data/` directory. The system expects MT5 format:

```csv
DATE,TIME,OPEN,HIGH,LOW,CLOSE,TICKVOL,VOL,SPREAD
2024.01.02,01:00:00,2063.570,2064.650,2063.060,2064.280,100,0,350
2024.01.02,01:01:00,2064.120,2064.720,2064.060,2064.720,47,0,350
```

### 2. Load and Split Data

```bash
python data_loader.py --input your_data.csv --split --info
```

This will:
- Load and validate your data
- Split into train/validation/test sets
- Display data information

### 3. Train the Model

```bash
python train_model.py --data your_data --num-envs 8 --timesteps 1000000
```

**Parameters**:
- `--data`: Base filename for data splits
- `--num-envs`: Number of parallel environments (default: 4)
- `--timesteps`: Training timesteps (default: 2,000,000)
- `--gpu`: Use GPU if available
- `--model-path`: Load existing model

### 4. Monitor Training

The system provides comprehensive monitoring:

- **Real-time Logs**: Check `logs/training.log`
- **TensorBoard**: `tensorboard --logdir logs/tensorboard`
- **Performance Plots**: Automatically generated in `models/results/`

## ⚙️ Configuration

Key configuration options in `config/config.py`:

### Trading Strategy
```python
trading_config = TradingConfig(
    risk_per_trade=0.02,           # 2% risk per trade
    max_positions=3,               # Maximum concurrent positions
    stop_loss_atr_multiplier=2.0,  # ATR multiplier for stop loss
    take_profit_atr_multiplier=4.0 # ATR multiplier for take profit
)
```

### DRL Model
```python
drl_config = DRLConfig(
    model_type="PPO",
    learning_rate=3e-4,
    batch_size=128,                # Optimized for 7970x
    hidden_layers=[512, 512, 256, 256],
    total_timesteps=2000000
)
```

### System Optimization
```python
system_config = SystemConfig(
    num_workers=32,                # 7970x optimization
    num_envs=16,                   # Parallel environments
    memory_limit_gb=100            # RAM optimization
)
```

## 📊 Performance Tracking

The system tracks comprehensive metrics:

### Real-time Metrics
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Total PnL**: Cumulative profit/loss
- **Max Drawdown**: Maximum peak-to-trough decline
- **Profit Factor**: Gross profit / Gross loss
- **Total Trades**: Number of completed trades

### Training Visualization
- Win rate progression
- Sharpe ratio evolution
- PnL curve
- Trade count over time

## 🎯 Strategy Implementation

### Liquidity Sweep Detection
```python
# Detects when price sweeps beyond previous highs/lows
# and then reverses, indicating stop loss triggers
sweeps = strategy.detect_liquidity_sweeps(df, timeframe)
```

### Order Block Identification
```python
# Identifies supply/demand zones where significant
# buying/selling occurred
order_blocks = strategy.identify_order_blocks(df, timeframe)
```

### Multi-timeframe Confluence
```python
# Combines signals across multiple timeframes
# for stronger confirmation
signals = strategy.get_multi_timeframe_signals(data_dict)
```

## 🔧 Hardware Optimization

### CPU Optimization (7970x)
- **32 Parallel Environments**: Utilizes all CPU cores
- **Large Batch Size**: 128 for efficient training
- **Increased Buffer Size**: 2M for better experience replay
- **Multiple Gradient Steps**: 4 per update for faster learning

### Memory Optimization (128GB RAM)
- **100GB Memory Limit**: Leaves room for system
- **Efficient Data Structures**: Optimized for large datasets
- **Garbage Collection**: Automatic memory management

## 📈 Expected Results

With proper training, the system should achieve:

- **Win Rate**: 80-90%
- **Sharpe Ratio**: 2.0-4.0
- **Max Drawdown**: 5-15%
- **Profit Factor**: 1.5-3.0

## 🚨 Risk Management

- **Position Sizing**: 2% risk per trade
- **Stop Loss**: 2x ATR from entry
- **Take Profit**: 4x ATR from entry
- **Max Positions**: 3 concurrent trades
- **No Commissions**: Focus on strategy performance

## 📝 Logging and Monitoring

### Training Logs
```
2024-01-15 10:30:15 - INFO - Step 50000: Win Rate: 0.823, Sharpe: 2.45, PnL: 15420.50, Trades: 156
2024-01-15 10:35:20 - INFO - New best win rate: 0.847
2024-01-15 10:40:25 - INFO - New best Sharpe ratio: 2.89
```

### Performance Reports
```
DRL Liquidity Sweep Training Report
===================================

Final Performance Metrics:
- Win Rate: 0.847
- Sharpe Ratio: 2.89
- Total PnL: 15420.50
- Max Drawdown: 0.089
- Profit Factor: 2.34
- Total Trades: 156

Target Performance:
- Win Rate Target: 85%
- Current Win Rate: 84.7%
- Status: ACHIEVED
```

## 🔄 Training Workflow

1. **Data Preparation**: Load and validate CSV data
2. **Feature Engineering**: Create 50+ features from price data
3. **Environment Setup**: Create trading environments
4. **Model Training**: Train PPO model with performance tracking
5. **Evaluation**: Test on unseen data
6. **Results Analysis**: Generate performance reports and plots

## 🎛️ Advanced Configuration

### Custom Reward Functions
```python
env_config = EnvironmentConfig(
    reward_type="sharpe_ratio",  # Options: pnl, sharpe_ratio, sortino_ratio
    reward_scale=1.0
)
```

### Feature Engineering
```python
data_config = DataConfig(
    feature_window=100,
    normalize_features=True,
    use_technical_indicators=True
)
```

### Network Architecture
```python
drl_config = DRLConfig(
    hidden_layers=[512, 512, 256, 256],  # Custom network size
    activation="relu"
)
```

## 📊 Performance Analysis

The system generates comprehensive analysis:

### Training Plots
- Win rate progression over time
- Sharpe ratio evolution
- PnL curve
- Trade frequency

### Final Metrics
- Detailed performance statistics
- Risk metrics
- Trade analysis
- Configuration summary

## 🚀 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Prepare data**: Place CSV files in `data/` directory
3. **Load data**: `python data_loader.py --input your_data.csv --split`
4. **Train model**: `python train_model.py --data your_data`
5. **Monitor**: Check logs and TensorBoard
6. **Analyze**: Review performance reports

## ⚠️ Disclaimer

This system is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly on historical data before live trading.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 