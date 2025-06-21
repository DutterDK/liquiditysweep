"""
Trading Environment for DRL Liquidity Sweep Trading System

This module implements a Gymnasium-compatible trading environment
for training deep reinforcement learning models on the liquidity sweep strategy.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import random

from ..strategy.liquidity_sweep import LiquiditySweepStrategy
from ..features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class Action(Enum):
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2

@dataclass
class Position:
    """Represents a trading position"""
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    side: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    is_open: bool = True

@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    side: str
    pnl: float
    pnl_pct: float
    duration: pd.Timedelta
    exit_reason: str

class TradingEnvironment(gym.Env):
    """
    Trading environment for liquidity sweep strategy
    """
    
    def __init__(self, data: pd.DataFrame, config: Dict, strategy_config: Dict = None, features_df: pd.DataFrame = None):
        super().__init__()
        
        self.config = config
        self.data = data.copy()
        self.current_step = 0
        self.max_steps = config.get('max_steps', len(data))
        
        # Initialize components
        self.strategy = LiquiditySweepStrategy(strategy_config or {})
        self.feature_engineer = FeatureEngineer(config)
        self.features_df = features_df
        
        # Trading state
        self.initial_balance = config.get('initial_balance', 100000.0)
        self.balance = self.initial_balance
        self.positions: List[Position] = []
        self.closed_trades: List[Trade] = []
        self.max_positions = config.get('max_positions', 3)
        
        # Risk management
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 2.0)
        self.take_profit_atr_multiplier = config.get('take_profit_atr_multiplier', 4.0)
        self.transaction_fee = config.get('transaction_fee', 0.0)
        
        # Observation and action spaces
        self.observation_window = config.get('observation_window', 50)
        self.action_space_type = config.get('action_space_type', 'discrete')
        self.num_actions = config.get('num_actions', 3)
        
        # Reward configuration
        self.reward_type = config.get('reward_type', 'sharpe_ratio')
        self.reward_scale = config.get('reward_scale', 1.0)
        
        # Performance tracking
        self.returns_history = []
        self.equity_curve = []
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        # Setup spaces
        self._setup_spaces()
        
        # Initialize features if not provided
        if self.features_df is None:
            self._initialize_features()
        else:
            # Features are pre-computed, just ensure ATR is available for risk management.
            self._calculate_atr()
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Calculate observation space size
        feature_names = self.feature_engineer.get_feature_names(self.observation_window)
        obs_size = len(feature_names)
        
        # Add position and account information
        if self.config.get('include_position_info', True):
            obs_size += 6  # position info
        if self.config.get('include_account_info', True):
            obs_size += 4  # account info
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action space
        if self.action_space_type == 'discrete':
            self.action_space = spaces.Discrete(self.num_actions)
        else:
            self.action_space = spaces.Box(
                low=-1, 
                high=1, 
                shape=(1,), 
                dtype=np.float32
            )
    
    def _initialize_features(self):
        """Initialize feature engineering"""
        # Generate strategy signals for the entire dataset
        strategy_signals = self.strategy.generate_signals(self.data, 'M1')
        
        # Create features
        self.features_df = self.feature_engineer.create_features(
            self.data, strategy_signals
        )
        
        # Calculate ATR for stop loss and take profit
        self._calculate_atr()
    
    def _calculate_atr(self):
        """Calculate Average True Range for position sizing"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.atr = true_range.rolling(window=14).mean()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset trading state
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = []
        self.closed_trades = []
        self.returns_history = []
        self.equity_curve = [self.initial_balance]
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment"""
        # Execute action
        reward = self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Update positions
        self._update_positions()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps - 1
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action and return reward"""
        current_price = self._get_current_price()
        current_time = self._get_current_time()
        
        # Calculate current equity
        current_equity = self._calculate_equity(current_price)
        
        # Execute action
        if action == Action.BUY.value and len(self.positions) < self.max_positions:
            self._open_long_position(current_price, current_time)
        elif action == Action.SELL.value and len(self.positions) < self.max_positions:
            self._open_short_position(current_price, current_time)
        # Action.HOLD does nothing
        
        # Calculate reward
        reward = self._calculate_reward(current_equity)
        
        return reward
    
    def _open_long_position(self, price: float, time: pd.Timestamp):
        """Open a long position"""
        # Calculate position size based on risk
        atr = self.atr.iloc[self.current_step]
        stop_loss = price - (atr * self.stop_loss_atr_multiplier)
        take_profit = price + (atr * self.take_profit_atr_multiplier)
        
        # Calculate position size
        risk_amount = self.balance * self.risk_per_trade
        position_size = risk_amount / (price - stop_loss)
        
        # Create position
        position = Position(
            entry_price=price,
            entry_time=time,
            size=position_size,
            side='long',
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions.append(position)
        
        # Apply transaction fee
        self.balance -= position_size * price * self.transaction_fee
    
    def _open_short_position(self, price: float, time: pd.Timestamp):
        """Open a short position"""
        # Calculate position size based on risk
        atr = self.atr.iloc[self.current_step]
        stop_loss = price + (atr * self.stop_loss_atr_multiplier)
        take_profit = price - (atr * self.take_profit_atr_multiplier)
        
        # Calculate position size
        risk_amount = self.balance * self.risk_per_trade
        position_size = risk_amount / (stop_loss - price)
        
        # Create position
        position = Position(
            entry_price=price,
            entry_time=time,
            size=position_size,
            side='short',
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions.append(position)
        
        # Apply transaction fee
        self.balance -= position_size * price * self.transaction_fee
    
    def _update_positions(self):
        """Update all open positions"""
        current_price = self._get_current_price()
        current_time = self._get_current_time()
        
        positions_to_close = []
        
        for i, position in enumerate(self.positions):
            if not position.is_open:
                continue
            
            # Check stop loss
            if (position.side == 'long' and current_price <= position.stop_loss) or \
               (position.side == 'short' and current_price >= position.stop_loss):
                self._close_position(i, current_price, current_time, 'stop_loss')
                positions_to_close.append(i)
            
            # Check take profit
            elif (position.side == 'long' and current_price >= position.take_profit) or \
                 (position.side == 'short' and current_price <= position.take_profit):
                self._close_position(i, current_price, current_time, 'take_profit')
                positions_to_close.append(i)
        
        # Close positions in reverse order to maintain indices
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def _close_position(self, position_idx: int, price: float, time: pd.Timestamp, reason: str):
        """Close a position"""
        position = self.positions[position_idx]
        position.is_open = False
        
        # Calculate PnL
        if position.side == 'long':
            pnl = (price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - price) * position.size
        
        # Apply transaction fee
        pnl -= position.size * price * self.transaction_fee
        
        # Update balance
        self.balance += pnl
        
        # Create trade record
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=time,
            entry_price=position.entry_price,
            exit_price=price,
            size=position.size,
            side=position.side,
            pnl=pnl,
            pnl_pct=pnl / (position.entry_price * position.size),
            duration=time - position.entry_time,
            exit_reason=reason
        )
        
        self.closed_trades.append(trade)
        
        # Update returns history
        self.returns_history.append(pnl / self.initial_balance)
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized PnL"""
        equity = self.balance
        
        for position in self.positions:
            if position.is_open:
                if position.side == 'long':
                    unrealized_pnl = (current_price - position.entry_price) * position.size
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.size
                equity += unrealized_pnl
        
        return equity
    
    def _calculate_reward(self, current_equity: float) -> float:
        """Calculate reward based on equity change"""
        # Update equity curve
        self.equity_curve.append(current_equity)
        
        # Update peak equity and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Calculate reward based on type
        if self.reward_type == 'pnl':
            # Simple PnL-based reward
            if len(self.equity_curve) > 1:
                reward = (current_equity - self.equity_curve[-2]) / self.initial_balance
            else:
                reward = 0.0
        
        elif self.reward_type == 'sharpe_ratio':
            # Sharpe ratio-based reward
            if len(self.returns_history) > 1:
                returns = np.array(self.returns_history)
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
                reward = sharpe * self.reward_scale
            else:
                reward = 0.0
        
        elif self.reward_type == 'sortino_ratio':
            # Sortino ratio-based reward
            if len(self.returns_history) > 1:
                returns = np.array(self.returns_history)
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns)
                    sortino = np.mean(returns) / (downside_deviation + 1e-8)
                    reward = sortino * self.reward_scale
                else:
                    reward = np.mean(returns) * self.reward_scale
            else:
                reward = 0.0
        
        else:
            reward = 0.0
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Get feature observation
        observation = self.feature_engineer.create_observation_vector(
            self.features_df, self.current_step, self.observation_window
        )
        
        # Add position information
        if self.config.get('include_position_info', True):
            position_info = self._get_position_info()
            observation = np.concatenate([observation, position_info])
        
        # Add account information
        if self.config.get('include_account_info', True):
            account_info = self._get_account_info()
            observation = np.concatenate([observation, account_info])
        
        return observation.astype(np.float32)
    
    def _get_position_info(self) -> np.ndarray:
        """Get position information"""
        current_price = self._get_current_price()
        
        # Count positions by type
        long_positions = sum(1 for p in self.positions if p.side == 'long' and p.is_open)
        short_positions = sum(1 for p in self.positions if p.side == 'short' and p.is_open)
        
        # Calculate total unrealized PnL
        unrealized_pnl = 0.0
        for position in self.positions:
            if position.is_open:
                if position.side == 'long':
                    unrealized_pnl += (current_price - position.entry_price) * position.size
                else:
                    unrealized_pnl += (position.entry_price - current_price) * position.size
        
        # Position size ratio
        total_position_value = sum(p.size * current_price for p in self.positions if p.is_open)
        position_ratio = total_position_value / self.initial_balance
        
        return np.array([
            long_positions,
            short_positions,
            unrealized_pnl / self.initial_balance,
            position_ratio,
            len(self.positions) / self.max_positions,
            total_position_value
        ], dtype=np.float32)
    
    def _get_account_info(self) -> np.ndarray:
        """Get account information"""
        current_equity = self._calculate_equity(self._get_current_price())
        
        return np.array([
            self.balance / self.initial_balance,
            current_equity / self.initial_balance,
            self.max_drawdown,
            len(self.closed_trades)
        ], dtype=np.float32)
    
    def _get_current_price(self) -> float:
        """Get current price"""
        return self.data.iloc[self.current_step]['close']
    
    def _get_current_time(self) -> pd.Timestamp:
        """Get current timestamp"""
        return self.data.iloc[self.current_step]['timestamp']
    
    def _get_info(self) -> Dict:
        """Get environment information"""
        current_equity = self._calculate_equity(self._get_current_price())
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': current_equity,
            'positions': len(self.positions),
            'trades': len(self.closed_trades),
            'max_drawdown': self.max_drawdown,
            'win_rate': self._calculate_win_rate(),
            'avg_trade': self._calculate_avg_trade(),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.closed_trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.closed_trades if trade.pnl > 0)
        return winning_trades / len(self.closed_trades)
    
    def _calculate_avg_trade(self) -> float:
        """Calculate average trade PnL"""
        if not self.closed_trades:
            return 0.0
        
        return np.mean([trade.pnl for trade in self.closed_trades])
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        return np.mean(returns) / (np.std(returns) + 1e-8)
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        if not self.closed_trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for trade in self.closed_trades if trade.pnl > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades
        avg_win = np.mean([trade.pnl for trade in self.closed_trades if trade.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([trade.pnl for trade in self.closed_trades if trade.pnl < 0]) if losing_trades > 0 else 0
        
        # Risk metrics
        total_pnl = sum(trade.pnl for trade in self.closed_trades)
        returns = [trade.pnl / self.initial_balance for trade in self.closed_trades]
        
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self.max_drawdown
        
        # Profit factor
        gross_profit = sum(trade.pnl for trade in self.closed_trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.closed_trades if trade.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'final_equity': self._calculate_equity(self._get_current_price()),
            'total_return': (self._calculate_equity(self._get_current_price()) - self.initial_balance) / self.initial_balance
        }
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility (Gym API)"""
        np.random.seed(seed)
        random.seed(seed)
        return [seed] 
