"""
Feature Engineering for DRL Liquidity Sweep Trading System

This module creates features from price data and strategy signals
for the deep reinforcement learning model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import logging

from src.strategy.liquidity_sweep import LiquiditySweepStrategy

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for liquidity sweep strategy
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_window = config.get('feature_window', 100)
        self.normalize_features = config.get('normalize_features', True)
        self.use_technical_indicators = config.get('use_technical_indicators', True)
        
        # Feature scalers
        self.price_scaler = None
        self.volume_scaler = None
        self.indicator_scalers = {}
        
        # Get all lookback periods from the config to determine max lookback
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_sign = self.config.get('macd_sign', 9)
        self.atr_period = self.config.get('atr_period', 14)
        self.stoch_period = self.config.get('stoch_period', 14)
        self.adx_period = self.config.get('adx_period', 14)
        self.ema_periods = self.config.get('ema_periods', [12, 26, 50, 100, 200])

        self.strategy = LiquiditySweepStrategy({}) # Use default strategy config

        # Calculate the maximum lookback period required by any indicator
        self.max_lookback_period = max(
            self.rsi_period,
            self.macd_slow + self.macd_sign,
            self.atr_period,
            self.stoch_period,
            self.adx_period,
            max(self.ema_periods)
        )
        # Add a buffer for safety
        self.max_lookback_period += 5 
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all features for the given dataframe.
        """
        logger.info(f"Starting feature calculation for {len(df)} records...")
        
        # Ensure the index is a datetime object for time-based features
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # --- Basic Price and Volume Features ---
        features = self._add_price_volume_features(df.copy())

        # --- Technical Indicators ---
        features = self._add_technical_indicators(features)
        
        # --- Strategy-Specific Signals as Features ---
        logger.info("Generating strategy signals to use as features...")
        strategy_signals = self.strategy.generate_signals(df.copy(), 'M1') # Pass a copy to be safe
        
        # Combine strategy signals into the feature dataframe
        for key, value in strategy_signals.items():
            if isinstance(value, pd.Series):
                features[key] = value.reindex(features.index).fillna(False) # Align and fill
            elif isinstance(value, list) and value: # e.g., buy_signals, sell_signals
                 strength_series = pd.Series(0.0, index=features.index)
                 for signal in value:
                     ts = pd.to_datetime(signal['timestamp'])
                     if ts in strength_series.index:
                         strength_series.loc[ts] = signal.get('strength', 0.0)
                 features[key + '_strength'] = strength_series
            else: # For simple scalar values like trend
                features[key] = value

        # --- Time-Based Features ---
        features = self._add_time_features(features)

        # --- Clean up ---
        original_len = len(features)
        # Use the calculated max_lookback_period to drop initial rows
        features = features.iloc[self.max_lookback_period:]
        
        # Convert all boolean features to integer (0 or 1)
        for col in features.columns:
            if features[col].dtype == 'bool':
                features[col] = features[col].astype(int)

        # Forward-fill any remaining NaNs, then backfill
        features.fillna(method='ffill', inplace=True)
        features.fillna(method='bfill', inplace=True)
        features.dropna(inplace=True) # Drop any that remain

        logger.info(f"Feature calculation complete. Dropped {original_len - len(features)} rows.")
        logger.info(f"Final feature set has {len(features.columns)} features and {len(features)} records.")

        return features

    def _add_price_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic price-based features
        """
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['close'] / df['open']
        
        # Price ranges
        df['candle_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Price position relative to moving averages
        df['price_vs_sma5'] = df['close'] / df['sma_5'] - 1
        df['price_vs_sma10'] = df['close'] / df['sma_10'] - 1
        df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
        df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
        
        # Volatility measures
        df['price_volatility'] = df['price_change'].rolling(window=20).std()
        df['range_volatility'] = df['candle_range'].rolling(window=20).std()
        
        # Support and resistance levels
        df['resistance_level'] = df['high'].rolling(window=20).max()
        df['support_level'] = df['low'].rolling(window=20).min()
        df['price_vs_resistance'] = df['close'] / df['resistance_level'] - 1
        df['price_vs_support'] = df['close'] / df['support_level'] - 1
        
        # Volume indicators
        df['volume_sma'] = df['tickvol'].rolling(window=20).mean()
        df['volume_ratio'] = df['tickvol'] / df['volume_sma']
        
        # Volume price trend
        df['volume_price_trend'] = (df['tickvol'] * df['price_change']).rolling(window=10).sum()
        
        # Volume weighted average price
        df['vwap'] = (df['tickvol'] * (df['high'] + df['low'] + df['close']) / 3).rolling(window=20).sum() / df['tickvol'].rolling(window=20).sum()
        df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
        
        # Volume momentum
        df['volume_momentum'] = df['tickvol'].pct_change()
        df['volume_acceleration'] = df['volume_momentum'].pct_change()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based features
        """
        # Average True Range
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        df['atr_ratio'] = df['candle_range'] / df['atr']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators
        """
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        df['rsi_sma'] = df['rsi'].rolling(window=10).mean()
        df['rsi_momentum'] = df['rsi'] - df['rsi_sma']
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # EMA indicators
        ema_12 = EMAIndicator(close=df['close'], window=12)
        ema_26 = EMAIndicator(close=df['close'], window=26)
        df['ema_12'] = ema_12.ema_indicator()
        df['ema_26'] = ema_26.ema_indicator()
        df['ema_cross'] = df['ema_12'] - df['ema_26']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features
        """
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        # Market session features
        df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['is_tokyo_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        
        # Weekend gap
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        return df
    
    def create_observation_vector(self, df: pd.DataFrame, current_idx: int, 
                                window_size: int = 50) -> np.ndarray:
        """
        Create observation vector for the DRL model
        """
        # Get the window of data
        start_idx = max(0, current_idx - window_size + 1)
        window_data = df.iloc[start_idx:current_idx + 1]
        
        # Select features for the model
        feature_cols = [
            'open', 'high', 'low', 'close', 'tickvol',
            'price_change', 'high_low_ratio', 'open_close_ratio',
            'candle_range', 'body_size', 'upper_shadow', 'lower_shadow',
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
            'price_volatility', 'range_volatility',
            'price_vs_resistance', 'price_vs_support',
            'volume_ratio', 'volume_price_trend', 'price_vs_vwap',
            'volume_momentum', 'volume_acceleration',
            'rsi', 'rsi_momentum', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position', 'stoch_k', 'stoch_d',
            'atr_ratio', 'ema_cross',
            'is_london_session', 'is_ny_session', 'is_tokyo_session'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in window_data.columns]
        
        # Create feature matrix
        feature_matrix = window_data[available_cols].values
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # Flatten the matrix
        observation = feature_matrix.flatten()
        
        return observation
    
    def get_feature_names(self, window_size: int = 50) -> List[str]:
        """
        Get feature names for the observation vector
        """
        feature_cols = [
            'open', 'high', 'low', 'close', 'tickvol',
            'price_change', 'high_low_ratio', 'open_close_ratio',
            'candle_range', 'body_size', 'upper_shadow', 'lower_shadow',
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
            'price_volatility', 'range_volatility',
            'price_vs_resistance', 'price_vs_support',
            'volume_ratio', 'volume_price_trend', 'price_vs_vwap',
            'volume_momentum', 'volume_acceleration',
            'rsi', 'rsi_momentum', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position', 'stoch_k', 'stoch_d',
            'atr_ratio', 'ema_cross',
            'is_london_session', 'is_ny_session', 'is_tokyo_session'
        ]
        
        # Create feature names for each timestep
        feature_names = []
        for t in range(window_size):
            for col in feature_cols:
                feature_names.append(f"{col}_t{t}")
        
        return feature_names 
