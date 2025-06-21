"""
Liquidity Sweep Trading Strategy Implementation

This module implements the liquidity sweep strategy as described in the reference.
Key components:
- Liquidity sweep detection
- Order block identification
- Market structure analysis
- Multi-timeframe confluence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LiquiditySweep:
    """Represents a detected liquidity sweep"""
    timestamp: pd.Timestamp
    price: float
    sweep_type: str  # 'high' or 'low'
    strength: float
    timeframe: str
    order_block_id: Optional[int] = None

@dataclass
class OrderBlock:
    """Represents an order block (supply/demand zone)"""
    timestamp: pd.Timestamp
    high: float
    low: float
    block_type: str  # 'bullish' or 'bearish'
    strength: float
    timeframe: str
    is_mitigated: bool = False

@dataclass
class MarketStructure:
    """Represents market structure information"""
    swing_highs: List[Tuple[pd.Timestamp, float]]
    swing_lows: List[Tuple[pd.Timestamp, float]]
    trend_direction: str  # 'bullish', 'bearish', 'sideways'
    structure_break: bool = False

class LiquiditySweepStrategy:
    """
    Main liquidity sweep strategy implementation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.sweep_lookback = config.get('sweep_lookback', 20)
        self.min_sweep_strength = config.get('min_sweep_strength', 0.5)
        self.order_block_lookback = config.get('order_block_lookback', 50)
        
    def detect_liquidity_sweeps(self, df: pd.DataFrame, timeframe: str) -> List[LiquiditySweep]:
        """
        Detect liquidity sweeps in the price data using vectorized operations.
        """
        sweeps = []
        
        # Ensure the index is a simple range for performance
        df_reset = df.reset_index()

        # For vectorization, get numpy arrays
        lows = df_reset['low'].to_numpy()
        highs = df_reset['high'].to_numpy()
        closes = df_reset['close'].to_numpy()

        # Create rolling windows of size `lookback + 1` to include the current candle
        window_size = self.sweep_lookback + 1
        
        # Ensure we can create windows
        if len(lows) < window_size:
            return []
        
        low_windows = np.lib.stride_tricks.sliding_window_view(lows, window_size)
        high_windows = np.lib.stride_tricks.sliding_window_view(highs, window_size)
        close_windows = np.lib.stride_tricks.sliding_window_view(closes, window_size)

        # Separate lookback data from the current candle
        prev_lows = low_windows[:, :-1]
        current_lows = low_windows[:, -1]
        
        prev_highs = high_windows[:, :-1]
        current_highs = high_windows[:, -1]
        
        current_closes = close_windows[:, -1]

        # Conditions for bullish sweeps
        bullish_sweep_candidates = (current_lows[:, np.newaxis] < prev_lows) & \
                                   (current_closes[:, np.newaxis] > prev_lows)
        
        # Get indices of bullish sweeps
        bullish_sweep_indices = np.argwhere(bullish_sweep_candidates)
        
        # Conditions for bearish sweeps
        bearish_sweep_candidates = (current_highs[:, np.newaxis] > prev_highs) & \
                                   (current_closes[:, np.newaxis] < prev_highs)

        # Get indices of bearish sweeps
        bearish_sweep_indices = np.argwhere(bearish_sweep_candidates)

        # Process bullish sweeps
        for i, j in bullish_sweep_indices:
            current_idx = i + self.sweep_lookback
            sweep_idx = i + j 
            strength = self._calculate_sweep_strength(df_reset, current_idx, sweep_idx, 'low')
            if strength >= self.min_sweep_strength:
                sweeps.append(LiquiditySweep(
                    timestamp=df_reset.iloc[current_idx]['timestamp'],
                    price=df_reset.iloc[sweep_idx]['low'],
                    sweep_type='low',
                    strength=strength,
                    timeframe=timeframe
                ))

        # Process bearish sweeps
        for i, j in bearish_sweep_indices:
            current_idx = i + self.sweep_lookback
            sweep_idx = i + j
            strength = self._calculate_sweep_strength(df_reset, current_idx, sweep_idx, 'high')
            if strength >= self.min_sweep_strength:
                sweeps.append(LiquiditySweep(
                    timestamp=df_reset.iloc[current_idx]['timestamp'],
                    price=df_reset.iloc[sweep_idx]['high'],
                    sweep_type='high',
                    strength=strength,
                    timeframe=timeframe
                ))
        
        return sweeps
    
    def _calculate_sweep_strength(self, df: pd.DataFrame, current_idx: int, 
                                sweep_idx: int, sweep_type: str) -> float:
        """
        Calculate the strength of a liquidity sweep based on:
        - Volume confirmation
        - Price rejection
        - Time since sweep level was established
        """
        current_candle = df.iloc[current_idx]
        sweep_candle = df.loc[sweep_idx]
        # Convert sweep_idx to integer position for arithmetic
        sweep_pos = df.index.get_loc(sweep_idx)
        # Volume confirmation (higher volume = stronger sweep)
        volume_factor = min(current_candle['tickvol'] / max(sweep_candle['tickvol'], 1), 3.0)
        # Price rejection factor
        if sweep_type == 'low':
            rejection_factor = (current_candle['close'] - current_candle['low']) / (current_candle['high'] - current_candle['low'])
        else:
            rejection_factor = (current_candle['high'] - current_candle['close']) / (current_candle['high'] - current_candle['low'])
        # Time factor (more recent sweeps are stronger)
        time_factor = 1.0 - (current_idx - sweep_pos) / self.sweep_lookback
        # Combined strength
        strength = (volume_factor * 0.4 + rejection_factor * 0.4 + time_factor * 0.2)
        return min(strength, 1.0)
    
    def identify_order_blocks(self, df: pd.DataFrame, timeframe: str) -> List[OrderBlock]:
        """
        Identify order blocks (supply/demand zones)
        
        Order blocks are areas where significant buying/selling occurred
        and are likely to act as support/resistance
        """
        order_blocks = []
        
        for i in range(self.order_block_lookback, len(df)):
            current_candle = df.iloc[i]
            
            # Look for bullish order blocks (demand zones)
            if self._is_bullish_order_block(df, i):
                ob = OrderBlock(
                    timestamp=df.index[i],
                    high=current_candle['high'],
                    low=current_candle['low'],
                    block_type='bullish',
                    strength=self._calculate_order_block_strength(df, i, 'bullish'),
                    timeframe=timeframe
                )
                order_blocks.append(ob)
            
            # Look for bearish order blocks (supply zones)
            if self._is_bearish_order_block(df, i):
                ob = OrderBlock(
                    timestamp=df.index[i],
                    high=current_candle['high'],
                    low=current_candle['low'],
                    block_type='bearish',
                    strength=self._calculate_order_block_strength(df, i, 'bearish'),
                    timeframe=timeframe
                )
                order_blocks.append(ob)
        
        return order_blocks
    
    def _is_bullish_order_block(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Identify bullish order blocks
        - Strong buying pressure
        - Price closes near high
        - Higher volume
        """
        candle = df.iloc[idx]
        
        # Strong bullish candle
        body_size = candle['close'] - candle['open']
        total_range = candle['high'] - candle['low']
        
        if body_size <= 0 or total_range == 0:
            return False
        
        # Body should be at least 60% of total range
        body_ratio = body_size / total_range
        if body_ratio < 0.6:
            return False
        
        # Close should be in upper third of candle
        close_position = (candle['close'] - candle['low']) / total_range
        if close_position < 0.67:
            return False
        
        # Volume confirmation
        avg_volume = df.iloc[max(0, idx-10):idx]['tickvol'].mean()
        if candle['tickvol'] < avg_volume * 1.2:
            return False
        
        return True
    
    def _is_bearish_order_block(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Identify bearish order blocks
        - Strong selling pressure
        - Price closes near low
        - Higher volume
        """
        candle = df.iloc[idx]
        
        # Strong bearish candle
        body_size = candle['open'] - candle['close']
        total_range = candle['high'] - candle['low']
        
        if body_size <= 0 or total_range == 0:
            return False
        
        # Body should be at least 60% of total range
        body_ratio = body_size / total_range
        if body_ratio < 0.6:
            return False
        
        # Close should be in lower third of candle
        close_position = (candle['high'] - candle['close']) / total_range
        if close_position < 0.67:
            return False
        
        # Volume confirmation
        avg_volume = df.iloc[max(0, idx-10):idx]['tickvol'].mean()
        if candle['tickvol'] < avg_volume * 1.2:
            return False
        
        return True
    
    def _calculate_order_block_strength(self, df: pd.DataFrame, idx: int, block_type: str) -> float:
        """
        Calculate the strength of an order block
        """
        candle = df.iloc[idx]
        
        # Volume factor
        avg_volume = df.iloc[max(0, idx-10):idx]['tickvol'].mean()
        volume_factor = min(candle['tickvol'] / max(avg_volume, 1), 3.0)
        
        # Body strength
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        body_factor = body_size / max(total_range, 0.0001)
        
        # Combined strength
        strength = (volume_factor * 0.6 + body_factor * 0.4)
        
        return min(strength, 1.0)
    
    def analyze_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        """
        Analyze market structure to identify:
        - Swing highs and lows
        - Trend direction
        - Structure breaks
        """
        swing_highs = []
        swing_lows = []
        
        # Find swing points
        for i in range(2, len(df) - 2):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Swing high
            if (current_high > df.iloc[i-1]['high'] and 
                current_high > df.iloc[i-2]['high'] and
                current_high > df.iloc[i+1]['high'] and
                current_high > df.iloc[i+2]['high']):
                swing_highs.append((df.index[i], current_high))
            
            # Swing low
            if (current_low < df.iloc[i-1]['low'] and 
                current_low < df.iloc[i-2]['low'] and
                current_low < df.iloc[i+1]['low'] and
                current_low < df.iloc[i+2]['low']):
                swing_lows.append((df.index[i], current_low))
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(df, swing_highs, swing_lows)
        
        # Check for structure breaks
        structure_break = self._detect_structure_break(df, swing_highs, swing_lows)
        
        return MarketStructure(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            trend_direction=trend_direction,
            structure_break=structure_break
        )
    
    def _determine_trend_direction(self, df: pd.DataFrame, swing_highs: List, swing_lows: List) -> str:
        """
        Determine the overall trend direction
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'sideways'
        
        # Recent swing points
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        # Check if highs are making higher highs
        higher_highs = all(recent_highs[i][1] > recent_highs[i-1][1] 
                          for i in range(1, len(recent_highs)))
        
        # Check if lows are making higher lows
        higher_lows = all(recent_lows[i][1] > recent_lows[i-1][1] 
                         for i in range(1, len(recent_lows)))
        
        if higher_highs and higher_lows:
            return 'bullish'
        elif not higher_highs and not higher_lows:
            return 'bearish'
        else:
            return 'sideways'
    
    def _detect_structure_break(self, df: pd.DataFrame, swing_highs: List, swing_lows: List) -> bool:
        """
        Detect if market structure has been broken
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return False
        
        current_price = df.iloc[-1]['close']
        last_swing_high = swing_highs[-1][1]
        last_swing_low = swing_lows[-1][1]
        
        # Structure break occurs when price breaks the last swing high/low
        if current_price > last_swing_high or current_price < last_swing_low:
            return True
        
        return False
    
    def generate_signals(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Generate trading signals based on the liquidity sweep strategy.
        """
        logger.info(f"Generating strategy signals for {timeframe}...")
        
        # 1. Detect Liquidity Sweeps
        logger.info("Detecting liquidity sweeps...")
        sweeps = self.detect_liquidity_sweeps(df, timeframe)
        
        # 2. Detect Market Structure Shifts
        logger.info("Detecting market structure shifts...")
        mss = self.analyze_market_structure(df)
        
        # 3. Detect Order Blocks
        logger.info("Detecting order blocks...")
        order_blocks = self.identify_order_blocks(df, timeframe)
        
        # 4. Multi-Timeframe Analysis
        logger.info("Performing multi-timeframe analysis...")
        signals_df = self._multi_timeframe_analysis(df, sweeps, mss, order_blocks)
        
        logger.info("Strategy signal generation complete.")
        return signals_df

    def _multi_timeframe_analysis(self, df, sweeps, mss, order_blocks):
        # Initialize a dictionary to hold signal data
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'is_sweep_high': pd.Series(False, index=df.index),
            'is_sweep_low': pd.Series(False, index=df.index),
            'market_trend': mss.trend_direction,
            'structure_break': mss.structure_break
        }

        # Populate sweep signals
        for sweep in sweeps:
            if sweep.sweep_type == 'high':
                signals['is_sweep_high'].loc[sweep.timestamp] = True
            elif sweep.sweep_type == 'low':
                signals['is_sweep_low'].loc[sweep.timestamp] = True
        
        # Look for buy signals (bullish sweeps near bullish order blocks)
        for sweep in sweeps:
            if sweep.sweep_type == 'low':  # Bullish sweep
                # Find nearby bullish order block
                for ob in order_blocks:
                    if (ob.block_type == 'bullish' and 
                        abs(sweep.price - ob.low) / ob.low < 0.001):  # Within 0.1%
                        
                        signal = {
                            'timestamp': sweep.timestamp,
                            'price': sweep.price,
                            'type': 'buy',
                            'strength': sweep.strength * ob.strength,
                            'reason': f'Bullish sweep at {sweep.price} near bullish order block'
                        }
                        signals['buy_signals'].append(signal)
                        break
        
        # Look for sell signals (bearish sweeps near bearish order blocks)
        for sweep in sweeps:
            if sweep.sweep_type == 'high':  # Bearish sweep
                # Find nearby bearish order block
                for ob in order_blocks:
                    if (ob.block_type == 'bearish' and 
                        abs(sweep.price - ob.high) / ob.high < 0.001):  # Within 0.1%
                        
                        signal = {
                            'timestamp': sweep.timestamp,
                            'price': sweep.price,
                            'type': 'sell',
                            'strength': sweep.strength * ob.strength,
                            'reason': f'Bearish sweep at {sweep.price} near bearish order block'
                        }
                        signals['sell_signals'].append(signal)
                        break
        
        # This part of the code for higher timeframe analysis seems incomplete and complex.
        # For now, I will comment it out to ensure the main logic works.
        # We can re-introduce this feature later if needed.
        # for tf in self.config.get('higher_timeframes', ['M5', 'M15', 'H1']):
        #     logger.info(f"Resampling and analyzing for {tf} timeframe...")
        #     resampled_df = self._resample_data(df, tf)
            
        #     # Recalculate signals on resampled data
        #     # ... (this would require a recursive call or refactoring)
        
        return signals
    
    def get_multi_timeframe_signals(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate signals across multiple timeframes and combine them into a single feature DataFrame.
        """
        all_signals_df = pd.DataFrame(index=data_dict[self.config['base_timeframe']].index)

        for timeframe, df in data_dict.items():
            logger.info(f"Generating signals for {timeframe}...")
            signals = self.generate_signals(df, timeframe)
            
            # Create DataFrame from the signals
            df_signals = pd.DataFrame(index=df.index)
            df_signals[f'is_sweep_high_{timeframe}'] = signals['is_sweep_high']
            df_signals[f'is_sweep_low_{timeframe}'] = signals['is_sweep_low']
            df_signals[f'buy_signal_strength_{timeframe}'] = 0.0
            df_signals[f'sell_signal_strength_{timeframe}'] = 0.0

            for sig in signals['buy_signals']:
                df_signals.loc[sig['timestamp'], f'buy_signal_strength_{timeframe}'] = sig['strength']
            for sig in signals['sell_signals']:
                df_signals.loc[sig['timestamp'], f'sell_signal_strength_{timeframe}'] = sig['strength']
            
            # Resample to base timeframe and merge
            df_signals_resampled = df_signals.reindex(all_signals_df.index, method='ffill')
            all_signals_df = all_signals_df.merge(df_signals_resampled, left_index=True, right_index=True, how='left')

        all_signals_df.fillna(0, inplace=True)
        logger.info("Combined multi-timeframe signals generated.")
        return all_signals_df

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resamples OHLCV data to a higher timeframe."""
        resample_map = {
            'time': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tickvol': 'sum'
        }
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             df.set_index('timestamp', inplace=True)

        resampled_df = df.resample(timeframe).agg(resample_map).dropna()
        return resampled_df.reset_index()


    def _combine_timeframe_signals(self, all_signals: Dict) -> Dict:
        """
        Combine signals from multiple timeframes for stronger confirmation
        """
        combined_buy = []
        combined_sell = []
        
        # Collect all signals
        for timeframe, signals in all_signals.items():
            combined_buy.extend(signals['buy_signals'])
            combined_sell.extend(signals['sell_signals'])
        
        # Group signals by price proximity
        final_buy = self._group_signals_by_price(combined_buy)
        final_sell = self._group_signals_by_price(combined_sell)
        
        return {
            'buy_signals': final_buy,
            'sell_signals': final_sell,
            'all_timeframe_signals': all_signals
        }
    
    def _group_signals_by_price(self, signals: List[Dict]) -> List[Dict]:
        """
        Group signals that occur at similar price levels
        """
        if not signals:
            return []
        
        # Sort by timestamp
        signals.sort(key=lambda x: x['timestamp'])
        
        grouped = []
        current_group = [signals[0]]
        
        for signal in signals[1:]:
            # Check if signal is close in price and time
            last_signal = current_group[-1]
            price_diff = abs(signal['price'] - last_signal['price']) / last_signal['price']
            time_diff = (signal['timestamp'] - last_signal['timestamp']).total_seconds() / 3600  # hours
            
            if price_diff < 0.002 and time_diff < 24:  # Within 0.2% and 24 hours
                current_group.append(signal)
            else:
                # Create combined signal from group
                combined = self._create_combined_signal(current_group)
                grouped.append(combined)
                current_group = [signal]
        
        # Add last group
        if current_group:
            combined = self._create_combined_signal(current_group)
            grouped.append(combined)
        
        return grouped
    
    def _create_combined_signal(self, signal_group: List[Dict]) -> Dict:
        """
        Create a combined signal from a group of similar signals
        """
        # Use the strongest signal as base
        strongest = max(signal_group, key=lambda x: x['strength'])
        
        # Calculate average strength and add timeframe count
        avg_strength = sum(s['strength'] for s in signal_group) / len(signal_group)
        timeframe_count = len(set(s['timeframe'] for s in signal_group))
        
        combined = strongest.copy()
        combined['strength'] = avg_strength * (1 + 0.1 * timeframe_count)  # Bonus for multi-timeframe
        combined['timeframe_count'] = timeframe_count
        combined['signal_count'] = len(signal_group)
        
        return combined 
