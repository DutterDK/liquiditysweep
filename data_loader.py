"""
Data Loader for DRL Liquidity Sweep Trading System

This module loads and preprocesses CSV data files for training the DRL model.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

from config.config import data_config

logger = logging.getLogger(__name__)

class CSVDataLoader:
    """
    Data loader for CSV files in MT5 format
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = "data"
        
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """
        Load CSV data in MT5 format
        
        Expected format:
        DATE,TIME,OPEN,HIGH,LOW,CLOSE,TICKVOL,VOL,SPREAD
        2024.01.02,01:00:00,2063.570,2064.650,2063.060,2064.280,100,0,350
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        try:
            # Try reading as a standard CSV first
            df = pd.read_csv(filepath)
            # If only one column and tab characters are present, split it
            if len(df.columns) == 1 and '\t' in df.columns[0]:
                first_col = df.columns[0]
                if '\t' in str(df.iloc[0][first_col]):
                    split_data = df[first_col].str.split('\t', expand=True)
                    header_row = split_data.iloc[0]
                    split_data = split_data.iloc[1:]
                    split_data.columns = header_row
                    split_data.columns = [col.replace('<', '').replace('>', '') for col in split_data.columns]
                    df = split_data.reset_index(drop=True)
            logger.info(f"Loaded {len(df)} records from {filename}")
            
            # Validate and preprocess
            df = self._preprocess_data(df)
            
            # After loading, ensure timestamp is datetime if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data
        """
        # Ensure timestamp is datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Handle case where all data is in one column (tab-separated)
        if len(df.columns) == 1:
            # Split the single column by tabs
            first_col = df.columns[0]
            if '\t' in str(df.iloc[0][first_col]):
                # Split the data
                split_data = df[first_col].str.split('\t', expand=True)
                # Set column names based on the header
                header_row = split_data.iloc[0]
                split_data = split_data.iloc[1:]  # Remove header row
                split_data.columns = header_row
                # Clean column names (remove < > symbols)
                split_data.columns = [col.replace('<', '').replace('>', '') for col in split_data.columns]
                df = split_data.reset_index(drop=True)
        
        # Rename columns to match expected format
        column_mapping = {
            'DATE': 'date',
            'TIME': 'time',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close',
            'TICKVOL': 'tickvol',
            'VOL': 'vol',
            'SPREAD': 'spread'
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Combine date and time into timestamp
        if 'date' in df.columns and 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        elif 'timestamp' not in df.columns:
            # If no date/time columns, create a dummy timestamp
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'tickvol']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Validate data quality
        self._validate_data(df)
        
        logger.info(f"Preprocessed data: {len(df)} records")
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality
        """
        # Check for missing values
        missing_counts = df[['open', 'high', 'low', 'close', 'tickvol']].isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values found: {missing_counts.to_dict()}")
        
        # Check for invalid prices
        invalid_prices = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        )
        
        if invalid_prices.any():
            logger.error(f"Invalid price data found: {invalid_prices.sum()} records")
            return False
        
        # Check for duplicate timestamps
        if df['timestamp'].duplicated().any():
            logger.warning("Duplicate timestamps found")
        
        # Check data continuity
        time_diff = df['timestamp'].diff()
        expected_diff = pd.Timedelta(minutes=1)
        gaps = time_diff[time_diff > expected_diff * 2]
        if not gaps.empty:
            logger.warning(f"Data gaps found: {len(gaps)} gaps")
        
        logger.info("Data validation completed successfully")
        return True
    
    def split_data(self, df: pd.DataFrame, train_split: float = 0.7, 
                  validation_split: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        """
        total_records = len(df)
        train_end = int(total_records * train_split)
        validation_end = int(total_records * (train_split + validation_split))
        
        train_data = df.iloc[:train_end].copy()
        validation_data = df.iloc[train_end:validation_end].copy()
        test_data = df.iloc[validation_end:].copy()
        
        splits = {
            'train': train_data,
            'validation': validation_data,
            'test': test_data
        }
        
        logger.info(f"Data split: Train={len(train_data)}, Validation={len(validation_data)}, Test={len(test_data)}")
        
        return splits
    
    def save_splits(self, splits: Dict[str, pd.DataFrame], base_filename: str):
        """
        Save data splits to files
        """
        os.makedirs(self.data_dir, exist_ok=True)
        
        for split_name, split_data in splits.items():
            filename = f"{base_filename}_{split_name}.csv"
            filepath = os.path.join(self.data_dir, filename)
            print(f"[DEBUG] Saving {split_name} split to {filepath} with {len(split_data)} records...")
            split_data.to_csv(filepath, index=False)
            print(f"[DEBUG] Saved {split_name} split to {filepath}")
            logger.info(f"Saved {split_name} data to {filepath}: {len(split_data)} records")
    
    def load_splits(self, base_filename: str) -> Dict[str, pd.DataFrame]:
        """
        Load data splits from files
        """
        splits = {}
        
        for split_name in ['train', 'validation', 'test']:
            filename = f"{base_filename}_{split_name}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                df = self.load_csv_data(filename)
                splits[split_name] = df
            else:
                logger.warning(f"Split file not found: {filename}")
        
        return splits
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the dataset
        """
        info = {
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'symbol_info': {
                'price_range': {
                    'min': df['low'].min(),
                    'max': df['high'].max()
                },
                'avg_volume': df['tickvol'].mean(),
                'avg_spread': df['spread'].mean() if 'spread' in df.columns else 0
            },
            'data_quality': {
                'missing_values': df[['open', 'high', 'low', 'close', 'tickvol']].isnull().sum().to_dict(),
                'duplicate_timestamps': df['timestamp'].duplicated().sum(),
                'gaps': len(df['timestamp'].diff()[df['timestamp'].diff() > pd.Timedelta(minutes=2)])
            }
        }
        
        return info

def main():
    """Main function for data loading"""
    parser = argparse.ArgumentParser(description='Load and preprocess CSV data')
    parser.add_argument('--input', required=True, help='Input CSV filename')
    parser.add_argument('--output', help='Output base filename for splits')
    parser.add_argument('--split', action='store_true', help='Split data into train/validation/test')
    parser.add_argument('--info', action='store_true', help='Show data information')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = CSVDataLoader(data_config.__dict__)
    
    try:
        # Load data
        df = loader.load_csv_data(args.input)
        
        # Show data information
        if args.info:
            info = loader.get_data_info(df)
            print("\nData Information:")
            print(f"Total records: {info['total_records']}")
            print(f"Date range: {info['date_range']['start']} to {info['date_range']['end']}")
            print(f"Price range: {info['symbol_info']['price_range']['min']:.5f} - {info['symbol_info']['price_range']['max']:.5f}")
            print(f"Average volume: {info['symbol_info']['avg_volume']:.2f}")
            print(f"Missing values: {info['data_quality']['missing_values']}")
            print(f"Duplicate timestamps: {info['data_quality']['duplicate_timestamps']}")
            print(f"Data gaps: {info['data_quality']['gaps']}")
        
        # Split data if requested
        if args.split:
            output_name = args.output or args.input.replace('.csv', '')
            splits = loader.split_data(df)
            loader.save_splits(splits, output_name)
            
            print(f"\nData splits saved:")
            for split_name, split_data in splits.items():
                print(f"  {split_name}: {len(split_data)} records")
        
        print(f"\nData loaded successfully: {len(df)} records")
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    main() 