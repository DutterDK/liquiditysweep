import pandas as pd
import numpy as np

# Define the correct column names
colnames = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'VOL', 'SPREAD']

# Read the file, skipping the first row (which is the header in the file)
df = pd.read_csv('data/XAUUSD train.csv', sep='\t', header=None, names=colnames, skiprows=1)

print(f"Loaded shape: {df.shape}")
print("First few rows:")
print(df.head())

# Create timestamp
df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])

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
df = df.rename(columns=column_mapping)

# Select required columns
final_df = df[['timestamp', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']]

# Sort by timestamp
final_df = final_df.sort_values('timestamp').reset_index(drop=True)

# Remove duplicates
final_df = final_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

print(f"\nFinal data shape: {final_df.shape}")
print(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
print(f"Price range: {final_df['low'].min():.2f} - {final_df['high'].max():.2f}")

# Save the processed data
final_df.to_csv('data/XAUUSD_processed.csv', index=False)
print("\nProcessed data saved to: data/XAUUSD_processed.csv")

# Show first few rows
print("\nFirst few rows:")
print(final_df.head())

# Show data info
print("\nData info:")
print(final_df.info()) 