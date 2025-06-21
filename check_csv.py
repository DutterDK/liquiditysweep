import pandas as pd

# Read the CSV file
df = pd.read_csv('data/XAUUSD train.csv', nrows=5)

print("Columns found:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print(f"\nTotal rows in file: {len(pd.read_csv('data/XAUUSD train.csv'))}") 
