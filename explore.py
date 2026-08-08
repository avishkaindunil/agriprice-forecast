import pandas as pd

# Data
FILE_PATH = "data/wfp_food_prices_lka.csv"

df = pd.read_csv(FILE_PATH, skiprows=[1])

print("=" * 50)
print("SHAPE (rows, columns):", df.shape)
print("=" * 50)

print("\nCOLUMN NAMES:")
print(list(df.columns))

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nCOLUMN TYPES AND MISSING VALUES:")
print(df.info())

print("\nCOMMODITIES AVAILABLE:")
print(df['commodity'].unique())

print("\nMARKETS AVAILABLE:")
print(df['market'].unique())

print("\nDATE RANGE:")
print(df['date'].min(), "to", df['date'].max())