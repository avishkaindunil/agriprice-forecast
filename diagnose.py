import pandas as pd

df = pd.read_csv("data/wfp_food_prices_lka.csv")

print("PRICE TYPES:")
print(df['pricetype'].value_counts())

print("\nPRICE FLAGS:")
print(df['priceflag'].value_counts())

print("\nUNITS:")
print(df['unit'].value_counts())

print("\nCATEGORIES:")
print(df['category'].value_counts())