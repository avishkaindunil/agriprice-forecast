import pandas as pd

df = pd.read_csv("data/wfp_food_prices_lka.csv")
df['date'] = pd.to_datetime(df['date'])

# --- FILTER ---
vegetables = ['Tomatoes', 'Carrots', 'Cabbage', 'Eggplants', 'Pumpkin',
              'Snake gourd', 'Potatoes (local)', 'Beans']

veg = df[
    (df['commodity'].isin(vegetables)) &
    (df['pricetype'] == 'Retail') &
    (df['unit'] == 'KG') &
    (df['market'] != 'National Average')
].copy()

veg = veg[['date', 'market', 'commodity', 'price']]

# --- COLLAPSE DUPLICATES ---
# Same commodity+market+month can appear more than once. Average them.
veg = veg.groupby(['commodity', 'market', 'date'])['price'].mean().reset_index()

# --- SORT: critical before creating lags ---
veg = veg.sort_values(['commodity', 'market', 'date'])

# --- LAG FEATURES, computed WITHIN each commodity+market group ---
group = veg.groupby(['commodity', 'market'])['price']

veg['lag_1'] = group.shift(1)
veg['lag_2'] = group.shift(2)
veg['lag_3'] = group.shift(3)
veg['rolling_3'] = veg.groupby(['commodity', 'market'])['lag_1'].transform(
    lambda x: x.rolling(3).mean()
)

veg['month'] = veg['date'].dt.month

# --- REPORT ---
print("Rows before dropping incomplete lags:", len(veg))
clean = veg.dropna()
print("Rows usable for modelling:", len(clean))
print("Commodities:", clean['commodity'].nunique())
print("Markets:", clean['market'].nunique())
print("\nDate range:", clean['date'].min(), "to", clean['date'].max())
print("\nSample:")
print(clean.head(10).to_string())

print("\n--- LAG SANITY CHECK ---")
check = clean[(clean['commodity'] == 'Tomatoes') & (clean['market'] == 'Colombo City')]
print(check[['date', 'price', 'lag_1', 'lag_2', 'lag_3', 'rolling_3']].head(6).to_string())

clean.to_csv("data/vegetables_panel.csv", index=False)
print("\nSaved to data/vegetables_panel.csv")