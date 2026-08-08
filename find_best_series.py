import pandas as pd

# Use the SAME read line as explore.py — add skiprows=[1] if used it there
df = pd.read_csv("data/wfp_food_prices_lka.csv")
df['date'] = pd.to_datetime(df['date'])

# ---- PART 1: What's actually in the wholesale data ----
ws = df[df['pricetype'] == 'Wholesale']
print("WHOLESALE COMMODITIES:")
print(ws['commodity'].value_counts())
print("\nWHOLESALE MARKETS:")
print(ws['market'].value_counts())
print("\nWHOLESALE DATE RANGE:", ws['date'].min(), "to", ws['date'].max())

# ---- PART 2: Longest retail vegetable series ----
print("\n" + "="*60)
print("LONGEST RETAIL VEGETABLE SERIES")
print("="*60)

vegetables = ['Tomatoes', 'Carrots', 'Cabbage', 'Eggplants', 'Pumpkin',
              'Snake gourd', 'Potatoes (local)', 'Beans']

veg = df[(df['commodity'].isin(vegetables)) &
         (df['pricetype'] == 'Retail') &
         (df['market'] != 'National Average')]

counts = veg.groupby(['commodity', 'market']).agg(
    months=('date', 'nunique'),
    first=('date', 'min'),
    last=('date', 'max')
).reset_index()

print(counts.sort_values('months', ascending=False).head(15).to_string())