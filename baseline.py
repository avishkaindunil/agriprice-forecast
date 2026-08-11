import pandas as pd

df = pd.read_csv("data/vegetables_panel.csv")
df['date'] = pd.to_datetime(df['date'])

# ---- CHRONOLOGICAL SPLIT ----
# Find the date that sits 80% of the way through the timeline.
all_dates = sorted(df['date'].unique())
cutoff = all_dates[int(len(all_dates) * 0.8)]

train = df[df['date'] < cutoff]
test = df[df['date'] >= cutoff]

print(f"Cutoff date: {cutoff.date()}")
print(f"Train rows: {len(train)}  ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Test rows:  {len(test)}   ({test['date'].min().date()} to {test['date'].max().date()})")

# ---- THE NAIVE BASELINE ----
# Prediction = last month's price = lag_1
test = test.copy()
test['naive_pred'] = test['lag_1']
test['abs_error'] = (test['price'] - test['naive_pred']).abs()

mae = test['abs_error'].mean()
avg_price = test['price'].mean()
mape = (test['abs_error'] / test['price']).mean() * 100

print("\n" + "="*55)
print("NAIVE BASELINE RESULTS (test period only)")
print("="*55)
print(f"MAE:            {mae:.2f} LKR")
print(f"Average price:  {avg_price:.2f} LKR")
print(f"MAPE:           {mape:.1f}%")

# ---- AN EVEN DUMBER BASELINE, for reference ----
# Predict the training-set average price for every single row.
train_mean = train['price'].mean()
mean_mae = (test['price'] - train_mean).abs().mean()
print(f"\n'Always guess the average' MAE: {mean_mae:.2f} LKR")

# ---- WHICH VEGETABLES ARE HARDEST? ----
print("\n" + "="*55)
print("BASELINE ERROR BY COMMODITY")
print("="*55)
by_veg = test.groupby('commodity').agg(
    mae=('abs_error', 'mean'),
    avg_price=('price', 'mean')
)
by_veg['mape_pct'] = (by_veg['mae'] / by_veg['avg_price'] * 100).round(1)
print(by_veg.round(2).sort_values('mape_pct', ascending=False).to_string())

# ---- SAVE THE NUMBER ----
with open("results.txt", "w") as f:
    f.write(f"Cutoff date: {cutoff.date()}\n")
    f.write(f"Test rows: {len(test)}\n")
    f.write(f"Naive baseline MAE: {mae:.2f} LKR\n")
    f.write(f"Naive baseline MAPE: {mape:.1f}%\n")
    f.write(f"Mean-guess MAE: {mean_mae:.2f} LKR\n")

print("\nSaved to results.txt")