import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/vegetables_panel.csv")
df['date'] = pd.to_datetime(df['date'])

colombo = df[df['market'] == 'Colombo City']

plt.figure(figsize=(12, 6))
for veg in colombo['commodity'].unique():
    s = colombo[colombo['commodity'] == veg]
    plt.plot(s['date'], s['price'], label=veg, marker='o', markersize=3)

plt.title("Retail Vegetable Prices — Colombo City")
plt.xlabel("Date")
plt.ylabel("Price (LKR/kg)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("vegetable_prices.png", dpi=150)
print("Chart saved")