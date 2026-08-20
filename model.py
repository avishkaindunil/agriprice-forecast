import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/vegetables_panel.csv")
df['date'] = pd.to_datetime(df['date'])

# ---- ONE-HOT ENCODE the text columns ----
encoded = pd.get_dummies(df, columns=['commodity', 'market'], drop_first=False)

# ---- SAME CHRONOLOGICAL SPLIT AS DAY 3 ----
all_dates = sorted(df['date'].unique())
cutoff = all_dates[int(len(all_dates) * 0.8)]

train = encoded[encoded['date'] < cutoff]
test = encoded[encoded['date'] >= cutoff]

# Everything except date and price is a feature
feature_cols = [c for c in encoded.columns if c not in ['date', 'price']]

X_train, y_train = train[feature_cols], train['price']
X_test, y_test = test[feature_cols], test['price']

print(f"Training rows: {len(X_train)},  Test rows: {len(X_test)}")
print(f"Number of features: {len(feature_cols)}")

# ---- THE BASELINE, recomputed here so it's directly comparable ----
naive_mae = mean_absolute_error(y_test, X_test['lag_1'])
naive_mape = ((y_test - X_test['lag_1']).abs() / y_test).mean() * 100

results = [("Naive baseline", naive_mae, naive_mape)]

# ---- TRAIN THE MODELS ----
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = ((y_test - preds).abs() / y_test).mean() * 100
    results.append((name, mae, mape))
    trained[name] = model
    print(f"  trained {name}")

# ---- RESULTS TABLE ----
print("\n" + "="*60)
print(f"{'Model':<22}{'MAE':>10}{'MAPE':>10}{'vs baseline':>16}")
print("="*60)
for name, mae, mape in results:
    if name == "Naive baseline":
        print(f"{name:<22}{mae:>10.2f}{mape:>9.1f}%{'—':>16}")
    else:
        change = ((mae - naive_mae) / naive_mae) * 100
        marker = f"{change:+.1f}%"
        print(f"{name:<22}{mae:>10.2f}{mape:>9.1f}%{marker:>16}")
print("="*60)
print("Negative % = better than baseline")

# ---- SAVE THE BEST MODEL ----
best_name = min(results[1:], key=lambda r: r[1])[0]
joblib.dump(trained[best_name], "model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print(f"\nSaved best model ({best_name}) to model.pkl")

# ---- PER-COMMODITY BREAKDOWN ----
best_model = trained[best_name]
test_out = df[df['date'] >= cutoff].copy()
test_out['pred'] = best_model.predict(X_test)
test_out['model_err'] = (test_out['price'] - test_out['pred']).abs()
test_out['naive_err'] = (test_out['price'] - test_out['lag_1']).abs()

print("\n" + "="*60)
print(f"PER-COMMODITY: {best_name} vs naive")
print("="*60)
breakdown = test_out.groupby('commodity').agg(
    model_mae=('model_err', 'mean'),
    naive_mae=('naive_err', 'mean'),
).round(2)
breakdown['improvement_%'] = (
    (breakdown['naive_mae'] - breakdown['model_mae']) / breakdown['naive_mae'] * 100
).round(1)
print(breakdown.sort_values('improvement_%', ascending=False).to_string())

# ---- WHAT DID THE MODEL FIND USEFUL? ----
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols)
    print("\nTOP 10 FEATURES:")
    print(imp.sort_values(ascending=False).head(10).round(4).to_string())