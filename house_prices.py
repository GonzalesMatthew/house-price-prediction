import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load Ames dataset
housing = fetch_openml(name="house_prices", as_frame=True, version=1)
data = housing.data
data['PRICE'] = housing.target

# Features
X = data[['GrLivArea', 'OverallQual', 'BedroomAbvGr', 'YearBuilt', 'TotalBsmtSF', 'GarageArea']]
y = data['PRICE']

# Fill NaNs
X = X.fillna(X.mean())
y = y[X.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
print("Feature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")

# Plot GrLivArea vs PRICE (example)
plt.subplot(1, 2, 2)
plt.scatter(data['GrLivArea'], data['PRICE'], color='green', alpha=0.5)
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Price")
plt.title("GrLivArea vs Price")
plt.tight_layout()
plt.show()

# Save results
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")