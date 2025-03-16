import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # New import
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load Ames dataset
housing = fetch_openml(name="house_prices", as_frame=True, version=1)
data = housing.data
data['PRICE'] = housing.target

# More features for fun
X = data[['GrLivArea', 'OverallQual', 'BedroomAbvGr']]  # Added bedrooms
y = data['PRICE']

# Drop NaNs
X = X.dropna()
y = y[X.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Random Forest)")
plt.show()

# Save results
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")