# Import libraries
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the Ames Housing dataset
housing = fetch_openml(name="house_prices", as_frame=True, version=1)
data = housing.data  # Features
data['PRICE'] = housing.target  # Target (SalePrice)

# Select a few features for simplicity
X = data[['GrLivArea', 'OverallQual']]  # Living area and overall quality
y = data['PRICE']

# Handle missing values (simple approach: drop rows with NaN)
X = X.dropna()
y = y[X.index]  # Align target with filtered features

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize predictions vs actual prices
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Ames)")
plt.show()

# Save predictions to a CSV
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")