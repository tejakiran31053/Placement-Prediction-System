import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r'C:\Users\lenovo\PycharmProjects\PlacementPredictionSystem\data\placement_data.csv')

# Dataset must contain 'CGPA' and 'Salary Package'
X = df[['CGPA']]
Y = df['Salary Package']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, Y_train)

# Predict salary package
Y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

# Print model parameters
print(f"Intercept (b0): {model.intercept_:.2f}")
print(f"Slope (b1): {model.coef_[0]:.2f}")

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted values
plt.scatter(X_test, Y_test, label="Actual")
plt.plot(X_test, Y_pred, label="Predicted")
plt.xlabel("CGPA")
plt.ylabel("Salary Package")
plt.title("Linear Regression: CGPA vs Salary Package")
plt.legend()
plt.show()