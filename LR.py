import numpy as np
from sklearn.linear_model import LinearRegression

# Input feature (X)
X = np.array([[1],
              [2],
              [3],
              [4],
              [5]])

# Target variable (y)
y = np.array([2, 4, 6, 8, 10])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Display model parameters
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict values
y_pred = model.predict(X)

print("\nActual Values:")
print(y)

print("\nPredicted Values:")
print(y_pred)

# Predict for a new value
new_x = np.array([[6]])
prediction = model.predict(new_x)

print("\nPrediction for x = 6:")
print(prediction[0])