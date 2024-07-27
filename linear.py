import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)  # For reproducibility
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Create a LinearRegression model
model = LinearRegression()

# Fit the model to the data
model.fit(x, y)

# Predict values
y_pred = model.predict(x)

# Plot the original data and the fitted line
plt.scatter(x, y, label='Data', color='red')
plt.plot(x, y_pred, label='Fitted line', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Example')
plt.show()

# Print the fitted parameters
print(f"Intercept: {model.intercept_[0]}")
print(f"Slope: {model.coef_[0][0]}")
