# Define a simple dataset
houses = [
    {"sqft": 1000, "bedrooms": 2, "bathrooms": 2, "price": 200000},
    {"sqft": 1500, "bedrooms": 3, "bathrooms": 2, "price": 300000},
    {"sqft": 2000, "bedrooms": 4, "bathrooms": 3, "price": 400000},
    {"sqft": 1200, "bedrooms": 2, "bathrooms": 2, "price": 250000},
    {"sqft": 1800, "bedrooms": 3, "bathrooms": 3, "price": 350000},
    {"sqft": 2200, "bedrooms": 4, "bathrooms": 4, "price": 450000},
]

# Define features (X) and target variable (y)
X = [[house["sqft"], house["bedrooms"], house["bathrooms"]] for house in houses]
y = [house["price"] for house in houses]

# Split data into training and testing sets
train_size = int(len(houses) * 0.8)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Create and train linear regression model
class LinearRegressionModel:
    def __init__(self):
        self.weights = [0.0 for _ in range(len(X[0]))]
        self.bias = 0.0

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def train(self, X_train, y_train, learning_rate=0.01, iterations=1000):
        for _ in range(iterations):
            for x, y in zip(X_train, y_train):
                prediction = self.predict(x)
                error = prediction - y
                self.weights = [w - learning_rate * error * xi for w, xi in zip(self.weights, x)]
                self.bias -= learning_rate * error

model = LinearRegressionModel()
model.train(X_train, y_train)

# Make predictions on test data
y_pred = [model.predict(x) for x in X_test]

# Evaluate model performance
mse = sum((y - y_pred) ** 2 for y, y_pred in zip(y_test, y_pred)) / len(y_test)
r2 = 1 - (sum((y - y_pred) ** 2 for y, y_pred in zip(y_test, y_pred)) / sum((y - sum(y_test) / len(y_test)) ** 2 for y in y_test))
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-Squared Score: {r2:.2f}')

# Visualize predictions
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

# Use the model to make predictions
new_house = [2000, 4, 3]
predicted_price = model.predict(new_house)
print(f'Predicted Price: ${predicted_price:.2f}')