from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
iris = load_iris()

print("Feature Names:")
print(iris.feature_names)

print("\nDataset Shape:")
print(iris.data.shape)

print("\nFirst 5 rows of the dataset:")
print(iris.data[:5])


# Multiple features
# 0 = Sepal Length
# 1 = Sepal Width
# 3 = Petal Width

X = iris.data[:, [0, 1, 3]]

# Target
# 2 = Petal Length

y = iris.data[:, 2]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Results
print("\nIntercept:", model.intercept_)

print("Coefficients:", model.coef_)

print("MSE:", mean_squared_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))
