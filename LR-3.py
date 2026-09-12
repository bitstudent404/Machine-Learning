from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

iris = load_iris()

print("Feature Names:")
print(iris.feature_names)

print("\nDataset Shape:")
print(iris.data.shape)

print("\nFirst 5 rows of the dataset:")
print(iris.data[:5])

X = iris.data[:,0].reshape(-1,1)

y = iris.data[:,2]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


print("Intercept:",model.intercept_)
print("Slope:",model.coef_[0])
print("MSE:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))