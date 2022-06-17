from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

boston = datasets.load_boston()

X = boston.data
y = boston.target

print (boston.feature_names)

print (boston.data[0:5])
print (boston.target[0:5])

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size = 0.2)
print (X_train.shape)
print (X_test.shape)

LR = LinearRegression(normalize=True)
LR.fit(X_train, y_train)

y_pred = LR.predict(X_test) 

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print ('RMSE (Root Mean Squared Error) = ', rmse)

r2 = r2_score(y_test, y_pred)
print ('R2 = ', r2)
