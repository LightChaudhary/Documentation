import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

area = np.array([100, 300, 500, 800, 1000, 1500, 1800, 2000, 2100, 2500, 3000, 3500, 3800]).reshape(-1,1)
price = np.array([10, 30, 50, 80, 100, 150, 190, 210, 220, 270,320, 370, 400])

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(area, price, test_size = 0.2, random_state = 42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Square Error: ", mse)
print("R2 score: ", r2)

plt.scatter(X_train, y_train, color = "blue", label = "Train set")
plt.scatter(X_test, y_test, color = "green", label = "Test set")
plt.plot(area, model.predict(area), color = "red", label = "Regression Line")
plt.xlabel("Area(sqft)")
plt.ylabel("Price $")
plt.legend()
plt.title("Linear Regression with traintestsplit")
plt.show()