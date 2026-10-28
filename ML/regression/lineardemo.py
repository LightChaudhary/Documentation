#Linear Regression
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

area = np.array([500, 800, 1000, 1200, 1500, 1800]).reshape(-1, 1)
price = np.array([50, 80, 100, 120, 150, 180])

model = LinearRegression()
model.fit(area,price)

new_area = np.array([[1300]])
predicted_price = model.predict(new_area)
print("Predicted price for 1300 sq ft: ", predicted_price)

plt.scatter(area,price)
plt.plot(area, model.predict(area))
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (thousands)")
plt.title("Linear Regression Model")
plt.show()