import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.array([
    [1000, 2, 10],
    [1500, 3, 5],
    [1800, 4, 2],
    [2400, 4, 20],
    [3000, 5, 7]
])
y = np.array([100, 150, 180, 220, 300])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Actual:", y_test)
print("Predicted:", y_pred)