from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 

X = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([0, 0, 0, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(max_depth=2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Prediction: ", y_pred)
print("Actual values: ", y_test)
print("Accuracy: ", accuracy_score(y_test,y_pred))

print("Prediction for 6.5 hours: ", model.predict([[6.5]]))