from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np 

hours = np.array([[1], [2], [3], [4], [5], [6], [7]])
result = np.array([0, 0, 0, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(hours, result, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)

print("Actual value: ", y_test)
print("Predicted value: ", y_pred)
print("Accuracy score: ", accuracy)

y_pred2 = model.predict([[6.5]])
print("Prediction for 6.5 hours: ", y_pred2)

y_pred3 = model.predict_proba([[6.5]])
print("Fail probabilty: ", y_pred3[0][0])
print("Pass probability: ", y_pred3[0][1])

cm = confusion_matrix(y_test,y_pred)
print("Confusion matrix: ", cm)
print("Classification report", classification_report(y_test, y_pred))