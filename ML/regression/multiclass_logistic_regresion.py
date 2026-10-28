import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(
    n_samples=3000,
    n_features=4,
    n_classes=3, 
    n_informative=3,
    n_redundant=0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=100
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Classification report: ", classification_report(y_test, y_pred))

new_student = np.array([[30, 20, -12, 13]])
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)
probability = model.predict_proba(new_student_scaled)

print("Prediction class: ", prediction[0])
print("Prediction probability: ", probability)
