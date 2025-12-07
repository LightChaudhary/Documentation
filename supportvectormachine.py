import pandas as pd 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("/Users/prakashchaudhary/Desktop/Student Performance Project/data/student_performance.csv")

X = df[['Study_Hours', 'Attendance', 'Previous_Score']]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_svm = SVC(kernel= 'linear', C = 0.1)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

rbf_svm = SVC(kernel= 'rbf', C = 0.01, gamma = 0.1)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

print("Linear SVM accuracy: ", accuracy_score(y_pred_linear, y_test))
print("RBF SVM accuracy: ", accuracy_score(y_pred_rbf, y_test))

C_values = [0.01, 0.1, 1, 10, 100]
gamma_values = [0.01, 0.1, 1, 10]

for C in C_values: 
    for gamma in gamma_values: 
        model = SVC(kernel= 'rbf', C = C, gamma = gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"C:{C}, Gamma: {gamma} -> Accuracy:{accuracy:.4f}")