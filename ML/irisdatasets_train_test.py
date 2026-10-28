import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_train = pd.read_csv("Iris-Train.csv")
x = data_train.iloc[:, :-1]
y = data_train.iloc[:, -1]
model = RandomForestClassifier()
model = model.fit(x,y)

data_test = pd.read_csv("Iris-Test.csv")
y_test = data_test.iloc[:, -1]
x_test = data_test.iloc[:, :-1]
prediction = model.predict(x_test)

print(f"\nActual Iris Flowers: {y_test.values}")
print(f"\nPredicted Iris Flowers: {prediction}")
accuracy = 100 * (accuracy_score(y_test, prediction))
print(f"\nTest Accuracy: {accuracy:.2f}%")

misclassified = data_test[prediction != y_test]
print(f"\nMisclassified Samples(s): {misclassified} ")

train_peds = model.predict(x)
print("\nTrain Accuracy: ", 100 * accuracy_score(y, train_peds))