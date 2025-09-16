## SCIKIT LEARN
# Understanding how the decision tree works according to the samples and features.
# While understanding samples as weight and features as the attributes(animals)
from sklearn import tree
features = [[110, 0], 
            [150, 0],
            [90, 0], 
            [10, 1],
            [200, 1],
            [300, 1]]
labels = ["cat", "cat", "cat", "dog", "dog","dog"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
prediction = clf.predict([[100,0], [110,0], [90,1], [300,1]])
print(f"The animals is {prediction}")