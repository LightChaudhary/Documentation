import numpy as np

X = np.array([2,3])
w = np.array([0.5, 1])

b = -2

z = np.dot(w,X) + b

prediction = 1 if z > 0 else 0

print("Weighted sum:", z)
print("Prediction:", prediction)