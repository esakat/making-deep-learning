import numpy as np

# sigmoid関数
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def identity_function(x):
  return x

# 入力層
X = np.array([1.0, 0.5])

# 第1層
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)

# 第2層
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)

# 最後の層
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([[0.1, 0.2]])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(Y)