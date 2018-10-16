import numpy as np

# 偏微分
def function_2(x):
  return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x)

  for idx in range(x.size):
    # f(x+hの計算)
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x)

    # f(x-h)の計算
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val

  return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x
  for _ in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

# 勾配法
import sys, os
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
  def __init__(self):
    self.W = np.random.randn(2,3)

  def predict(self, x):
    return np.dot(x, self.W)

  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss

# 確認
net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p)
t = np.array([0,0,1])
net.loss(x,t)

def f(W):
  return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)