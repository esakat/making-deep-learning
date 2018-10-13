import numpy as np

def softmax(a):
  exp_a = np.exp(a - c) # オーバフロー対策
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  
  return y