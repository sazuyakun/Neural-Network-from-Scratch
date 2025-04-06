import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    # To prevent overflow when computing exp(x), we subtract the max of x. It doesn't change the result, just improves numerical stability.
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)