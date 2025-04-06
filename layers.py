import numpy as np
from activations import *

class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) # np.random.randn gives a random number from a standard normal distribution (mean=0, std=1)
        self.bias = np.zeros((1, output_size))
        self.activation = activation # eg. 'relu', 'sigmoid', 'softmax'
        self.input = None
        self.z = None   # pre-activation output
        self.a = None   # post-activation output
    
    def forward(self, X):
        # X is the input (batch_size, input_size)
        self.input = X
        self.z = np.dot(X, self.weights) + self.bias # Linear transformation
        if self.activation == 'relu':
            self.a = relu(self.z)
        elif self.activation =='sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation =='softmax':
            self.a = softmax(self.z)
        else:
            self.a = self.z # No activation
        return self.a
    
    def backward(self, delta, learning_rate):
        # Delta is the gradient from the next layer
        if self.activation == 'relu':
            delta = delta * relu_derivative(self.z)
        elif self.activation =='sigmoid':
            delta = delta * sigmoid_derivative(self.z)
        elif self.activation =='softmax':
            # Softmax derivative is handled in the loss function (cross-entropy)
            pass
        
        # Computing the gradients
        
        # self.input.T is the transpose of the input, shape (input_size, batch_size).
        # delta is (batch_size, output_size).
        # np.dot(self.input.T, delta) results in (input_size, output_size), matching self.weights’ shape.
        dW = np.dot(self.input.T, delta)
        
        # Computes the gradient of the loss with respect to the biases (db):
        # np.sum(delta, axis=0) sums delta across the batch (rows), reducing (batch_size, output_size) to (output_size,).
        # keepdims=True keeps it as (1, output_size) to match self.biases’ shape.
        # Intuition: Aggregates the error contribution for each neuron’s bias.
        db = np.sum(delta, axis=0, keepdims=True)

        # Computes the gradient of the loss with respect to the input (dX):
        # delta is (batch_size, output_size).
        # self.weights.T is (output_size, input_size).
        # Result is (batch_size, input_size), matching self.input’ shape.
        # This gradient is passed to the previous layer during backpropagation.
        dX = np.dot(delta, self.weights.T)
        
        # Update the weights and biases
        self.weights = self.weights - (learning_rate * dW)
        self.bias = self.bias - (learning_rate * db)
        
        return dX

# layer = Layer(2, 3, 'relu')
# X = np.array([[1, 2], [3, 4]])
# output = layer.forward(X)
# print(output)