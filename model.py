import numpy as np
from layers import Layer

class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, X):
        # Forward propagation through the layers
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def compute_loss(self, y_true, y_pred, loss_type='cross_entropy'):
        m = y_true.shape[0] # Number of examples
        
        if loss_type == 'cross_entropy':
            # add a small epsilon to avoid log(0) and divide by m
            loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m 
        elif loss_type == 'mse':
            loss = np.mean(np.square(y_true - y_pred))
        return loss
    
    def backward(self, y_true, y_pred, learning_rate, loss_type='cross_entropy'):
        if loss_type == 'cross_entropy':
            delta = (y_pred - y_true)
        elif loss_type == 'mse':
            delta = 2 * (y_pred - y_true)
        
        # Propagate the gradients backward through the layers
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)
        
    def train(self, X, y, epochs, learning_rate, loss_type='cross_entropy', verbose=True):
        losses = []
        from tqdm import tqdm
        
        for epoch in tqdm(range(epochs)):
            if verbose and epoch % 100 == 0:
                print("="*10)
                print(f"Epoch: {epoch+1}")

            # Forward Propagation
            y_pred = self.forward(X)
            
            # Calculate Loss
            loss = self.compute_loss(y, y_pred, loss_type)
            losses.append(loss)
            
            # Backward Propagation
            self.backward(y, y_pred, learning_rate, loss_type)
            
            # Update Weights and Biases
            # -> The updates are handled in the backward method of each layer
            
            print(f"Loss:  {loss:.4f}")
        return losses
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)