import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from model import NeuralNetwork
from layers import Layer

def load_data(file_path, target_column, one_hot=True):
    data = pd.read_csv(file_path)
    X = data.drop(target_column, axis=1).values # Features
    y = data[target_column].values              # Target
    
    # One-hot encoding for classification problems
    if one_hot:
        y_unique = np.unique(y)
        y_one_hot = np.zeros((y.shape[0], len(y_unique)))
        for i, label in enumerate(y_unique):
            y_one_hot[y == label, i] = 1
        y = y_one_hot
    return X, y

def preprocess_data(X, y, test_size=0.2):
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def accuracy(y_pred, y_true):
    y_true_labels = np.argmax(y_true, axis=1)
    return np.mean(y_pred == y_true_labels) * 100

def confusion_matrix(y_pred, y_true):
    y_true_labels = np.argmax(y_true, axis=1)
    num_classes = y_true.shape[1]
    cm = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true_labels, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    return precision, recall

def save_model(model, filename):
    params = [{'weights': layer.weights, 'biases': layer.bias} for layer in model.layers]
    np.save(filename, params)


def load_model(filename):
    # Load weights and biases
    params = np.load(filename, allow_pickle=True)
    nn = NeuralNetwork()
    for param in params:
        layer = Layer(param['weights'].shape[0], param['weights'].shape[1])
        layer.weights = param['weights']
        layer.biases = param['biases']
        nn.add_layer(layer)
    return nn

def plot_metrics(losses, title="Training Loss"):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()