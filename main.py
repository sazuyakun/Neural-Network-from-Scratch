from model import NeuralNetwork
from layers import Layer
from utils import load_data, preprocess_data, accuracy, confusion_matrix, precision_recall, save_model, plot_metrics, load_model

X, y = load_data('iris.csv', target_column='species', one_hot=True)
X_train, X_test, y_train, y_test = preprocess_data(X, y)

nn = NeuralNetwork()
nn.add_layer(Layer(input_size=X.shape[1], output_size=10, activation='relu'))
nn.add_layer(Layer(input_size=10, output_size=y_train.shape[1], activation='softmax'))

history = nn.train(X_train, y_train, epochs=1000, learning_rate=0.001, loss_type='cross_entropy')

y_pred = nn.predict(X_test)
acc = accuracy(y_pred, y_test)
cm = confusion_matrix(y_pred, y_test)
precision, recall = precision_recall(y_pred, y_test)
print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)

save_model(nn, 'model_weights.npy')
plot_metrics(history)