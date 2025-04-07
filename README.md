# Neural Network from Scratch in Python

Welcome to this **Neural Network from Scratch** project! Built with **NumPy** for matrix computations and **Pandas** for data handling, this is a beginner-friendly yet comprehensive implementation using **Object-Oriented Programming (OOP)**. No deep learning frameworks like TensorFlow or PyTorch are used.

This project lets you:
- Design and train a neural network with customizable layers and activation functions.
- Load datasets (e.g., Iris, MNIST, or custom CSVs), preprocess them, and evaluate performance.
- Save and load trained models for later use.
- Visualize training progress with Matplotlib (optional).

---

## Project Structure

The code is modularized into separate files for reusability:
- **`layers.py`**: Defines the `Layer` class for neurons, weights, and biases.
- **`activations.py`**: Contains activation functions (ReLU, Sigmoid, Softmax) and their derivatives.
- **`model.py`**: Implements the `NeuralNetwork` class with training logic.
- **`utils.py`**: Handles data loading, preprocessing, evaluation, and visualization.
- **`main.py`**: Ties it all together to train and test the model.

---

## Core Functionalities Explained

### 1. Neural Network Design with OOP
- The network is built using a `NeuralNetwork` class that holds a list of `Layer` objects.
- Each `Layer` has:
  - **Weights**: Randomly initialized connections between neurons.
  - **Biases**: Adjustable offsets for each neuron.
  - **Activation Functions**: Applied to the layer’s output (e.g., ReLU, Sigmoid, Softmax).
- **Why OOP?** It makes the code modular, reusable, and easier to extend!

### 2. Forward Propagation
- **What it does**: Passes input data through the network to predict an output.
- **How it works**:
  1. Input data (`X`) is multiplied by weights and added to biases: `z = X * W + b`.
  2. An activation function (e.g., ReLU) is applied to `z` to get the output: `a = activation(z)`.
  3. This repeats for each layer until the final output is produced.
- **Example**: For Iris, the network predicts probabilities for each class (e.g., `[0.1, 0.85, 0.05]`).

### 3. Backpropagation
- **What it does**: Calculates how much each weight and bias contributed to the error and adjusts them.
- **How it works**:
  1. Compute the **loss** (error) between predicted output (`y_pred`) and true output (`y_true`).
  2. Use the **chain rule** to propagate the error backward through the layers.
  3. Calculate gradients (e.g., `dW`, `db`) for weights and biases.
- **Why?** It’s the magic that lets the network “learn” by updating parameters.

### 4. Gradient Descent
- **What it does**: Updates weights and biases to minimize the loss.
- **How it works**:
  - Adjust parameters in the opposite direction of the gradient: `W = W - learning_rate * dW`.
  - The `learning_rate` controls how big each step is (e.g., `0.01`).
- **Analogy**: Think of it like walking downhill to find the lowest point (minimum loss).

### 5. Activation Functions
- Transform layer outputs to introduce non-linearity (so the network can learn complex patterns).
- **Implemented Functions**:
  - **ReLU**: `max(0, x)` — great for hidden layers, prevents negative values.
  - **Sigmoid**: `1 / (1 + e^(-x))` — outputs 0 to 1, good for binary classification.
  - **Softmax**: Normalizes outputs into probabilities, perfect for multi-class problems (like Iris).
- **Derivatives**: Used in backpropagation to compute gradients.

### 6. Loss Functions
- Measure how “wrong” the predictions are:
  - **Cross-Entropy**: For classification (e.g., Iris). Compares predicted probabilities to true labels.
  - **Mean Squared Error (MSE)**: For regression. Measures the average squared difference between predictions and targets.
- **When to use?** Use cross-entropy for classification, MSE for continuous outputs.

### 7. Data Handling with Pandas
- Load datasets from CSV files (e.g., Iris) using Pandas.
- Preprocess data:
  - **Normalization**: Scale features to have zero mean and unit variance.
  - **Train/Test Split**: Divide data into training (e.g., 80%) and testing (e.g., 20%) sets.
  - **One-Hot Encoding**: Convert labels (e.g., `[0, 1, 2]`) into vectors (e.g., `[[1, 0, 0], [0, 1, 0], [0, 0, 1]]`).

### 8. Evaluation Metrics
- **Accuracy**: Percentage of correct predictions.
- **Confusion Matrix**: Shows true vs. predicted labels for each class.
- **Precision & Recall**: Measures prediction quality per class (useful for imbalanced data).

### 9. Model Saving & Loading
- Save trained weights and biases to a `.npy` file using NumPy.
- Load them later to make predictions without retraining.

### 10. Visualization (Optional)
- Plot training loss over epochs using Matplotlib to see how the model learns.

---

## **GETTING STARTED**

### Prerequisites
Install the required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Dataset
- Use the **Iris dataset** (`iris.csv`) or any CSV with features and a target column.
- Example Iris format:
  ```
  sepal_length,sepal_width,petal_length,petal_width,species
  5.1,3.5,1.4,0.2,0
  4.9,3.0,1.4,0.2,0
  7.0,3.2,4.7,1.4,1
  ...
  ```

### Running the Code
1. Clone or download this repository.
2. Place your `iris.csv` (or custom CSV) in the project folder.
3. Run the main script:
   ```bash
   python main.py
   ```
4. Watch the model train, see the results, and check the generated plot!

---

## Example Output
```
==========
Epoch: 1
Loss:  1.9177
...
==========
Epoch: 901
Loss:  0.0471
100%|██████████████████████████████| 1000/1000 [00:00<00:00, 18907.14it/s]
Accuracy: 0.9667
Confusion Matrix:
 [[10.  0.  0.]
 [ 0.  9.  0.]
 [ 0.  0. 11.]]
Precision: [1.0, 1.0, 0.875]
Recall: [1.0, 0.923, 1.0]
```

---

## Tips for Beginners
- **Start Small**: Use a tiny network (e.g., 1 hidden layer with 10 neurons) to debug.
- **Tune Hyperparameters**: Experiment with `learning_rate` (e.g., 0.01), `epochs` (e.g., 1000), and layer sizes.
- **Visualize**: Check the loss plot—if it’s not decreasing, tweak the learning rate or network size.
- **Extend It**: Add more activation functions, layers, or regularization once you’re comfortable.

---

## Next Steps
Once you master this, try:
- Adding regularization (e.g., L2) to prevent overfitting.
- Switching to a larger dataset like MNIST.
- Upgrading to TensorFlow or PyTorch for automatic differentiation and GPU support.

---
*Built with ❤️ by a neural network enthusiast for beginners like you!*

---
