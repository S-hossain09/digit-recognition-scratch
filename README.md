# Neural Network for Digit Recognition

This document outlines the architecture and implementation of a simple two-layer neural network built from scratch to recognize handwritten digits from the MNIST dataset.

### 0. Data Loading and Processing

1.  **Loading**: The dataset is loaded from a `.csv` file using the `pandas` library.
2.  **Conversion & Shuffling**: The data is converted into a `numpy` array for efficient numerical computation. It's then shuffled randomly to ensure that the training and development sets are unbiased and representative of the overall data distribution.
3.  **Splitting**: The dataset is split into two parts:
    *   `data_train`: A larger portion used to train the network's parameters.
    *   `data_dev`: A smaller "development" or validation set used to check the model's performance on unseen data during development.
4.  **Transpose & Reshape**: The data is transposed so that each column represents a single image example and each row represents a pixel feature. This is a common convention that simplifies matrix operations during forward and backward propagation.
    *   `X`: The input data (pixel values for each image). Shape: `(784, m)`, where `m` is the number of images.
    *   `Y`: The corresponding labels (the actual digit). Shape: `(1, m)`.
5.  **Normalization**: The pixel values in `X` (originally from 0-255) are divided by 255. This scales them to a range of `[0, 1]`. Normalization helps the gradient descent algorithm converge faster and more reliably.

### 1. Forward Propagation

Forward propagation is the process of passing the input data `X` through the network's layers to generate an output prediction. For a single image, the process is:

1.  **Hidden Layer Calculation**:
    *   The weighted sum `Z1` is computed by taking the dot product of the first layer's weights `W1` and the input `X`, and then adding the bias `b1`.
    *   $Z1 = W1 \cdot X + b1$
    *   The result `Z1` is passed through an activation function (ReLU) to produce the hidden layer's output, `A1`.
    *   $A1 = ReLU(Z1)$

2.  **Output Layer Calculation**:
    *   The process is repeated for the second layer. The weighted sum `Z2` is computed using the output of the previous layer `A1` and the second layer's weights `W2` and bias `b2`.
    *   $Z2 = W2 \cdot A1 + b2$
    *   `Z2` is passed through a final activation function (Softmax) to produce the network's final output `A2`, which represents the predicted probabilities for each digit (0-9).
    *   $A2 = softmax(Z2)$

### 2. Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns that a simple linear model cannot.

*   **ReLU (Rectified Linear Unit)**: Used for the hidden layer.
    *   **Formula**: $ReLU(Z) = max(0, Z)$
    *   **Purpose**: It is computationally simple and effective. It outputs the input directly if it's positive, and zero otherwise. This helps prevent the "vanishing gradient" problem.

*   **Softmax**: Used for the output layer.
    *   **Formula**: $softmax(Z)_i = \frac{e^{Z_i}}{\sum_{j} e^{Z_j}}$
    *   **Purpose**: It converts the raw output scores (`Z2`) into a probability distribution. Each output neuron's value becomes a probability between 0 and 1, and the sum of all output probabilities is 1. The highest probability corresponds to the predicted digit.

### 3. Loss Function Calculation

The loss function measures how far the model's predictions (`A2`) are from the true labels (`Y`). While not explicitly defined as a separate function in the code, the derivative of the **Categorical Cross-Entropy** loss is used in backpropagation.

The error for the output layer is calculated as `dz2 = a2 - one_hot_y`. This is the derivative of the cross-entropy loss with respect to `z2` when using a Softmax activation function. `one_hot_y` is a "one-hot encoded" version of the true labels `Y`, where each label is a vector with a `1` at the index of the correct digit and `0`s elsewhere.

### 4. Backpropagation

Backpropagation is the core algorithm for training the network. It calculates the gradient of the loss function with respect to each weight and bias in the network. This gradient tells us how to adjust the parameters to reduce the loss.

The process works backward from the output layer:

1.  **Output Layer Gradients (`dW2`, `db2`)**:
    *   Calculate the error `dZ2` in the output layer's weighted sum. This is simply `A2 - Y_one_hot`.
    *   Use `dZ2` to find the gradients for `W2` and `b2`.

2.  **Hidden Layer Gradients (`dW1`, `db1`)**:
    *   Propagate the error `dZ2` back to the hidden layer to find the error `dZ1`. This involves the weights of the second layer (`W2`) and the derivative of the ReLU activation function.
    *   The derivative of ReLU, `g'(Z1)`, is `1` if `Z1 > 0` and `0` otherwise. This is efficiently calculated with the boolean expression `(z1 > 0)`.
    *   Use `dZ1` to find the gradients for `W1` and `b1`.

All gradients are averaged over the number of training examples (`m`).

### 5. Gradient Descent

Gradient descent is the optimization algorithm that updates the network's parameters (`W1`, `b1`, `W2`, `b2`) using the gradients calculated during backpropagation.

*   **Update Rule**: Each parameter is updated by subtracting a small fraction of its gradient. This fraction is determined by the **learning rate (`alpha`)**.
    *   $W1 = W1 - \alpha \cdot dW1$
    *   $b1 = b1 - \alpha \cdot db1$
    *   $W2 = W2 - \alpha \cdot dW2$
    *   $b2 = b2 - \alpha \cdot db2$

*   **Process**: This update step is repeated for a set number of `iterations`. With each iteration, the network's parameters are adjusted to move in the direction that minimizes the loss, making the model's predictions progressively more accurate.
