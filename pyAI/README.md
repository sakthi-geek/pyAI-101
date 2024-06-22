# pyAI: Core AI Components

Welcome to the **pyAI** directory, the core of the pyAI-101 project. This directory contains the essential components that form the backbone of our AI/ML framework, providing from-scratch implementations of core components essential for various machine learning and deep learning techniques.

## Overview

The `pyAI` directory is designed to provide a deep understanding of the inner workings of machine learning and deep learning frameworks. It includes implementations of tensors, automatic differentiation (autograd), neural network modules, layers, activation functions, loss functions, optimizers, data handling, and utility functions.

## Directory Structure

```plaintext
pyAI/
├── README.md
├── __init__.py
├── config.ini
├── main.py
├── autograd/
│   ├── autograd.py
│   └── tensor.py
├── data/
│   ├── dataset.py
│   └── data_loader.py
├── nn/
│   ├── __init__.py 
│   ├── module.py 
│   ├── loss.py 
│   ├── activation.py 
│   └── layers/
│       └── linear.py
├── optim/
│   ├── __init__.py 
│   ├── optimizer.py 
│   └── sgd.py 
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── visualization.py
```

## Core Components

### Tensors and Autograd

- `Tensor`: The fundamental data structure of pyAI, which stores data and the gradient computations.
- `Autograd`: A mechanism to automatically calculate gradients for tensor operations, enabling backpropagation.

### Modules and Neural Networks

- `Module`: The base class for all neural network components, handling parameter registration, and applying gradients.
- `Layers`: Includes various types of neural network layers like Linear, Convolutional, etc.
- `Activation Functions`: Non-linearities like ReLU, Sigmoid, and others are defined here.
- `Loss Functions`: Various loss functions necessary for training models, such as MSE, Cross-Entropy, and more.

### Optimizers

- `SGD`: Implements the stochastic gradient descent optimization method.

### Utilities

- `Dataset` and `DataLoader`: Tools for handling and batching data that are ready for training processes.
- `Metrics` and `Visualization`: Helpers for evaluating model performance and visualizing data and training results.

## Important Modules

### 1. Activation Functions (`activation.py`)
This module provides various activation functions commonly used in machine learning and deep learning models. Each function is implemented from scratch to help learners understand the underlying mechanics.

**Activation Functions:**  (IMPLEMENTED)

- `Sigmoid`: Sigmoid activation function.
- `ReLU`: Rectified Linear Unit activation function.
- `Tanh`: Hyperbolic Tangent activation function.
- `Softmax`: Softmax activation function.
- `Leaky ReLU`: Leaky ReLU activation function with a customizable slope.
- `ELU`: Exponential Linear Unit activation function.
- `Swish`: Swish activation function.

### 2. Loss Functions (`loss.py`)
This module provides various loss functions commonly used in machine learning and deep learning models. Each function is implemented from scratch to help learners understand the underlying mechanics.

**Loss Functions:**         (IMPLEMENTED)

- `MSE` (Mean Squared Error)
- `BCE` (Binary Cross-Entropy)
- `CCE` (Categorical Cross-Entropy)

**Can be expanded to**      (WORK IN PROGRESS)
- `MAE` (Mean Absolute Error)
- `Huber Loss` (Smooth L1 Loss)
- `Hinge Loss` 

### 3. Optimizers (`optimizer.py`)
This module provides various optimization algorithms commonly used in machine learning and deep learning models. Each optimizer is implemented from scratch to help learners understand the underlying mechanics.

**Optimizers:**             (IMPLEMENTED)

- `SGD` (Stochastic Gradient Descent)
- `Momentum`

**Can be expanded to**      (WORK IN PROGRESS)
- `NAG` (Nesterov Accelerated Gradient)
- `RMSprop`
- `AdaGrad`
- `AdaDelta`
- `Adamax`
- `Adam`
- `Nadam` (Nesterov-accelerated Adaptive Moment Estimation)

### 4. Evaluation Metrics (`metrics.py`)
This module provides various evaluation metrics commonly used in machine learning and deep learning models. Each metric is implemented from scratch to help learners understand the underlying mechanics.

**Evaluation Metrics:**         (IMPLEMENTED)

- `MSE` (Mean Squared Error)
- `MAE` (Mean Absolute Error)
- `RMSE` (Root Mean Squared Error)
- `Accuracy`
- `Precision`
- `Recall`
- `F1 Score`
- `Mean IoU` (Mean Intersection over Union)
- `Confusion Matrix`
- `Classification Report`
- `ROC curve` and `ROC AUC score`
- `Gini Coefficient`
- `Log Loss`
- `Mean Squared Log Error`
- `R-Squared`

### 5. Layers (layers directory)
This directory contains various types of neural network layers, each implemented from scratch to provide a clear understanding of their functionalities.

**Layers**                  (IMPLEMENTED)

- `Linear`: a fully connected neural network layer.

**Can be expanded to**      (WORK IN PROGRESS)

- `Conv`: a convolutional neural network layer.
- `Pooling`: pooling layers like max pooling and average pooling.
- `Dropout`: dropout regularization layer.
- `Flatten`: a layer to flatten the input.
- `Batch Norm`: batch normalization layer.
- `RNN`: a recurrent neural network layer.
- `Attention`: an attention mechanism layer.