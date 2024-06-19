"""
activation_functions.py

This module provides various activation functions commonly used in machine learning and deep learning models. 
Each function is implemented from scratch to help learners understand the underlying mechanics.
"""

import numpy as np

class ActivationFunction:
    """
    Base class for all activation functions.
    """
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented!")

    def backward(self, x):
        raise NotImplementedError("Backward method not implemented!")
    
#-------------------------------------------------------------------------------------------

class Sigmoid(ActivationFunction):        # f(x) = 1 / (1 + exp(-x))
    """
    Sigmoid activation function.
    
    f(x) = 1 / (1 + exp(-x))
    """
    def forward(self, x):
        """
        Forward pass of the sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying the sigmoid function.
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, x):
        """
        Backward pass (derivative) of the sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient of the sigmoid function.
        """
        return self.output * (1 - self.output)

#-------------------------------------------------------------------------------------------

class ReLU(ActivationFunction):             # f(x) = max(0, x)
    """
    ReLU (Rectified Linear Unit) activation function.
    
    f(x) = max(0, x)
    """
    def forward(self, x):
        """
        Forward pass of the ReLU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying the ReLU function.
        """
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, x):
        """
        Backward pass (derivative) of the ReLU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient of the ReLU function.
        """
        return np.where(x > 0, 1, 0)

#-------------------------------------------------------------------------------------------

class Tanh(ActivationFunction):             # f(x) = tanh(x)
    """
    Tanh (Hyperbolic Tangent) activation function.
    
    f(x) = tanh(x)
    """
    def forward(self, x):
        """
        Forward pass of the tanh activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying the tanh function.
        """
        self.output = np.tanh(x)
        return self.output

    def backward(self, x):
        """
        Backward pass (derivative) of the tanh activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient of the tanh function.
        """
        return 1 - self.output ** 2

#-------------------------------------------------------------------------------------------

class Softmax(ActivationFunction):          # f(x) = exp(x) / sum(exp(x))
    """
    Softmax activation function.
    
    f(x) = exp(x) / sum(exp(x))
    """
    def forward(self, x):
        """
        Forward pass of the softmax activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying the softmax function.
        """
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output

    def backward(self, x):
        """
        Backward pass (derivative) of the softmax activation function.
        
        Note: The backward pass for softmax is typically used in combination with a loss function.
        """
        # Placeholder: actual implementation should be coupled with a loss function
        raise NotImplementedError("Backward pass for softmax is typically combined with a loss function.")
    
#-------------------------------------------------------------------------------------------

class LeakyReLU(ActivationFunction):        # f(x) = x if x > 0 else alpha * x
    """
    Leaky ReLU activation function.
    
    f(x) = x if x > 0 else alpha * x
    """
    def __init__(self, alpha=0.01):
        """
        Initialize the Leaky ReLU activation function with a given alpha value.

        Args:
            alpha (float, optional): Slope of the negative part of the function. Defaults to 0.01.
        """
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass of the Leaky ReLU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying the Leaky ReLU function.
        """
        self.output = np.where(x > 0, x, self.alpha * x)
        return self.output

    def backward(self, x):
        """
        Backward pass (derivative) of the Leaky ReLU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient of the Leaky ReLU function.
        """
        return np.where(x > 0, 1, self.alpha)
    
#-------------------------------------------------------------------------------------------

class ELU(ActivationFunction):              # f(x) = x if x > 0 else alpha * (exp(x) - 1)
    """
    ELU (Exponential Linear Unit) activation function.
    
    f(x) = x if x > 0 else alpha * (exp(x) - 1)
    """
    def __init__(self, alpha=1.0):
        """
        Initialize the ELU activation function with a given alpha value.

        Args:
            alpha (float, optional): Slope of the negative part of the function. Defaults to 1.0.
        """
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass of the ELU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying the ELU function.
        """
        self.output = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return self.output

    def backward(self, x):
        """
        Backward pass (derivative) of the ELU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient of the ELU function.
        """
        return np.where(x > 0, 1, self.output + self.alpha)
    
#-------------------------------------------------------------------------------------------

class Swish(ActivationFunction):            # f(x) = x / (1 + exp(-x))  |  f(x) = x * sigmoid(x)
    """
    Swish activation function.

    f(x) = x / (1 + exp(-x))
    """
    def forward(self, x):
        """
        Forward pass of the Swish activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output after applying the Swish function.
        """
        self.sigmoid = 1 / (1 + np.exp(-x))
        self.output = x * self.sigmoid
        return self.output

    def backward(self, x):
        """
        Backward pass (derivative) of the Swish activation function.

        Args:
            x (np.ndarray): Input array.
        
        Returns:
            np.ndarray: Gradient of the Swish function.
        """
        self.sigmoid = 1 / (1 + np.exp(-x))
        return self.output + self.sigmoid * (1 - self.output)
    
#-------------------------------------------------------------------------------------------

#===========================================================================================================

# Example usage:
if __name__ == "__main__":
    x = np.array([[1, 2, 3], [1, 2, -1]])

    sigmoid = Sigmoid()
    print("Sigmoid forward:\n", sigmoid.forward(x))
    print("Sigmoid backward:\n", sigmoid.backward(x))

    relu = ReLU()
    print("ReLU forward:\n", relu.forward(x))
    print("ReLU backward:\n", relu.backward(x))

    tanh = Tanh()
    print("Tanh forward:\n", tanh.forward(x))
    print("Tanh backward:\n", tanh.backward(x))

    softmax = Softmax()
    print("Softmax forward:\n", softmax.forward(x))
    # Note: Softmax backward pass is not implemented for standalone use.

    leaky_relu = LeakyReLU()
    print("Leaky ReLU forward:\n", leaky_relu.forward(x))
    print("Leaky ReLU backward:\n", leaky_relu.backward(x))

    elu = ELU()
    print("ELU forward:\n", elu.forward(x))
    print("ELU backward:\n", elu.backward(x))

    swish = Swish()
    print("Swish forward:\n", swish.forward(x))
    print("Swish backward:\n", swish.backward(x))
