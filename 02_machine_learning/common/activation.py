import numpy as np
from pyAI.nn.module import Module
    
#-------------------------------------------------------------------------------------------

class Sigmoid(Module):        # f(x) = 1 / (1 + exp(-x))
    """
    Sigmoid activation function.
    
    f(x) = 1 / (1 + exp(-x))
    """
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, x):
        return self.output * (1 - self.output)

#-------------------------------------------------------------------------------------------

class ReLU(Module):             # f(x) = max(0, x)
    """
    ReLU (Rectified Linear Unit) activation function.
    
    f(x) = max(0, x)
    """
    def forward(self, x):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, x):
        return np.where(x > 0, 1, 0)

#-------------------------------------------------------------------------------------------

class Tanh(Module):             # f(x) = tanh(x)
    """
    Tanh (Hyperbolic Tangent) activation function.
    
    f(x) = tanh(x)
    """
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, x):
        return 1 - self.output ** 2

#-------------------------------------------------------------------------------------------

class Softmax(Module):          # f(x) = exp(x) / sum(exp(x))
    """
    Softmax activation function.
    
    f(x) = exp(x) / sum(exp(x))
    """
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.output

    def backward(self, x):
        raise NotImplementedError("Backward pass for softmax is typically combined with a loss function.")

#-------------------------------------------------------------------------------------------

class LeakyReLU(Module):        # f(x) = x if x > 0 else alpha * x
    """
    Leaky ReLU activation function.
    
    f(x) = x if x > 0 else alpha * x
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.output = np.where(x > 0, x, self.alpha * x)
        return self.output

    def backward(self, x):
        return np.where(x > 0, 1, self.alpha)
    
#-------------------------------------------------------------------------------------------

class ELU(Module):              # f(x) = x if x > 0 else alpha * (exp(x) - 1)
    """
    ELU (Exponential Linear Unit) activation function.
    
    f(x) = x if x > 0 else alpha * (exp(x) - 1)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        self.output = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return self.output

    def backward(self, x):
        return np.where(x > 0, 1, self.output + self.alpha)
    
#-------------------------------------------------------------------------------------------

class Swish(Module):            # f(x) = x / (1 + exp(-x))  |  f(x) = x * sigmoid(x)
    """
    Swish activation function.

    f(x) = x / (1 + exp(-x))
    """
    def forward(self, x):
        self.sigmoid = 1 / (1 + np.exp(-x))
        self.output = x * self.sigmoid
        return self.output

    def backward(self, x):
        return self.sigmoid + self.output * (1 - self.sigmoid)
    
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
