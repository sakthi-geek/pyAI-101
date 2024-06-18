import numpy as np

class Activation(Layer):
    """
    Base class for activation layers. Inherits from Layer.
    """
    def forward(self, input):
        """
        Computes the forward pass of the activation function.
        """
        raise NotImplementedError

    def backward(self, input, grad_output):
        """
        Computes the backward pass of the activation function.
        """
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, input):
        """
        Forward pass for ReLU.
        """
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        """
        Backward pass for ReLU.
        """
        relu_grad = input > 0
        return grad_output * relu_grad


class Sigmoid(Activation):
    def forward(self, input):
        """
        Forward pass for Sigmoid.
        """
        return 1 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        """
        Backward pass for Sigmoid.
        """
        sigmoid = self.forward(input)
        return grad_output * sigmoid * (1 - sigmoid)


class Tanh(Activation):
    def forward(self, input):
        """
        Forward pass for Tanh.
        """
        return np.tanh(input)

    def backward(self, input, grad_output):
        """
        Backward pass for Tanh.
        """
        tanh = self.forward(input)
        return grad_output * (1 - tanh ** 2)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        """
        Initialize LeakyReLU with a slope parameter alpha.
        """
        self.alpha = alpha

    def forward(self, input):
        """
        Forward pass for LeakyReLU.
        """
        return np.where(input > 0, input, self.alpha * input)

    def backward(self, input, grad_output):
        """
        Backward pass for LeakyReLU.
        """
        dx = np.ones_like(input)
        dx[input < 0] = self.alpha
        return grad_output * dx


class Softmax(Activation):
    def forward(self, input):
        """
        Forward pass for Softmax.
        """
        exp_vals = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return probabilities

    def backward(self, input, grad_output):
        """
        Backward pass for Softmax.
        
        The gradient of Softmax is a bit more complex because it depends on the output values.
        Here, for simplicity, we're computing the gradient in the context of a loss function, 
        which is often the case in practice.
        """
        output = self.forward(input)
        return output - grad_output


# Assuming X is your input data
relu = ReLU()
output = relu.forward(X)
grad_input = relu.backward(X, grad_output)

# Assuming X is your input data
leaky_relu = LeakyReLU(alpha=0.01)
output = leaky_relu.forward(X)
grad_input = leaky_relu.backward(X, grad_output)

# For the final layer in a classification task
softmax = Softmax()
output = softmax.forward(X)
# Note: Softmax backward pass typically integrates with the loss function,
# such as cross-entropy, for efficiency and numerical stability.
