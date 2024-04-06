import numpy as np

class Layer:
    """
    Base class for all layers
    """
    def __init__(self):
        # A flag to denote if the layer has weights
        self.has_weights = False

    def forward(self, input):
        """
        Forward pass which every layer should use
        """
        raise NotImplementedError

    def backward(self, input, grad_output):
        """
        Backward pass which every layer should use
        """
        raise NotImplementedError


class Dense(Layer):
    """
    A fully connected layer that linearly transforms its input data: output = input @ W + b
    """
    def __init__(self, input_units, output_units, learning_rate=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)
        self.has_weights = True

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b
        """
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        """
        Backpropagate through the layer to update weights and biases
        """
        grad_input = np.dot(grad_output, self.weights.T)

        # Compute gradients w.r.t weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]

        # Update weights and biases
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases

        return grad_input

# Example usage
if __name__ == "__main__":
    np.random.seed(42)  # for reproducibility
    X = np.random.rand(10, 20)  # dummy input

    dense = Dense(input_units=20, output_units=10)
    output = dense.forward(X)
    print("Output Shape:", output.shape)  # should be (10, 10)

    # Example backward pass
    grad_output = np.random.rand(10, 10)  # dummy gradient coming from the next layer
    grad_input = dense.backward(X, grad_output)
    print("Grad Input Shape:", grad_input.shape)  # should match X's shape
