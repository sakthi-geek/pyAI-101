import math

class Neuron:
    """
    This class represents a single neuron.
    """
    def __init__(self, weight_init, bias_init):
        # Initialize the weight and bias with the provided values
        self.weight = weight_init
        self.bias = bias_init

        # Initialize the input, output, and gradients to zero
        self.input = 0
        self.output = 0
        self.gradInput = 0      # The gradient of the input with respect to the loss
        self.gradWeight = 0     # The gradient of the weight with respect to the loss
        self.gradBias = 0       # The gradient of the bias with respect to the loss

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        Takes a number x, and applies the sigmoid function to it.
        """
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid activation function.
        Takes a number x, and computes the derivative of the sigmoid function at that point.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, input):
        """
        Forward propagation.
        Takes an input, applies the weights and bias, and then applies the activation function.
        """
        self.input = input

        # Compute the output of the neuron
        self.output = self.sigmoid(self.input * self.weight + self.bias)
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backward propagation.
        Takes an input, applies the weights and bias, and then applies the activation function.
        """
        # Compute the gradient of the input, which is the derivative of the activation function
        # multiplied by the output error
        self.gradInput = output_error * self.sigmoid_derivative(self.output)

        # Compute the gradient of the weight, which is the input multiplied by the gradient of the input
        self.gradWeight = self.gradInput * self.input

         # The gradient of the bias is just the gradient of the input
        self.gradBias = self.gradInput

        # Update the weight and bias using the gradients and the learning rate
        self.weight -= learning_rate * self.gradWeight
        self.bias -= learning_rate * self.gradBias

        # Return the gradient of the input for use in the next layer
        return self.gradInput
    
    def __repr__(self):
        """
        Return a string representation of the neuron.
        """
        return f"Neuron(weight={self.weight}, bias={self.bias}, output={self.output})"

class Layer:
    """
    This class represents a layer of neurons.
    """
    def __init__(self, input_size, output_size):
        self.weights = [[Neuron(0, 0) for _ in range(output_size)] for _ in range(input_size)]
        self.biases = [Neuron(0, 0) for _ in range(output_size)]

    def forward(self, input):
        self.input = input
        self.output = [Neuron(sum(i.value * w.value for i, w in zip(input, ws)) + b.value, 0) for ws, b in zip(self.weights, self.biases)]
        return self.output

    def backward(self, output_error, learning_rate):
        for i, (ws, b) in enumerate(zip(self.weights, self.biases)):
            for j, w in enumerate(ws):
                w.grad = output_error[j].value * self.input[i].value
                b.grad = output_error[j].value
                w.value -= learning_rate * w.grad
                b.value -= learning_rate * b.grad

class NeuralNetwork:
    """
    This class represents a neural network.
    """
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_error, learning_rate):
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)
        return output_error

    def train(self, data, true_output, learning_rate, epochs):
        for _ in range(epochs):
            output = self.forward(data)
            output_error = [Neuron(o.value - t.value, 0) for o, t in zip(output, true_output)]
            self.backward(output_error, learning_rate)

