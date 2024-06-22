import numpy as np
from pyAI.autograd.autograd import Function
from pyAI.autograd.tensor import Tensor
from pyAI.nn.module import Module

class ReLU(Function):
    def forward(self, x):
        result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self, grad):
        self._x.backward(grad * (self._x.data > 0))

class Sigmoid(Function):
    def forward(self, x):
        sigmoid = 1 / (1 + np.exp(-x.data))
        result = Tensor(sigmoid, requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self, grad):
        sigmoid = 1 / (1 + np.exp(-self._x.data))
        self._x.backward(grad * sigmoid * (1 - sigmoid))

class Tanh(Function):
    def forward(self, x):
        result = Tensor(np.tanh(x.data), requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self, grad):
        tanh = np.tanh(self._x.data)
        self._x.backward(grad * (1 - tanh ** 2))

class Softmax(Function):
    def forward(self, x):
        exps = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        result = Tensor(exps / np.sum(exps, axis=-1, keepdims=True), requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self, grad):
        softmax = self.forward(self._x).data
        grad_input = np.empty_like(grad)
        for i, (s, g) in enumerate(zip(softmax, grad)):
            s = s.reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            grad_input[i] = np.dot(jacobian, g)
        self._x.backward(grad_input)

class LeakyReLU(Function):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        result = Tensor(np.where(x.data > 0, x.data, self.alpha * x.data), requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self, grad):
        grad_input = np.where(self._x.data > 0, grad, self.alpha * grad)
        self._x.backward(grad_input)

class ELU(Function):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        result = Tensor(np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1)), requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self, grad):
        grad_input = np.where(self._x.data > 0, grad, grad * (self.alpha * np.exp(self._x.data)))
        self._x.backward(grad_input)

class Swish(Function):
    def forward(self, x):
        sigmoid = 1 / (1 + np.exp(-x.data))
        result = Tensor(x.data * sigmoid, requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self, grad):
        sigmoid = 1 / (1 + np.exp(-self._x.data))
        grad_input = grad * (sigmoid + self._x.data * sigmoid * (1 - sigmoid))
        self._x.backward(grad_input)


