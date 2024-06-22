import numpy as np
from pyAI.autograd.tensor import Tensor

class Context:
    def __init__(self):
        self._saved_tensors = ()

    def save_for_backward(self, *tensors):
        self._saved_tensors = tensors

    @property
    def saved_tensors(self):
        return self._saved_tensors

class Function:
    def __init__(self, *parents):
        self._ctx = None
        self.parents = parents
        for parent in parents:
            if parent.requires_grad:
                parent.set_grad_fn(self)
    
    @property
    def ctx(self):  # Dynamically creating a context object only when needed
        if self._ctx is None:
            self._ctx = Context()
        return self._ctx

    def forward(self, *args):
        raise NotImplementedError

    def backward(self,  grad_output):
        raise NotImplementedError

class Add(Function):
    def forward(self, a, b):  
        result = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        self._b = b
        return result

    def backward(self, grad_output):
        if self._a.requires_grad:
            self._a.backward( grad_output)
        if self._b.requires_grad:
            if grad_output.shape != self._b.data.shape:
                # Ensure gradients for b are summed if b is a bias term
                summed_grad_output = np.sum(grad_output, axis=0)
                self._b.backward(summed_grad_output)
            else:
                self._b.backward(grad_output)

class Mul(Function):
    def forward(self, a, b):
        result = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        self._b = b
        return result

    def backward(self, grad_output):
        if self._a.requires_grad:
            # print("Mul backward a", grad_output * self._b.data)
            self._a.backward( grad_output * self._b.data)
        if self._b.requires_grad:
            # print("Mul backward b", grad_output * self._a.data)
            self._b.backward( grad_output * self._a.data)

class MatMul(Function):
    def forward(self, a, b):
        result = Tensor(np.dot(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)
        result.set_grad_fn(self)
        self.ctx.save_for_backward(a, b)
        return result

    def backward(self, grad_output):
        a, b = self.ctx.saved_tensors
        # print( grad_output.shape, b.data.T.shape)
        grad_a = np.dot( grad_output, b.data.T)
        # print("MatMul backward a", grad_a, grad_a.shape)

        # print(a.data.T.shape,  grad_output.shape)
        grad_b = np.dot(a.data.T,  grad_output)
        if grad_b.shape != b.data.shape:
            grad_b = np.sum(grad_b, axis=0)  # Summing across the batch dimension for biases
        # print("MatMul backward b", grad_b, grad_b.shape)
                 
        if a.requires_grad:
            a.backward(grad_a)
        
        if b.requires_grad:
            b.backward(grad_b)

class Sigmoid(Function):
    def forward(self, x):
        sigmoid = 1 / (1 + np.exp(-x.data))
        result = Tensor(sigmoid, requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self,  grad_output):
        sigmoid = 1 / (1 + np.exp(-self._x.data))
        self._x.backward( grad_output * sigmoid * (1 - sigmoid))

class ReLU(Function):
    def forward(self, x):
        result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        result.set_grad_fn(self)
        self._x = x
        return result

    def backward(self,  grad_output):
        self._x.backward( grad_output * (self._x.data > 0))

#----------------------------------------------------------------------------------------------------

class Sub(Function):
    def forward(self, a, b):
        result = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        self._b = b
        return result

    def backward(self, grad_output):
        if self._a.requires_grad:
            self._a.backward(grad_output)
        if self._b.requires_grad:
            self._b.backward(-grad_output)

class Div(Function):
    def forward(self, a, b):
        result = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        self._b = b
        return result

    def backward(self, grad_output):
        if self._a.requires_grad:
            self._a.backward(grad_output / self._b.data)
        if self._b.requires_grad:
            self._b.backward(-grad_output * self._a.data / (self._b.data ** 2))

class Pow(Function):
    def forward(self, a, exponent):
        self.exponent = exponent
        result = Tensor(a.data ** exponent, requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        return result

    def backward(self, grad_output):
        grad = self.exponent * (self._a.data ** (self.exponent - 1))
        self._a.backward(grad_output * grad)

class Sum(Function):
    def forward(self, a, axis=None, keepdims=False):
        result = Tensor(np.sum(a.data, axis=axis, keepdims=keepdims), requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        return result

    def backward(self, grad_output):
        grad = np.ones_like(self._a.data) * grad_output
        self._a.backward(grad)

class Mean(Function):
    def forward(self, a, axis=None, keepdims=False):
        result = Tensor(np.mean(a.data, axis=axis, keepdims=keepdims), requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        self._size = np.prod(a.data.shape) if axis is None else a.data.shape[axis]
        return result

    def backward(self, grad_output):
        grad = (np.ones_like(self._a.data) * grad_output) / self._size
        self._a.backward(grad)

class Log(Function):
    def forward(self, a):
        result = Tensor(np.log(a.data), requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        return result

    def backward(self, grad_output):
        grad = grad_output / self._a.data
        self._a.backward(grad)

class Exp(Function):
    def forward(self, a):
        result = Tensor(np.exp(a.data), requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        return result

    def backward(self, grad_output):
        grad = grad_output * np.exp(self._a.data)
        self._a.backward(grad)

class Sin(Function):
    def forward(self, a):
        result = Tensor(np.sin(a.data), requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        return result

    def backward(self, grad_output):
        grad = grad_output * np.cos(self._a.data)
        self._a.backward(grad)

class Cos(Function):
    def forward(self, a):
        result = Tensor(np.cos(a.data), requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        return result

    def backward(self, grad_output):
        grad = -grad_output * np.sin(self._a.data)
        self._a.backward(grad)

class Tanh(Function):
    def forward(self, a):
        result = Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        return result

    def backward(self, grad_output):
        grad = grad_output * (1 - np.tanh(self._a.data) ** 2)
        self._a.backward(grad)

class Clip(Function):
    def forward(self, a, min_value, max_value):
        result = Tensor(np.clip(a.data, min_value, max_value), requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        self._min_value = min_value
        self._max_value = max_value
        return result

    def backward(self, grad_output):
        grad = grad_output * ((self._a.data >= self._min_value) & (self._a.data <= self._max_value))
        self._a.backward(grad)

class Transpose(Function):
    def forward(self, a):
        result = Tensor(a.data.T, requires_grad=a.requires_grad)
        result.set_grad_fn(self)
        self._a = a
        return result

    def backward(self, grad_output):
        grad = grad_output.T
        self._a.backward(grad)
