import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    @property
    def shape(self):
        return self.data.shape 
    
    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)

    def set_grad_fn(self, grad_fn):
        self._grad_fn = grad_fn

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = np.ones_like(self.data)      

        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output) 

        if self.grad is None:
            self.grad = grad_output.data
        else:
            # print("in gradient  :", self.grad.shape, self.grad)
            # print("grad         :", grad_output.data.shape, grad_output.data)
            self.grad += grad_output.data
            # print("out gradient :", self.grad.shape, self.grad)

        if self._grad_fn:
            self._grad_fn.backward(grad_output.data)
            
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __getitem__(self, key):
        return Tensor(self.data[key], requires_grad=self.requires_grad)

    def __add__(self, other):
        from pyAI.autograd.autograd import Add
        if isinstance(other, Tensor):
            result = Add().forward(self, other)
        else:
            result = Add().forward(self, Tensor(np.array(other)))
        return result
    
    def __mul__(self, other):
        from pyAI.autograd.autograd import Mul
        if isinstance(other, Tensor):
            result = Mul().forward(self, other)
        else:
            result = Mul().forward(self, Tensor(np.array(other)))
        return result

    def __matmul__(self, other):
        from pyAI.autograd.autograd import MatMul
        if isinstance(other, Tensor):
            result = MatMul().forward(self, other)
            return result
        else:
            raise ValueError("Both operands must be Tensor")

#-------------------------------------------------------------------------

    def __sub__(self, other):
        from pyAI.autograd.autograd import Sub
        if isinstance(other, Tensor):
            result = Sub().forward(self, other)
        else:
            result = Sub().forward(self, Tensor(np.array(other)))
        return result
    
    def __truediv__(self, other):
        from pyAI.autograd.autograd import Div
        if isinstance(other, Tensor):
            result = Div().forward(self, other)
        else:
            result = Div().forward(self, Tensor(np.array(other)))
        return result
    
    def __pow__(self, exponent):
        from pyAI.autograd.autograd import Pow
        result = Pow().forward(self, exponent)
        return result
    
    def sum(self, axis=None, keepdims=False):
        from pyAI.autograd.autograd import Sum
        result = Sum().forward(self, axis, keepdims)
        return result
    
    def mean(self, axis=None, keepdims=False):
        from pyAI.autograd.autograd import Mean
        result = Mean().forward(self, axis, keepdims)
        return result

    def log(self):
        from pyAI.autograd.autograd import Log
        result = Log().forward(self)
        return result

    def exp(self):
        from pyAI.autograd.autograd import Exp
        result = Exp().forward(self)
        return result
    
    def sin(self):
        from pyAI.autograd.autograd import Sin
        result = Sin().forward(self)
        return result
    
    def cos(self):
        from pyAI.autograd.autograd import Cos
        result = Cos().forward(self)
        return result

    def tanh(self):
        from pyAI.autograd.autograd import Tanh
        result = Tanh().forward(self)
        return result
    
    def clip(self, min_value, max_value):
        from pyAI.autograd.autograd import Clip
        result = Clip().forward(self, min_value, max_value)
        return result 

    def transpose(self):
        from pyAI.autograd.autograd import Transpose
        result = Transpose().forward(self)
        return result
