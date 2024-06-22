from pyAI.autograd.tensor import Tensor
from pyAI.nn.module import Module
import numpy as np
from pyAI.autograd.autograd import MatMul

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        std = np.sqrt(2. / in_features)  # He initialization
        self.weight = Tensor(np.random.randn(in_features, out_features) * std, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self.add_parameter(self.weight)
        self.add_parameter(self.bias)

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise ValueError("Input must be a Tensor")
        return x @ self.weight + self.bias            # return MatMul().forward(x, self.weight) + self.bias

