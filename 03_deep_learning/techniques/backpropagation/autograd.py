import numpy as np

class Tensor:
    """
    Represents a numerical tensor for computations that require automatic differentiation.
    
    Attributes:
        data (np.ndarray): The tensor's data.
        requires_grad (bool): If True, gradients will be computed for this tensor.
        grad (Tensor): Gradient of the tensor after backward pass.
        creator (Operation): The operation that produced this tensor.
        children (dict): A record of operations that depend on this tensor.
    
    Methods:
        backward(grad=None): Computes gradients of this tensor.
        zero_grad(): Resets the gradient of the tensor.
        add_child(child_op): Registers an operation that uses this tensor.
        all_gradients_accounted_for(): Checks if gradients from all children are accounted for.
    """

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None
        self.children = {}  # Track how many children an operation has

    def add_child(self, child_op):
        """Register a child operation."""
        if id(child_op) not in self.children:
            self.children[id(child_op)] = 1
        else:
            self.children[id(child_op)] += 1

    def all_gradients_accounted_for(self):
        """Check if this tensor has received all gradients from children."""
        for count in self.children.values():
            if count > 0:
                return False
        return True

    def backward(self, grad=None):
        """
        Computes the gradient of this tensor.
        
        If this tensor is the end of the computation graph (e.g., loss), grad can be omitted.
        """
        if not self.requires_grad:
            raise RuntimeError("This tensor is not marked for gradient computation.")
        
        if grad is None:
            grad = Tensor(np.ones_like(self.data))
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        # Proceed with backpropagation if all gradients from children are received
        if self.creator is not None and self.all_gradients_accounted_for():
            if isinstance(self.creator, Operation):
                self.creator.backward(self.grad)
            else:
                # Handle case where creator is not an operation (for flexibility)
                pass

    def zero_grad(self):
        """Resets the gradient of the tensor to None."""
        self.grad = None

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"



#-------------- Operation registry --------------#

operation_registry = {}

def register_operation(op_name):
    def decorator(cls):
        operation_registry[op_name] = cls
        return cls
    return decorator


#-------------- Operation base class --------------#

class Operation:
    def __call__(self, *input_tensors):
        """
        Executes the operation, performing the forward pass, and sets up for the backward pass.
        """
        self.input_tensors = input_tensors
        self.output_tensor = self.forward(*input_tensors)
        if self.output_tensor.requires_grad:
            self.output_tensor.creator = self
            for input_tensor in input_tensors:
                if isinstance(input_tensor, Tensor) and input_tensor.requires_grad:
                    input_tensor.add_child(self)
        return self.output_tensor

    def forward(self, *input_tensors):
        """Defines the forward pass of the operation. Must be implemented by subclasses."""
        raise NotImplementedError

    def backward(self, grad_output):
        """Defines the backward pass of the operation. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def all_gradients_accounted_for(self):
        return all(tensor.grad is not None for tensor in self.input_tensors)


#-------------- Arithmetic operations --------------#

#-------------- Addition --------------#

@register_operation('add')
class Add(Operation):
    def forward(self, a, b):
        return Tensor(a.data + b.data, requires_grad=True)

    def backward(self, grad_output):
        a, b = self.input_tensors
        if a.requires_grad:
            a.backward(grad_output)
        if b.requires_grad:
            b.backward(grad_output)

#-------------- Multiplication --------------#

@register_operation('mul')
class Mul(Operation):
    def forward(self, a, b):
        return Tensor(a.data * b.data, requires_grad=True)

    def backward(self, grad_output):
        a, b = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output * b.data
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = grad_output * a.data
            b.backward(grad_b)

#-------------- Subtraction --------------#

@register_operation('sub')
class Sub(Operation):
    def forward(self, a, b):
        return Tensor(a.data - b.data, requires_grad=True)

    def backward(self, grad_output):
        a, b = self.input_tensors
        if a.requires_grad:
            a.backward(grad_output)
        if b.requires_grad:
            b.backward(-grad_output)

#-------------- Division --------------#

@register_operation('div')
class Div(Operation):
    def forward(self, a, b):
        return Tensor(a.data / b.data, requires_grad=True)

    def backward(self, grad_output):
        a, b = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output / b.data
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = grad_output * -a.data / (b.data ** 2)
            b.backward(grad_b)

#-------------- Power --------------#

@register_operation('pow')
class Pow(Operation):
    def forward(self, a, exponent):
        # Assume exponent is a scalar for simplicity
        assert isinstance(exponent, (int, float)), "Exponent must be a scalar"
        return Tensor(a.data ** exponent, requires_grad=True)

    def backward(self, grad_output):
        a, exponent = self.input_tensors[0], self.input_tensors[1]
        if a.requires_grad:
            grad_a = grad_output * exponent * (a.data ** (exponent - 1))
            a.backward(grad_a)

#-------------- Exponential --------------#

@register_operation('exp')
class Exp(Operation):
    def forward(self, a):
        return Tensor(np.exp(a.data), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output * np.exp(a.data)
            a.backward(grad_a)

#-------------- Logarithm --------------#

@register_operation('log')
class Log(Operation):
    def forward(self, a):
        return Tensor(np.log(a.data), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output / a.data
            a.backward(grad_a)

#-------------- Sine --------------#

@register_operation('sin')
class Sin(Operation):
    def forward(self, a):
        return Tensor(np.sin(a.data), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output * np.cos(a.data)
            a.backward(grad_a)

#-------------- Cosine --------------#

@register_operation('cos')
class Cos(Operation):
    def forward(self, a):
        return Tensor(np.cos(a.data), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output * -np.sin(a.data)
            a.backward(grad_a)

#-------------- Tanh --------------#

@register_operation('tanh')
class Tanh(Operation):
    def forward(self, a):
        return Tensor(np.tanh(a.data), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output * (1 - np.tanh(a.data) ** 2)
            a.backward(grad_a)

#-------------- ReLU --------------#

@register_operation('relu')
class ReLU(Operation):
    def forward(self, a):
        return Tensor(np.maximum(0, a.data), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output * (a.data > 0)
            a.backward(grad_a)

#-------------- Softmax --------------#

@register_operation('softmax')
class Softmax(Operation):
    def forward(self, a):
        exps = np.exp(a.data - np.max(a.data, axis=-1, keepdims=True))
        return Tensor(exps / np.sum(exps, axis=-1, keepdims=True), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            s = self.output_tensor.data
            grad_a = grad_output * s * (1 - s)
            a.backward(grad_a)

#-------------- Mean squared error --------------#

@register_operation('mse')
class MeanSquaredError(Operation):
    def forward(self, a, b):
        return Tensor(np.mean((a.data - b.data) ** 2), requires_grad=True)

    def backward(self, grad_output):
        a, b = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output * 2 * (a.data - b.data) / len(a.data)
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = grad_output * -2 * (a.data - b.data) / len(a.data)
            b.backward(grad_b)

#-------------- Cross-entropy loss --------------#

@register_operation('cross_entropy')
class CrossEntropyLoss(Operation):
    def forward(self, a, b):
        return Tensor(-np.sum(b.data * np.log(a.data + 1e-8)) / len(a.data), requires_grad=True)

    def backward(self, grad_output):
        a, b = self.input_tensors
        if a.requires_grad:
            grad_a = grad_output * -b.data / (a.data + 1e-8) / len(a.data)
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = grad_output * -np.log(a.data + 1e-8) / len(a.data)
            b.backward(grad_b)

#-------------- Matrix multiplication --------------#

@register_operation('matmul')
class MatMul(Operation):
    def forward(self, a, b):
        return Tensor(np.matmul(a.data, b.data), requires_grad=True)

    def backward(self, grad_output):
        a, b = self.input_tensors
        if a.requires_grad:
            grad_a = np.matmul(grad_output, b.data.T)
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = np.matmul(a.data.T, grad_output)
            b.backward(grad_b)

#-------------- Transpose --------------#
            
@register_operation('transpose')
class Transpose(Operation):
    def forward(self, a):
        return Tensor(np.transpose(a.data), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.transpose(grad_output)
            a.backward(grad_a)

#-------------- Reshape --------------#
class Reshape(Operation):
    def forward(self, a, shape):
        return Tensor(np.reshape(a.data, shape), requires_grad=True)

    def backward(self, grad_output):
        a, shape = self.input_tensors
        if a.requires_grad:
            grad_a = np.reshape(grad_output, a.data.shape)
            a.backward(grad_a)

#-------------- Sum --------------#

@register_operation('sum')
class Sum(Operation):
    def forward(self, a, axis=None, keepdims=False):
        return Tensor(np.sum(a.data, axis=axis, keepdims=keepdims), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.broadcast_to(grad_output, a.data.shape)
            a.backward(grad_a)

#-------------- Mean --------------#

@register_operation('mean')
class Mean(Operation):
    def forward(self, a, axis=None, keepdims=False):
        return Tensor(np.mean(a.data, axis=axis, keepdims=keepdims), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.broadcast_to(grad_output, a.data.shape) / np.prod(a.data.shape)
            a.backward(grad_a)

#-------------- Concatenate --------------#

@register_operation('concatenate')
class Concatenate(Operation):
    def forward(self, a, b, axis):
        return Tensor(np.concatenate((a.data, b.data), axis=axis), requires_grad=True)

    def backward(self, grad_output):
        a, b = self.input_tensors
        if a.requires_grad:
            grad_a = np.split(grad_output, [a.data.shape[axis]], axis=axis)[0]
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = np.split(grad_output, [a.data.shape[axis]], axis=axis)[1]
            b.backward(grad_b)

#-------------- Stack --------------#

@register_operation('stack')
class Stack(Operation):
    def forward(self, tensors, axis):
        return Tensor(np.stack([t.data for t in tensors], axis=axis), requires_grad=True)

    def backward(self, grad_output):
        tensors, = self.input_tensors
        for tensor in tensors:
            if tensor.requires_grad:
                grad_tensor = np.take(grad_output, [i for i in range(grad_output.shape[axis])], axis=axis)
                tensor.backward(grad_tensor)

#-------------- Squeeze --------------#

@register_operation('squeeze')
class Squeeze(Operation):
    def forward(self, a, axis=None):
        return Tensor(np.squeeze(a.data, axis=axis), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.broadcast_to(grad_output, a.data.shape)
            a.backward(grad_a)

#-------------- Unsqueeze --------------#

@register_operation('unsqueeze')
class Unsqueeze(Operation):
    def forward(self, a, axis):
        return Tensor(np.expand_dims(a.data, axis=axis), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.sum(grad_output, axis=axis)
            a.backward(grad_a)

#-------------- Indexing --------------#

@register_operation('index')
class Index(Operation):
    def forward(self, a, indices):
        return Tensor(a.data[indices], requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.zeros_like(a.data)
            np.put(grad_a, indices, grad_output)
            a.backward(grad_a)

#-------------- Reduction --------------#

@register_operation('reduce')
class Reduce(Operation):
    def forward(self, a, axis, reduction):
        return Tensor(reduction(a.data, axis=axis), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.broadcast_to(grad_output, a.data.shape)
            a.backward(grad_a)

#-------------- Broadcast --------------#

@register_operation('broadcast')
class Broadcast(Operation):
    def forward(self, a, shape):
        return Tensor(np.broadcast_to(a.data, shape), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.sum(grad_output, axis=tuple(range(grad_output.ndim - a.data.ndim)))
            a.backward(grad_a)

#-------------- Tile --------------#

@register_operation('tile')
class Tile(Operation):
    def forward(self, a, reps):
        return Tensor(np.tile(a.data, reps), requires_grad=True)

    def backward(self, grad_output):
        a, = self.input_tensors
        if a.requires_grad:
            grad_a = np.sum(np.reshape(grad_output, a.data.shape + (grad_output.size // a.data.size,)), axis=-1)
            a.backward(grad_a)



#-----------------------------------------------------#
#------------------- Model operations -----------------#

#-------------- Convolution --------------#

@register_operation('conv2d')
class Conv2D(Operation):
    def forward(self, input, weight, bias, stride, padding):
        # Assume input and weight are 4D tensors
        batch_size, in_channels, in_height, in_width = input.data.shape
        out_channels, _, kernel_height, kernel_width = weight.data.shape

        # Compute output dimensions
        out_height = (in_height - kernel_height + 2 * padding) // stride + 1
        out_width = (in_width - kernel_width + 2 * padding) // stride + 1

        # Pad input
        input_padded = np.pad(input.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

        # Initialize output
        output = np.zeros((batch_size, out_channels, out_height, out_width))

        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                output[:, :, i, j] = np.sum(input_padded[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width] * weight.data, axis=(2, 3)) + bias.data

        return Tensor(output, requires_grad=True)
    
    def backward(self, grad_output):
        # Assume input and weight are 4D tensors
        input, weight, bias = self.input_tensors[0], self.input_tensors[1], self.input_tensors[2]
        stride, padding = self.input_tensors[3], self.input_tensors[4]
        batch_size, in_channels, in_height, in_width = input.data.shape
        out_channels, _, kernel_height, kernel_width = weight.data.shape
        _, _, out_height, out_width = grad_output.shape

        # Initialize gradients
        grad_input = np.zeros_like(input.data)
        grad_weight = np.zeros_like(weight.data)
        grad_bias = np.sum(grad_output, axis=(0, 2, 3))

        # Pad input
        input_padded = np.pad(input.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        grad_input_padded = np.pad(grad_input, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

        # Compute gradients
        for i in range(out_height):
            for j in range(out_width):
                grad_input_padded[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width] += np.expand_dims(np.expand_dims(grad_output[:, :, i, j], -1), -1) * weight.data
                grad_weight += np.sum(np.expand_dims(np.expand_dims(grad_output[:, :, i, j], -1), -1) * input_padded[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width], axis=0)

        # Remove padding from gradients
        grad_input = grad_input_padded[:, :, padding:-padding, padding:-padding]

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight.requires_grad:
            weight.backward(Tensor(grad_weight))
        if bias.requires_grad:
            bias.backward(Tensor(grad_bias))


    
#-------------- Convolution transpose --------------#
            
@register_operation('conv_transpose2d')
class ConvTranspose2D(Operation):
    def forward(self, input, weight, bias, stride, padding):
        # Assume input and weight are 4D tensors
        batch_size, in_channels, in_height, in_width = input.data.shape
        out_channels, _, kernel_height, kernel_width = weight.data.shape

        # Compute output dimensions
        out_height = (in_height - 1) * stride + kernel_height - 2 * padding
        out_width = (in_width - 1) * stride + kernel_width - 2 * padding

        # Initialize output
        output = np.zeros((batch_size, out_channels, out_height, out_width))

        # Perform convolution transpose
        for i in range(in_height):
            for j in range(in_width):
                output[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width] += np.sum(input.data[:, :, i, j][:, None, None, None] * weight.data, axis=1)

        return Tensor(output + bias.data, requires_grad=True)
    
    def backward(self, grad_output):
        # Assume input and weight are 4D tensors
        input, weight, bias = self.input_tensors[0], self.input_tensors[1], self.input_tensors[2]
        stride, padding = self.input_tensors[3], self.input_tensors[4]
        batch_size, in_channels, in_height, in_width = input.data.shape
        out_channels, _, kernel_height, kernel_width = weight.data.shape
        _, _, out_height, out_width = grad_output.shape

        # Initialize gradients
        grad_input = np.zeros_like(input.data)
        grad_weight = np.zeros_like(weight.data)
        grad_bias = np.sum(grad_output, axis=(0, 2, 3))

        # Compute gradients
        for i in range(in_height):
            for j in range(in_width):
                grad_input[:, :, i, j] += np.sum(grad_output[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width] * weight.data, axis=(1, 2, 3))
                grad_weight += np.sum(input.data[:, :, i, j][:, None, None, None] * grad_output[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width], axis=0)

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight.requires_grad:
            weight.backward(Tensor(grad_weight))
        if bias.requires_grad:
            bias.backward(Tensor(grad_bias))
    


#-------------- Max pooling --------------#
    
@register_operation('max_pool2d')
class MaxPool2D(Operation):
    def forward(self, input, kernel_size, stride, padding):
        # Assume input is a 4D tensor
        batch_size, in_channels, in_height, in_width = input.data.shape
        kernel_height, kernel_width = kernel_size

        # Compute output dimensions
        out_height = (in_height - kernel_height + 2 * padding) // stride + 1
        out_width = (in_width - kernel_width + 2 * padding) // stride + 1

        # Pad input
        input_padded = np.pad(input.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

        # Initialize output
        output = np.zeros((batch_size, in_channels, out_height, out_width))

        # Perform max pooling
        for i in range(out_height):
            for j in range(out_width):
                output[:, :, i, j] = np.max(input_padded[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width], axis=(2, 3))

        return Tensor(output, requires_grad=True)
    
    def backward(self, grad_output):
        # Assume input is a 4D tensor
        input = self.input_tensors[0]
        kernel_size, stride, padding = self.input_tensors[1], self.input_tensors[2], self.input_tensors[3]
        batch_size, in_channels, in_height, in_width = input.data.shape
        kernel_height, kernel_width = kernel_size
        _, _, out_height, out_width = grad_output.shape

        # Initialize gradients
        grad_input = np.zeros_like(input.data)

        # Pad input
        input_padded = np.pad(input.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        grad_input_padded = np.pad(grad_input, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

        # Compute gradients
        for i in range(out_height):
            for j in range(out_width):
                input_slice = input_padded[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width]
                grad_output_slice = np.expand_dims(np.expand_dims(grad_output[:, :, i, j], -1), -1)
                grad_input_padded[:, :, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width] += grad_output_slice * (input_slice == np.max(input_slice, axis=(2, 3), keepdims=True))

        # Remove padding from gradients
        grad_input = grad_input_padded[:, :, padding:-padding, padding:-padding]

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        
#-------------- Flatten --------------#

@register_operation('flatten')
class Flatten(Operation):
    def forward(self, input):
        return Tensor(np.reshape(input.data, (input.data.shape[0], -1)), requires_grad=True)

    def backward(self, grad_output):
        input, = self.input_tensors
        if input.requires_grad:
            grad_input = np.reshape(grad_output, input.data.shape)
            input.backward(Tensor(grad_input))

#-------------- Linear --------------#

@register_operation('linear')
class Linear(Operation):
    def forward(self, input, weight, bias):
        return Tensor(np.dot(input.data, weight.data.T) + bias.data, requires_grad=True)

    def backward(self, grad_output):
        input, weight, bias = self.input_tensors
        if input.requires_grad:
            grad_input = np.dot(grad_output, weight.data)
            input.backward(Tensor(grad_input))
        if weight.requires_grad:
            grad_weight = np.dot(grad_output.T, input.data)
            weight.backward(Tensor(grad_weight))
        if bias.requires_grad:
            grad_bias = np.sum(grad_output, axis=0)
            bias.backward(Tensor(grad_bias))
    
#-------------- Dropout --------------#

@register_operation('dropout')
class Dropout(Operation):
    def forward(self, input, p, training):
        if training:
            mask = np.random.rand(*input.data.shape) < p
            return Tensor(input.data * mask / p, requires_grad=True)
        else:
            return Tensor(input.data, requires_grad=True)

    def backward(self, grad_output):
        input, = self.input_tensors
        if input.requires_grad:
            input.backward(Tensor(grad_output))

#-------------- Batch normalization --------------#

@register_operation('batch_norm')
class BatchNorm(Operation):
    def forward(self, input, weight, bias, running_mean, running_var, training, momentum, eps):
        # Assume input is a 4D tensor
        batch_size, in_channels, in_height, in_width = input.data.shape

        # Compute mean and variance
        if training:
            mean = np.mean(input.data, axis=(0, 2, 3), keepdims=True)
            var = np.var(input.data, axis=(0, 2, 3), keepdims=True)
            running_mean.data = momentum * running_mean.data + (1 - momentum) * mean
            running_var.data = momentum * running_var.data + (1 - momentum) * var
        else:
            mean, var = running_mean.data, running_var.data

        # Normalize input
        input_normalized = (input.data - mean) / np.sqrt(var + eps)

        # Scale and shift
        output = input_normalized * weight.data + bias.data

        return Tensor(output, requires_grad=True)
    
    def backward(self, grad_output):
        input, weight, bias, running_mean, running_var, training, momentum, eps = self.input_tensors
        batch_size, in_channels, in_height, in_width = input.data.shape

        # Compute mean and variance
        if training:
            mean = np.mean(input.data, axis=(0, 2, 3), keepdims=True)
            var = np.var(input.data, axis=(0, 2, 3), keepdims=True)
            running_mean.data = momentum * running_mean.data + (1 - momentum) * mean
            running_var.data = momentum * running_var.data + (1 - momentum) * var
        else:
            mean, var = running_mean.data, running_var.data

        # Normalize input
        input_normalized = (input.data - mean) / np.sqrt(var + eps)

        # Compute gradients
        grad_input_normalized = grad_output.data * weight.data
        grad_var = np.sum(grad_input_normalized * (input.data - mean) * -0.5 * (var + eps) ** -1.5, axis=(0, 2, 3), keepdims=True)
        grad_mean = np.sum(grad_input_normalized * -1 / np.sqrt(var + eps), axis=(0, 2, 3), keepdims=True)
        grad_input = grad_input_normalized / np.sqrt(var + eps) + grad_var * 2 * (input.data - mean) / batch_size + grad_mean / batch_size
        grad_weight = np.sum(grad_output.data * input_normalized, axis=(0, 2, 3), keepdims=True)
        grad_bias = np.sum(grad_output.data, axis=(0, 2, 3), keepdims=True)

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight.requires_grad:
            weight.backward(Tensor(grad_weight))
        if bias.requires_grad:
            bias.backward(Tensor(grad_bias))

#-------------- Layer normalization --------------#

@register_operation('layer_norm')
class LayerNorm(Operation):
    def forward(self, input, weight, bias, eps):
        # Assume input is a 4D tensor
        batch_size, in_channels, in_height, in_width = input.data.shape

        # Compute mean and variance
        mean = np.mean(input.data, axis=(1, 2, 3), keepdims=True)
        var = np.var(input.data, axis=(1, 2, 3), keepdims=True)

        # Normalize input
        input_normalized = (input.data - mean) / np.sqrt(var + eps)

        # Scale and shift
        output = input_normalized * weight.data + bias.data

        return Tensor(output, requires_grad=True)
    
    def backward(self, grad_output):
        input, weight, bias, eps = self.input_tensors
        batch_size, in_channels, in_height, in_width = input.data.shape

        # Compute mean and variance
        mean = np.mean(input.data, axis=(1, 2, 3), keepdims=True)
        var = np.var(input.data, axis=(1, 2, 3), keepdims=True)

        # Normalize input
        input_normalized = (input.data - mean) / np.sqrt(var + eps)

        # Compute gradients
        grad_input_normalized = grad_output.data * weight.data
        grad_var = np.sum(grad_input_normalized * (input.data - mean) * -0.5 * (var + eps) ** -1.5, axis=(1, 2, 3), keepdims=True)
        grad_mean = np.sum(grad_input_normalized * -1 / np.sqrt(var + eps), axis=(1, 2, 3), keepdims=True)
        grad_input = grad_input_normalized / np.sqrt(var + eps) + grad_var * 2 * (input.data - mean) / in_channels + grad_mean / in_channels
        grad_weight = np.sum(grad_output.data * input_normalized, axis=(1, 2, 3), keepdims=True)
        grad_bias = np.sum(grad_output.data, axis=(1, 2, 3), keepdims=True)

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight.requires_grad:
            weight.backward(Tensor(grad_weight))
        if bias.requires_grad:
            bias.backward(Tensor(grad_bias))

#-------------- Embedding --------------#

@register_operation('embedding')
class Embedding(Operation):
    def forward(self, input, weight):
        return Tensor(weight.data[input.data], requires_grad=True)

    def backward(self, grad_output):
        input, weight = self.input_tensors
        if weight.requires_grad:
            grad_weight = np.zeros_like(weight.data)
            np.add.at(grad_weight, input.data, grad_output.data)
            weight.backward(Tensor(grad_weight))

#-------------- RNN --------------#

@register_operation('rnn')
class RNN(Operation):
    def forward(self, input, hidden, weight_ih, weight_hh, bias_ih, bias_hh):
        # Assume input and hidden are 3D tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = hidden.data.shape[-1]

        # Initialize output and hidden state
        output = np.zeros((batch_size, seq_length, hidden_size))
        hidden_states = np.zeros((batch_size, hidden_size))

        # Perform RNN
        for t in range(seq_length):
            hidden_states = np.tanh(np.dot(input.data[:, t], weight_ih.data.T) + bias_ih.data + np.dot(hidden_states, weight_hh.data.T) + bias_hh.data)
            output[:, t] = hidden_states

        return Tensor(output, requires_grad=True)
    
    def backward(self, grad_output):
        # Assume input and hidden are 3D tensors
        input, hidden, weight_ih, weight_hh, bias_ih, bias_hh = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = hidden.data.shape[-1]

        # Initialize gradients
        grad_input = np.zeros_like(input.data)
        grad_hidden = np.zeros_like(hidden.data)
        grad_weight_ih = np.zeros_like(weight_ih.data)
        grad_weight_hh = np.zeros_like(weight_hh.data)
        grad_bias_ih = np.zeros_like(bias_ih.data)
        grad_bias_hh = np.zeros_like(bias_hh.data)

        # Initialize hidden state gradients
        grad_hidden_states = np.zeros((batch_size, hidden_size))

        # Perform RNN backward pass
        for t in reversed(range(seq_length)):
            grad_hidden_states += grad_output.data[:, t]
            grad_hidden_states_tanh = grad_hidden_states * (1 - np.tanh(np.dot(input.data[:, t], weight_ih.data.T) + bias_ih.data + np.dot(hidden.data[:, t], weight_hh.data.T) + bias_hh.data) ** 2)
            grad_input[:, t] = np.dot(grad_hidden_states_tanh, weight_ih.data)
            grad_weight_ih += np.dot(grad_hidden_states_tanh.T, input.data[:, t])
            grad_hidden[:, t] = np.dot(grad_hidden_states_tanh, weight_hh.data)
            grad_weight_hh += np.dot(grad_hidden_states_tanh.T, hidden.data[:, t])
            grad_bias_ih += np.sum(grad_hidden_states_tanh, axis=0)
            grad_bias_hh += np.sum(grad_hidden_states_tanh, axis=0)
            grad_hidden_states = np.dot(grad_hidden_states_tanh, weight_hh.data)

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if hidden.requires_grad:
            hidden.backward(Tensor(grad_hidden))
        if weight_ih.requires_grad:
            weight_ih.backward(Tensor(grad_weight_ih))
        if weight_hh.requires_grad:
            weight_hh.backward(Tensor(grad_weight_hh))
        if bias_ih.requires_grad:
            bias_ih.backward(Tensor(grad_bias_ih))
        if bias_hh.requires_grad:
            bias_hh.backward(Tensor(grad_bias_hh))

#-------------- LSTM --------------#
    
@register_operation('lstm')
class LSTM(Operation):
    def forward(self, input, hidden, cell, weight_ih, weight_hh, bias_ih, bias_hh):
        # Assume input, hidden, and cell are 3D tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = hidden.data.shape[-1]

        # Initialize output, hidden state, and cell state
        output = np.zeros((batch_size, seq_length, hidden_size))
        hidden_states = np.zeros((batch_size, hidden_size))
        cell_states = np.zeros((batch_size, hidden_size))

        # Perform LSTM
        for t in range(seq_length):
            input_gate = np.sigmoid(np.dot(input.data[:, t], weight_ih.data[0].T) + bias_ih.data[0] + np.dot(hidden_states, weight_hh.data[0].T) + bias_hh.data[0])
            forget_gate = np.sigmoid(np.dot(input.data[:, t], weight_ih.data[1].T) + bias_ih.data[1] + np.dot(hidden_states, weight_hh.data[1].T) + bias_hh.data[1])
            cell_update = np.tanh(np.dot(input.data[:, t], weight_ih.data[2].T) + bias_ih.data[2] + np.dot(hidden_states, weight_hh.data[2].T) + bias_hh.data[2])
            cell_states = forget_gate * cell_states + input_gate * cell_update
            output_gate = np.sigmoid(np.dot(input.data[:, t], weight_ih.data[3].T) + bias_ih.data[3] + np.dot(hidden_states, weight_hh.data[3].T) + bias_hh.data[3])
            hidden_states = output_gate * np.tanh(cell_states)
            output[:, t] = hidden_states

        return Tensor(output, requires_grad=True)
    
    def backward(self, grad_output):
        # Assume input, hidden, and cell are 3D tensors
        input, hidden, cell, weight_ih, weight_hh, bias_ih, bias_hh = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = hidden.data.shape[-1]

        # Initialize gradients
        grad_input = np.zeros_like(input.data)
        grad_hidden = np.zeros_like(hidden.data)
        grad_cell = np.zeros_like(cell.data)
        grad_weight_ih = np.zeros_like(weight_ih.data)
        grad_weight_hh = np.zeros_like(weight_hh.data)
        grad_bias_ih = np.zeros_like(bias_ih.data)
        grad_bias_hh = np.zeros_like(bias_hh.data)

        # Initialize hidden state and cell state gradients
        grad_hidden_states = np.zeros((batch_size, hidden_size))
        grad_cell_states = np.zeros((batch_size, hidden_size))

        # Perform LSTM backward pass
        for t in reversed(range(seq_length)):
            grad_hidden_states += grad_output.data[:, t]
            grad_hidden_states_tanh = grad_hidden_states * (1 - np.tanh(np.dot(input.data[:, t], weight_ih.data[3].T) + bias_ih.data[3] + np.dot(hidden.data[:, t], weight_hh.data[3].T) + bias_hh.data[3]) ** 2)
            grad_cell_states += grad_hidden_states_tanh * np.sigmoid(np.dot(input.data[:, t], weight_ih.data[3].T) + bias_ih.data[3] + np.dot(hidden.data[:, t], weight_hh.data[3].T) + bias_hh.data[3]) * (1 - np.tanh(cell.data[:, t]) ** 2)
            grad_input_gate = grad_cell_states * grad_output.data[:, t] * np.tanh(np.dot(input.data[:, t], weight_ih.data[2].T) + bias_ih.data[2] + np.dot(hidden.data[:, t], weight_hh.data[2].T) + bias_hh.data[2])
            grad_forget_gate = grad_cell_states * cell.data[:, t] * grad_output.data[:, t] * np.sigmoid(np.dot(input.data[:, t], weight_ih.data[1].T) + bias_ih.data[1] + np.dot(hidden.data[:, t], weight_hh.data[1].T) + bias_hh.data[1]) * (1 - np.sigmoid(np.dot(input.data[:, t], weight_ih.data[1].T) + bias_ih.data[1] + np.dot(hidden.data[:, t], weight_hh.data[1].T) + bias_hh.data[1])
            grad_cell_update = grad_cell_states * grad_output.data[:, t] * np.sigmoid(np.dot(input.data[:, t], weight_ih.data[0].T) + bias_ih.data[0] + np.dot(hidden.data[:, t], weight_hh.data[0].T) + bias_hh.data[0]) * (1 - np.tanh(np.dot(input.data[:, t], weight_ih.data[2].T) + bias_ih.data[2] + np.dot(hidden.data[:, t], weight_hh.data[2].T) + bias_hh.data[2]) ** 2)
            grad_output_gate = grad_hidden_states * np.tanh(cell.data[:, t]) * grad_output.data[:, t] * np.sigmoid(np.dot(input.data[:, t], weight_ih.data[3].T) + bias_ih.data[3] + np.dot(hidden.data[:, t], weight_hh.data[3].T) + bias_hh.data[3]) * (1 - np.sigmoid(np.dot(input.data[:, t], weight_ih.data[3].T) + bias_ih.data[3] + np.dot(hidden.data[:, t], weight_hh.data[3].T) + bias_hh.data[3]) * np.tanh(np.dot(input.data[:, t], weight_ih.data[3].T) + bias_ih.data[3] + np.dot(hidden.data[:, t], weight_hh.data[3].T) + bias_hh.data[3]) ** 2)
            grad_input[:, t] = np.dot(grad_input_gate, weight_ih.data[0])
            grad_weight_ih[0] += np.dot(grad_input_gate.T, input.data[:, t])
            grad_hidden[:, t] = np.dot(grad_input_gate, weight_hh.data[0])
            grad_weight_hh[0] += np.dot(grad_input_gate.T, hidden.data[:, t])
            grad_bias_ih[0] += np.sum(grad_input_gate, axis=0)
            grad_bias_hh[0] += np.sum(grad_input_gate, axis=0)
            grad_input[:, t] += np.dot(grad_forget_gate, weight_ih.data[1])
            grad_weight_ih[1] += np.dot(grad_forget_gate.T, input.data[:, t])
            grad_hidden[:, t] += np.dot(grad_forget_gate, weight_hh.data[1])
            grad_weight_hh[1] += np.dot(grad_forget_gate.T, hidden.data[:, t])
            grad_bias_ih[1] += np.sum(grad_forget_gate, axis=0)
            grad_bias_hh[1] += np.sum(grad_forget_gate, axis=0)
            grad_input[:, t] += np.dot(grad_cell_update, weight_ih.data[2])
            grad_weight_ih[2] += np.dot(grad_cell_update.T, input.data[:, t])
            grad_hidden[:, t] += np.dot(grad_cell_update, weight_hh.data[2])
            grad_weight_hh[2] += np.dot(grad_cell_update.T, hidden.data[:, t])
            grad_bias_ih[2] += np.sum(grad_cell_update, axis=0)
            grad_bias_hh[2] += np.sum(grad_cell_update, axis=0)
            grad_input[:, t] += np.dot(grad_output_gate, weight_ih.data[3])
            grad_weight_ih[3] += np.dot(grad_output_gate.T, input.data[:, t])
            grad_hidden[:, t] += np.dot(grad_output_gate, weight_hh.data[3])
            grad_weight_hh[3] += np.dot(grad_output_gate.T, hidden.data[:, t])
            grad_bias_ih[3] += np.sum(grad_output_gate, axis=0)
            grad_bias_hh[3] += np.sum(grad_output_gate, axis=0)
            grad_hidden_states = np.dot(grad_output_gate, weight_hh.data[3])

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if hidden.requires_grad:
            hidden.backward(Tensor(grad_hidden))
        if cell.requires_grad:
            cell.backward(Tensor(grad_cell))
        if weight_ih.requires_grad:
            weight_ih.backward(Tensor(grad_weight_ih))
        if weight_hh.requires_grad:
            weight_hh.backward(Tensor(grad_weight_hh))
        if bias_ih.requires_grad:
            bias_ih.backward(Tensor(grad_bias_ih))
        if bias_hh.requires_grad:
            bias_hh.backward(Tensor(grad_bias_hh))

#-------------- GRU --------------#

@register_operation('gru')
class GRU(Operation):
    def forward(self, input, hidden, weight_ih, weight_hh, bias_ih, bias_hh):
        # Assume input and hidden are 3D tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = hidden.data.shape[-1]

        # Initialize output and hidden state
        output = np.zeros((batch_size, seq_length, hidden_size))
        hidden_states = np.zeros((batch_size, hidden_size))

        # Perform GRU
        for t in range(seq_length):
            reset_gate = np.sigmoid(np.dot(input.data[:, t], weight_ih.data[0].T) + bias_ih.data[0] + np.dot(hidden_states, weight_hh.data[0].T) + bias_hh.data[0])
            update_gate = np.sigmoid(np.dot(input.data[:, t], weight_ih.data[1].T) + bias_ih.data[1] + np.dot(hidden_states, weight_hh.data[1].T) + bias_hh.data[1])
            hidden_states = (1 - update_gate) * np.tanh(np.dot(input.data[:, t], weight_ih.data[2].T) + bias_ih.data[2] + np.dot(reset_gate * hidden_states, weight_hh.data[2].T) + bias_hh.data[2]) + update_gate * hidden_states
            output[:, t] = hidden_states

        return Tensor(output, requires_grad=True)
    
    def backward(self, grad_output):
        # Assume input and hidden are 3D tensors
        input, hidden, weight_ih, weight_hh, bias_ih, bias_hh = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = hidden.data.shape[-1]

        # Initialize gradients
        grad_input = np.zeros_like(input.data)
        grad_hidden = np.zeros_like(hidden.data)
        grad_weight_ih = np.zeros_like(weight_ih.data)
        grad_weight_hh = np.zeros_like(weight_hh.data)
        grad_bias_ih = np.zeros_like(bias_ih.data)
        grad_bias_hh = np.zeros_like(bias_hh.data)

        # Initialize hidden state gradients
        grad_hidden_states = np.zeros((batch_size, hidden_size))

        # Perform GRU backward pass
        for t in reversed(range(seq_length)):
            grad_hidden_states += grad_output.data[:, t]
            grad_hidden_states_tanh = grad_hidden_states * (1 - np.tanh(np.dot(input.data[:, t], weight_ih.data[2].T) + bias_ih.data[2] + np.dot(hidden.data[:, t], weight_hh.data[2].T) + bias_hh.data[2]) ** 2)
            grad_update_gate = grad_hidden_states * (np.tanh(np.dot(input.data[:, t], weight_ih.data[2].T) + bias_ih.data[2] + np.dot(hidden.data[:, t], weight_hh.data[2].T) + bias_hh.data[2] - hidden.data[:, t]) * np.sigmoid(np.dot(input.data[:, t], weight_ih.data[1].T) + bias_ih.data[1] + np.dot(hidden.data[:, t], weight_hh.data[1].T) + bias_hh.data[1]) * (1 - np.sigmoid(np.dot(input.data[:, t], weight_ih.data[1].T) + bias_ih.data[1] + np.dot(hidden.data[:, t], weight_hh.data[1].T) + bias_hh.data[1]))
            grad_reset_gate = grad_hidden_states * (np.dot(input.data[:, t], weight_ih.data[2].T) + bias_ih.data[2] + np.dot(hidden.data[:, t], weight_hh.data[2].T) + bias_hh.data[2] - hidden.data[:, t]) * np.tanh(hidden.data[:, t]) * np.sigmoid(np.dot(input.data[:, t], weight_ih.data[0].T) + bias_ih.data[0] + np.dot(hidden.data[:, t], weight_hh.data[0].T) + bias_hh.data[0]) * (1 - np.sigmoid(np.dot(input.data[:, t], weight_ih.data[0].T) + bias_ih.data[0] + np.dot(hidden.data[:, t], weight_hh.data[0].T) + bias_hh.data[0])
            grad_input[:, t] = np.dot(grad_update_gate, weight_ih.data[1]) + np.dot(grad_reset_gate, weight_ih.data[0]) + np.dot(grad_hidden_states_tanh, weight_ih.data[2])
            grad_weight_ih[0] += np.dot(grad_reset_gate.T, input.data[:, t])
            grad_weight_ih[1] += np.dot(grad_update_gate.T, input.data[:, t])
            grad_weight_ih[2] += np.dot(grad_hidden_states_tanh.T, input.data[:, t])
            grad_hidden[:, t] = np.dot(grad_update_gate, weight_hh.data[1]) + np.dot(grad_reset_gate, weight_hh.data[0]) + np.dot(grad_hidden_states_tanh, weight_hh.data[2])
            grad_weight_hh[0] += np.dot(grad_reset_gate.T, hidden.data[:, t])
            grad_weight_hh[1] += np.dot(grad_update_gate.T, hidden.data[:, t])
            grad_weight_hh[2] += np.dot(grad_hidden_states_tanh.T, hidden.data[:, t])
            grad_bias_ih[0] += np.sum(grad_reset_gate, axis=0)
            grad_bias_ih[1] += np.sum(grad_update_gate, axis=0)
            grad_bias_ih[2] += np.sum(grad_hidden_states_tanh, axis=0)
            grad_bias_hh[0] += np.sum(grad_reset_gate, axis=0)
            grad_bias_hh[1] += np.sum(grad_update_gate, axis=0)
            grad_bias_hh[2] += np.sum(grad_hidden_states_tanh, axis=0)
            grad_hidden_states = np.dot(grad_update_gate, weight_hh.data[1]) + np.dot(grad_reset_gate, weight_hh.data[0]) + np.dot(grad_hidden_states_tanh, weight_hh.data[2])

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if hidden.requires_grad:
            hidden.backward(Tensor(grad_hidden))
        if weight_ih.requires_grad:
            weight_ih.backward(Tensor(grad_weight_ih))
        if weight_hh.requires_grad:
            weight_hh.backward(Tensor(grad_weight_hh))
        if bias_ih.requires_grad:
            bias_ih.backward(Tensor(grad_bias_ih))
        if bias_hh.requires_grad:
            bias_hh.backward(Tensor(grad_bias_hh))

#-------------- Transformer --------------#

@register_operation('transformer')
class Transformer(Operation):
    def forward(self, input, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k, bias_v, bias_o):
        # Assume input is a 3D tensor
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_o.data.shape[1]

        # Compute queries, keys, and values
        queries = np.dot(input.data, weight_q.data.T) + bias_q.data
        keys = np.dot(input.data, weight_k.data.T) + bias_k.data
        values = np.dot(input.data, weight_v.data.T) + bias_v.data

        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 2, 1)) / np.sqrt(hidden_size)
        attention_probs = np.softmax(attention_scores, axis=-1)

        # Compute output
        output = np.dot(attention_probs, values) + bias_o.data

        return Tensor(output, requires_grad=True)

    def backward(self, grad_output):
        # Assume input is a 3D tensor
        input, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k, bias_v, bias_o = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_o.data.shape[1]

        # Compute queries, keys, and values
        queries = np.dot(input.data, weight_q.data.T) + bias_q.data
        keys = np.dot(input.data, weight_k.data.T) + bias_k.data
        values = np.dot(input.data, weight_v.data.T) + bias_v.data

        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 2, 1)) / np.sqrt(hidden_size)
        attention_probs = np.softmax(attention_scores, axis=-1)

        # Compute output
        output = np.dot(attention_probs, values) + bias_o.data

        # Compute gradients
        grad_attention_probs = np.matmul(grad_output.data, values.transpose(0, 2, 1))
        grad_values = np.matmul(grad_output.data.transpose(0, 2, 1), attention_probs)
        grad_queries = np.matmul(grad_attention_probs, keys)
        grad_keys = np.matmul(grad_attention_probs.transpose(0, 2, 1), queries)

        grad_input = np.dot(grad_queries, weight_q.data)
        grad_weight_q = np.dot(queries.transpose(0, 2, 1), grad_queries)
        grad_weight_k = np.dot(keys.transpose(0, 2, 1), grad_keys)
        grad_weight_v = np.dot(values.transpose(0, 2, 1), grad_values)
        grad_weight_o = np.sum(np.matmul(attention_probs.transpose(0, 2, 1), grad_output.data), axis=0)
        grad_bias_q = np.sum(grad_queries, axis=0)
        grad_bias_k = np.sum(grad_keys, axis=0)
        grad_bias_v = np.sum(grad_values, axis=0)
        grad_bias_o = np.sum(grad_output.data, axis=(0, 1))

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight_q.requires_grad:
            weight_q.backward(Tensor(grad_weight_q))
        if weight_k.requires_grad:
            weight_k.backward(Tensor(grad_weight_k))
        if weight_v.requires_grad:
            weight_v.backward(Tensor(grad_weight_v))
        if weight_o.requires_grad:
            weight_o.backward(Tensor(grad_weight_o))
        if bias_q.requires_grad:
            bias_q.backward(Tensor(grad_bias_q))
        if bias_k.requires_grad:    
            bias_k.backward(Tensor(grad_bias_k))
        if bias_v.requires_grad:
            bias_v.backward(Tensor(grad_bias_v))
        if bias_o.requires_grad:    
            bias_o.backward(Tensor(grad_bias_o))
            
#-------------- Attention --------------#

@register_operation('attention')
class Attention(Operation):
    def forward(self, input, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k, bias_v, bias_o):
        # Assume input is a 3D tensor
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_o.data.shape[1]

        # Compute queries, keys, and values
        queries = np.dot(input.data, weight_q.data.T) + bias_q.data
        keys = np.dot(input.data, weight_k.data.T) + bias_k.data
        values = np.dot(input.data, weight_v.data.T) + bias_v.data

        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 2, 1)) / np.sqrt(hidden_size)
        attention_probs = np.softmax(attention_scores, axis=-1)

        # Compute output
        output = np.dot(attention_probs, values) + bias_o.data

        return Tensor(output, requires_grad=True)

    def backward(self, grad_output):
        # Assume input is a 3D tensor
        input, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k, bias_v, bias_o = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_o.data.shape[1]

        # Compute queries, keys, and values
        queries = np.dot(input.data, weight_q.data.T) + bias_q.data
        keys = np.dot(input.data, weight_k.data.T) + bias_k.data
        values = np.dot(input.data, weight_v.data.T) + bias_v.data

        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 2, 1)) / np.sqrt(hidden_size)
        attention_probs = np.softmax(attention_scores, axis=-1)

        # Compute output
        output = np.dot(attention_probs, values) + bias_o.data

        # Compute gradients
        grad_attention_probs = np.matmul(grad_output.data, values.transpose(0, 2, 1))
        grad_values = np.matmul(grad_output.data.transpose(0, 2, 1), attention_probs)
        grad_queries = np.matmul(grad_attention_probs, keys)
        grad_keys = np.matmul(grad_attention_probs.transpose(0, 2, 1), queries)

        grad_input = np.dot(grad_queries, weight_q.data)
        grad_weight_q = np.dot(queries.transpose(0, 2, 1), grad_queries)
        grad_weight_k = np.dot(keys.transpose(0, 2, 1), grad_keys)
        grad_weight_v = np.dot(values.transpose(0, 2, 1), grad_values)
        grad_weight_o = np.sum(np.matmul(attention_probs.transpose(0, 2, 1), grad_output.data), axis=0)
        grad_bias_q = np.sum(grad_queries, axis=0)
        grad_bias_k = np.sum(grad_keys, axis=0)
        grad_bias_v = np.sum(grad_values, axis=0)
        grad_bias_o = np.sum(grad_output.data, axis=(0, 1))

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight_q.requires_grad:
            weight_q.backward(Tensor(grad_weight_q))
        if weight_k.requires_grad:
            weight_k.backward(Tensor(grad_weight_k))
        if weight_v.requires_grad:
            weight_v.backward(Tensor(grad_weight_v))
        if weight_o.requires_grad:
            weight_o.backward(Tensor(grad_weight_o))
        if bias_q.requires_grad:
            bias_q.backward(Tensor(grad_bias_q))
        if bias_k.requires_grad:
            bias_k.backward(Tensor(grad_bias_k))
        if bias_v.requires_grad:
            bias_v.backward(Tensor(grad_bias_v))
        if bias_o.requires_grad:
            bias_o.backward(Tensor(grad_bias_o))

#-------------- Self-attention --------------#
    
@register_operation('self_attention')
class SelfAttention(Operation):
    def forward(self, input, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k, bias_v, bias_o):
        # Assume input is a 3D tensor
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_o.data.shape[1]

        # Compute queries, keys, and values
        queries = np.dot(input.data, weight_q.data.T) + bias_q.data
        keys = np.dot(input.data, weight_k.data.T) + bias_k.data
        values = np.dot(input.data, weight_v.data.T) + bias_v.data

        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 2, 1)) / np.sqrt(hidden_size)
        attention_probs = np.softmax(attention_scores, axis=-1)

        # Compute output
        output = np.dot(attention_probs, values) + bias_o.data

        return Tensor(output, requires_grad=True)

    def backward(self, grad_output):
        # Assume input is a 3D tensor
        input, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k, bias_v, bias_o = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_o.data.shape[1]

        # Compute queries, keys, and values
        queries = np.dot(input.data, weight_q.data.T) + bias_q.data
        keys = np.dot(input.data, weight_k.data.T) + bias_k.data
        values = np.dot(input.data, weight_v.data.T) + bias_v.data

        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 2, 1)) / np.sqrt(hidden_size)
        attention_probs = np.softmax(attention_scores, axis=-1)

        # Compute output
        output = np.dot(attention_probs, values) + bias_o.data

        # Compute gradients
        grad_attention_probs = np.matmul(grad_output.data, values.transpose(0, 2, 1))
        grad_values = np.matmul(grad_output.data.transpose(0, 2, 1), attention_probs)
        grad_queries = np.matmul(grad_attention_probs, keys)
        grad_keys = np.matmul(grad_attention_probs.transpose(0, 2, 1), queries)

        grad_input = np.dot(grad_queries, weight_q.data)
        grad_weight_q = np.dot(queries.transpose(0, 2, 1), grad_queries)
        grad_weight_k = np.dot(keys.transpose(0, 2, 1), grad_keys)
        grad_weight_v = np.dot(values.transpose(0, 2, 1), grad_values)
        grad_weight_o = np.sum(np.matmul(attention_probs.transpose(0, 2, 1), grad_output.data), axis=0)
        grad_bias_q = np.sum(grad_queries, axis=0)
        grad_bias_k = np.sum(grad_keys, axis=0)
        grad_bias_v = np.sum(grad_values, axis=0)
        grad_bias_o = np.sum(grad_output.data, axis=(0, 1))

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight_q.requires_grad:
            weight_q.backward(Tensor(grad_weight_q))
        if weight_k.requires_grad:  
            weight_k.backward(Tensor(grad_weight_k))
        if weight_v.requires_grad:
            weight_v.backward(Tensor(grad_weight_v))
        if weight_o.requires_grad:
            weight_o.backward(Tensor(grad_weight_o))
        if bias_q.requires_grad:
            bias_q.backward(Tensor(grad_bias_q))
        if bias_k.requires_grad:
            bias_k.backward(Tensor(grad_bias_k))
        if bias_v.requires_grad:
            bias_v.backward(Tensor(grad_bias_v))
        if bias_o.requires_grad:
            bias_o.backward(Tensor(grad_bias_o))

#-------------- Multi-head attention --------------#
    
@register_operation('multi_head_attention')
class MultiHeadAttention(Operation):
    def forward(self, input, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k, bias_v, bias_o, num_heads):
        # Assume input is a 3D tensor
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_o.data.shape[1]

        # Compute queries, keys, and values
        queries = np.dot(input.data, weight_q.data.T) + bias_q.data
        keys = np.dot(input.data, weight_k.data.T) + bias_k.data
        values = np.dot(input.data, weight_v.data.T) + bias_v.data

        # Split heads
        queries = np.reshape(queries, (batch_size, seq_length, num_heads, hidden_size // num_heads))
        keys = np.reshape(keys, (batch_size, seq_length, num_heads, hidden_size // num_heads))
        values = np.reshape(values, (batch_size, seq_length, num_heads, hidden_size // num_heads))

        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 1, 3, 2)) / np.sqrt(hidden_size // num_heads)
        attention_probs = np.softmax(attention_scores, axis=-1)

        # Compute output
        output = np.matmul(attention_probs, values)
        output = np.reshape(output, (batch_size, seq_length, hidden_size))

        # Compute output
        output = np.dot(output, weight_o.data.T) + bias_o.data

        return Tensor(output, requires_grad=True)

    def backward(self, grad_output):
        # Assume input is a 3D tensor
        input, weight_q, weight_k, weight_v, weight_o, bias_q, bias_k, bias_v, bias_o, num_heads = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_o.data.shape[1]

        # Compute queries, keys, and values
        queries = np.dot(input.data, weight_q.data.T) + bias_q.data
        keys = np.dot(input.data, weight_k.data.T) + bias_k.data
        values = np.dot(input.data, weight_v.data.T) + bias_v.data

        # Split heads
        queries = np.reshape(queries, (batch_size, seq_length, num_heads, hidden_size // num_heads))
        keys = np.reshape(keys, (batch_size, seq_length, num_heads, hidden_size // num_heads))
        values = np.reshape(values, (batch_size, seq_length, num_heads, hidden_size // num_heads))

        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 1, 3, 2)) / np.sqrt(hidden_size // num_heads)
        attention_probs = np.softmax(attention_scores, axis=-1)

        # Compute output
        output = np.matmul(attention_probs, values)
        output = np.reshape(output, (batch_size, seq_length, hidden_size))

        # Compute output
        output = np.dot(output, weight_o.data.T) + bias_o.data

        # Compute gradients
        grad_output = np.reshape(grad_output, (batch_size, seq_length, num_heads, hidden_size // num_heads))
        grad_attention_probs = np.matmul(grad_output, values.transpose(0, 1, 3, 2))
        grad_values = np.matmul(grad_output.transpose(0, 1, 3, 2), attention_probs)
        grad_queries = np.matmul(grad_attention_probs, keys)
        grad_keys = np.matmul(grad_attention_probs.transpose(0, 1, 3, 2), queries)

        grad_input = np.dot(grad_queries, weight_q.data)
        grad_weight_q = np.dot(queries.transpose(0, 1, 3, 2), grad_queries)
        grad_weight_k = np.dot(keys.transpose(0, 1, 3, 2), grad_keys)
        grad_weight_v = np.dot(values.transpose(0, 1, 3, 2), grad_values)
        grad_weight_o = np.sum(np.matmul(attention_probs.transpose(0, 1, 3, 2), grad_output), axis=0)
        grad_bias_q = np.sum(grad_queries, axis=0)
        grad_bias_k = np.sum(grad_keys, axis=0)
        grad_bias_v = np.sum(grad_values, axis=0)
        grad_bias_o = np.sum(grad_output, axis=(0, 1))

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight_q.requires_grad:
            weight_q.backward(Tensor(grad_weight_q))
        if weight_k.requires_grad:
            weight_k.backward(Tensor(grad_weight_k))
        if weight_v.requires_grad:
            weight_v.backward(Tensor(grad_weight_v))
        if weight_o.requires_grad:  
            weight_o.backward(Tensor(grad_weight_o))
        if bias_q.requires_grad:
            bias_q.backward(Tensor(grad_bias_q))
        if bias_k.requires_grad:
            bias_k.backward(Tensor(grad_bias_k))
        if bias_v.requires_grad:
            bias_v.backward(Tensor(grad_bias_v))
        if bias_o.requires_grad:
            bias_o.backward(Tensor(grad_bias_o))

#-------------- Positional encoding --------------#
    
@register_operation('positional_encoding')
class PositionalEncoding(Operation):
    def forward(self, input, max_length, hidden_size):
        # Assume input is a 3D tensor
        batch_size, seq_length, input_size = input.data.shape

        # Initialize positional encoding
        positional_encoding = np.zeros((max_length, hidden_size))

        # Compute positional encoding
        for pos in range(max_length):
            for i in range(0, hidden_size, 2):
                positional_encoding[pos, i] = np.sin(pos / 10000 ** (2 * i / hidden_size))
                positional_encoding[pos, i + 1] = np.cos(pos / 10000 ** (2 * (i + 1) / hidden_size))

        # Add positional encoding to input
        output = input.data + positional_encoding[:seq_length]

        return Tensor(output, requires_grad=True)

    def backward(self, grad_output):
        # Assume input is a 3D tensor
        input, max_length, hidden_size = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape

        # Initialize positional encoding
        positional_encoding = np.zeros((max_length, hidden_size))

        # Compute positional encoding
        for pos in range(max_length):
            for i in range(0, hidden_size, 2):
                positional_encoding[pos, i] = np.sin(pos / 10000 ** (2 * i / hidden_size))
                positional_encoding[pos, i + 1] = np.cos(pos / 10000 ** (2 * (i + 1) / hidden_size))

        # Add positional encoding to input
        output = input.data + positional_encoding[:seq_length]

        # Compute gradients
        grad_input = grad_output.data

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))

#-------------- Feedforward --------------#

@register_operation('feedforward')
class FeedForward(Operation):
    def forward(self, input, weight_1, weight_2, bias_1, bias_2):
        # Assume input is a 3D tensor
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_1.data.shape[1]

        # Compute output
        output = np.dot(np.tanh(np.dot(input.data, weight_1.data.T) + bias_1.data), weight_2.data.T) + bias_2.data

        return Tensor(output, requires_grad=True)

    def backward(self, grad_output):
        # Assume input is a 3D tensor
        input, weight_1, weight_2, bias_1, bias_2 = self.input_tensors
        batch_size, seq_length, input_size = input.data.shape
        hidden_size = weight_1.data.shape[1]

        # Compute output
        output = np.dot(np.tanh(np.dot(input.data, weight_1.data.T) + bias_1.data), weight_2.data.T) + bias_2.data

        # Compute gradients
        grad_output = np.dot(grad_output.data, weight_2.data)
        grad_weight_2 = np.dot(output.transpose(0, 2, 1), grad_output)
        grad_bias_2 = np.sum(grad_output, axis=(0, 1))
        grad_output = np.dot(grad_output, weight_2.data.T)
        grad_output = grad_output * (1 - np.tanh(np.dot(input.data, weight_1.data.T) + bias_1.data) ** 2)
        grad_weight_1 = np.dot(input.data.transpose(0, 2, 1), grad_output)
        grad_bias_1 = np.sum(grad_output, axis=(0, 1))
        grad_input = np.dot(grad_output, weight_1.data)

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if weight_1.requires_grad:
            weight_1.backward(Tensor(grad_weight_1))
        if weight_2.requires_grad:
            weight_2.backward(Tensor(grad_weight_2))
        if bias_1.requires_grad:
            bias_1.backward(Tensor(grad_bias_1))
        if bias_2.requires_grad:
            bias_2.backward(Tensor(grad_bias_2))

#-------------- Residual connection --------------#
    
@register_operation('residual')
class Residual(Operation):
    def forward(self, input, residual):
        # Assume input and residual are 3D tensors
        output = input.data + residual.data

        return Tensor(output, requires_grad=True)

    def backward(self, grad_output):
        # Assume input and residual are 3D tensors
        input, residual = self.input_tensors

        # Compute gradients
        grad_input = grad_output.data
        grad_residual = grad_output.data

        # Backpropagate gradients
        if input.requires_grad:
            input.backward(Tensor(grad_input))
        if residual.requires_grad:
            residual.backward(Tensor(grad_residual))
            

#-----------------------------------------------------#

if __name__ == "__main__":

    #-------------- Test --------------#

    # Create tensors
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)

    # Perform operations
    c = add(a, b)
    d = mul(c, b)

    # Backward pass
    d.backward()

    print(a.grad)  # Should print gradients w.r.t a
    print(b.grad)  # Should print gradients w.r.t b

    #-----------------------------------#