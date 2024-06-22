"""
optimizers.py

This module provides various optimization algorithms commonly used in machine learning and deep learning models.
Each optimizer is implemented from scratch to help learners understand the underlying mechanics.
"""

import numpy as np

import torch

class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, lr_decay=0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                if self.weight_decay != 0:
                    param.grad = param.grad + self.weight_decay * param.data
                if self.momentum != 0:
                    self.velocities[i] = self.momentum * self.velocities[i] + param.grad
                    update = self.lr * self.velocities[i]
                else:
                    update = self.lr * param.grad
                param.data -= update
        if self.lr_decay != 0:
            self.lr *= (1.0 / (1.0 + self.lr_decay))

# Usage example with mini-batch
# params = [torch.tensor(10.0, requires_grad=True), torch.tensor(-3.0, requires_grad=True)]
# optimizer = SGD(params, lr=0.1, momentum=0.9, weight_decay=0.01, lr_decay=0.001)

# # Dummy data for mini-batch
# inputs = torch.randn(100, 2)
# targets = torch.randn(100, 1)
# dataset = TensorDataset(inputs, targets)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# # Training loop with mini-batch gradient descent
# for epoch in range(100):
#     for batch_inputs, batch_targets in dataloader:
#         optimizer.zero_grad()
#         outputs = model(batch_inputs)
#         loss = criterion(outputs, batch_targets)
#         loss.backward()
#         optimizer.step()

# print(params)


#-------------------------------------------------------------------------------------------

class Momentum(Optimizer):
    """
    SGD with Momentum optimizer.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        """
        Update parameters using SGD with momentum.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.

        Returns:
            dict: Updated parameters.
        """
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            params[key] += self.velocity[key]
        return params

#-------------------------------------------------------------------------------------------

class NAG(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG) optimizer.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        """
        Update parameters using NAG.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.

        Returns:
            dict: Updated parameters.
        """
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] *= self.momentum
            self.velocity[key] -= self.learning_rate * grads[key]
            params[key] += self.momentum * self.momentum * self.velocity[key] - (1 + self.momentum) * self.learning_rate * grads[key]
        return params
    
#-------------------------------------------------------------------------------------------

class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    """
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads):
        """
        Update parameters using RMSprop.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.

        Returns:
            dict: Updated parameters.
        """
        for key in params.keys():
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            self.cache[key] = self.rho * self.cache[key] + (1 - self.rho) * grads[key] ** 2
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        return params

#-------------------------------------------------------------------------------------------

class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads):
        """
        Update parameters using AdaGrad.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.

        Returns:
            dict: Updated parameters.
        """
        for key in params.keys():
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            self.cache[key] += grads[key] ** 2
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        return params
    
#-------------------------------------------------------------------------------------------

class AdaDelta(Optimizer):
    """
    AdaDelta optimizer.
    """
    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon
        self.E_g = {}
        self.E_dx = {}

    def update(self, params, grads):
        """
        Update parameters using AdaDelta.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.

        Returns:
            dict: Updated parameters.
        """
        for key in params.keys():
            if key not in self.E_g:
                self.E_g[key] = np.zeros_like(params[key])
                self.E_dx[key] = np.zeros_like(params[key])
            
            self.E_g[key] = self.rho * self.E_g[key] + (1 - self.rho) * grads[key] ** 2
            dx = -np.sqrt(self.E_dx[key] + self.epsilon) * grads[key] / np.sqrt(self.E_g[key] + self.epsilon)
            params[key] += dx
            self.E_dx[key] = self.rho * self.E_dx[key] + (1 - self.rho) * dx ** 2
        return params
    
#-------------------------------------------------------------------------------------------

class Adamax(Optimizer):
    """
    Adamax optimizer.
    """
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.u = {}
        self.t = 0

    def update(self, params, grads):
        """
        Update parameters using Adamax.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.

        Returns:
            dict: Updated parameters.
        """
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.u[key] = np.zeros_like(params[key])
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.u[key] = np.maximum(self.beta2 * self.u[key], np.abs(grads[key]))
            
            params[key] -= self.learning_rate * self.m[key] / (1 - self.beta1 ** self.t) / (self.u[key] + self.epsilon)
        return params

#-------------------------------------------------------------------------------------------

class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        """
        Update parameters using Adam.

        Args:
            params (np.ndarray or dict): Parameters to be updated.
            grads (np.ndarray or dict): Gradients for each parameter.

        Returns:
            np.ndarray or dict: Updated parameters.
        """
        self.t += 1
        if isinstance(params, dict):
            if self.m is None:
                self.m = {key: np.zeros_like(value) for key, value in params.items()}
                self.v = {key: np.zeros_like(value) for key, value in params.items()}
                
            for key in params.keys():
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
                
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            if self.m is None:
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)

            self.m = self.beta1 * self.m + (1 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params


#-------------------------------------------------------------------------------------------

class Nadam(Optimizer):
    """
    Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimizer.
    """
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        """
        Update parameters using Nadam.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.

        Returns:
            dict: Updated parameters.
        """
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.learning_rate * (self.beta1 * m_hat + (1 - self.beta1) * grads[key]) / (np.sqrt(v_hat) + self.epsilon)
        return params
    
#-------------------------------------------------------------------------------------------

#===========================================================================================================

# Example usage:
if __name__ == "__main__":
    params = {'w1': np.array([0.2, -0.5]), 'b1': np.array([0.1]), 'w2': np.array([0.3, -0.2]), 'b2': np.array([0.2])}
    grads = {'w1': np.array([0.1, -0.2]), 'b1': np.array([0.01]), 'w2': np.array([0.15, -0.1]), 'b2': np.array([0.05])}

    sgd = SGD(learning_rate=0.01)
    print("SGD update:\n", sgd.update(params.copy(), grads))

    momentum = Momentum(learning_rate=0.01, momentum=0.9)
    print("Momentum update:\n", momentum.update(params.copy(), grads))

    nag = NAG(learning_rate=0.01, momentum=0.9)
    print("NAG update:\n", nag.update(params.copy(), grads))

    rmsprop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-7)
    print("RMSprop update:\n", rmsprop.update(params.copy(), grads))

    adagrad = AdaGrad(learning_rate=0.01, epsilon=1e-7)
    print("AdaGrad update:\n", adagrad.update(params.copy(), grads))

    adadelta = AdaDelta(rho=0.95, epsilon=1e-6)
    print("AdaDelta update:\n", adadelta.update(params.copy(), grads))

    adamax = Adamax(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8)
    print("Adamax update:\n", adamax.update(params.copy(), grads))

    adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7)
    print("Adam update:\n", adam.update(params.copy(), grads))

    nadam = Nadam(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8)
    print("Nadam update:\n", nadam.update(params.copy(), grads))

#-------------------------------------------------------------------------------------------
