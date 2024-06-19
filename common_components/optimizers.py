"""
optimizers.py

This module provides various optimization algorithms commonly used in machine learning and deep learning models.
Each optimizer is implemented from scratch to help learners understand the underlying mechanics.
"""

import numpy as np

class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        raise NotImplementedError("Update method not implemented!")

#-------------------------------------------------------------------------------------------

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def update(self, params, grads):
        """
        Update parameters using stochastic gradient descent.

        Args:
            params (np.ndarray or dict): Dictionary of parameters to be updated.
            grads (np.ndarray or dict): Dictionary of gradients for each parameter.

        Returns:
            np.ndarray or dict: Updated parameters.
        """
        if isinstance(params, dict):
            for key in params.keys():
                params[key] -= self.learning_rate * grads[key]
        else:
            params -= self.learning_rate * grads
        return params

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
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        """
        Update parameters using Adam.

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
            
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
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
