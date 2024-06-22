from .optimizer import Optimizer
import numpy as np

class SGD(Optimizer):

    def __init__(self, parameters, lr=0.01,  momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in self.parameters]

    def step(self, batch_size=1):
        for param, velocity in zip(self.parameters, self.velocities):
            # print("param : ", param, type(param), type(param.grad))
            # print("self.lr : ", self.lr)
            # print("param.grad : ", param.grad.shape, param.grad, "batch_size : ", batch_size)
            if param.requires_grad and param.grad is not None:
               # Average the gradient by dividing by batch size if necessary
                if param.grad.shape != param.data.shape:
                    # If param.grad has a batch dimension, average it
                    averaged_grad = np.mean(param.grad, axis=0)
                else:
                    # Otherwise, it's likely a bias term or already averaged
                    averaged_grad = param.grad / batch_size
                # print("averaged_grad :", averaged_grad.shape, type(averaged_grad), averaged_grad)

                # Update the velocity
                velocity *= self.momentum
                velocity += self.lr * averaged_grad
                # print("velocity : ", velocity.shape, velocity)
                # Update the parameter using the velocity vector
                param.data -= velocity
                # print("param.data :", param.data.shape, param.data)

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()
