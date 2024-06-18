

class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [0 for _ in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if self.momentum:
                self.velocities[i] = self.momentum * self.velocities[i] + param.grad
                param.data -= self.lr * self.velocities[i]
            else:
                param.data -= self.lr * param.grad


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.ms = [0 for _ in parameters]
        self.vs = [0 for _ in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            self.ms[i] = self.betas[0] * self.ms[i] + (1 - self.betas[0]) * param.grad
            self.vs[i] = self.betas[1] * self.vs[i] + (1 - self.betas[1]) * (param.grad ** 2)
            m_hat = self.ms[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.vs[i] / (1 - self.betas[1] ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    def __init__(self, parameters, lr=0.001, alpha=0.99, eps=1e-8):
        super(RMSprop, self).__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.sq_grads = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            self.sq_grads[i] = self.alpha * self.sq_grads[i] + (1 - self.alpha) * (param.grad ** 2)
            param.data -= self.lr * param.grad / (np.sqrt(self.sq_grads[i]) + self.eps)


class Adagrad(Optimizer):
    def __init__(self, parameters, lr=0.01, eps=1e-8):
        super(Adagrad, self).__init__(parameters, lr)
        self.eps = eps
        self.sum_sq_grads = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            self.sum_sq_grads[i] += param.grad ** 2
            param.data -= self.lr * param.grad / (np.sqrt(self.sum_sq_grads[i]) + self.eps)




#----------------------------------------------------------
# # Assuming model is your neural network instance and model.parameters()
# # returns a list of parameters (weights and biases)
# optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# # During the training loop
# optimizer.zero_grad()  # Clear previous gradients
# loss = compute_loss(model(input), target)  # Compute the loss
# loss.backward()  # Backpropagate to compute gradients
# optimizer.step()  # Update model parameters
#----------------------------------------------------------