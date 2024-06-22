class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()







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

