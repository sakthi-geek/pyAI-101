import sys
import os

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
print(project_root_dir)
sys.path.append(project_root_dir)

import numpy as np
from pyAI.autograd.tensor import Tensor
from pyAI.nn.layers.linear import Linear
from pyAI.nn.activation import ReLU
from pyAI.nn.loss import MSELoss
from pyAI.optim.sgd import SGD
from pyAI.nn.module import Module
from pyAI.utils.visualization import plot_loss
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


#---------------------------- TESTING AUTOGRAD -------------------------------------

# # Example usage
# a = Tensor([1, 2, 3], requires_grad=True)
# b = Tensor([4, 5, 6], requires_grad=True)

# print(a.shape, a.data)
# print(b.shape, b.data)

# c = a + b

# print(c.shape, c.data)  
# print(a.grad)  # Should print None
# print(b.grad)  # Should print None

# a.zero_grad()
# b.zero_grad()

# print(a.grad.shape, a.grad)  # Should print [0, 0, 0]
# print(b.grad.shape, b.grad)  # Should print [0, 0, 0]

# c.backward()

# print(a.grad.shape, a.grad)  # Should print the gradient of a 
# print(b.grad.shape, b.grad)  # Should print the gradient of b 

# print(" Addition operation verified")
# #----------------------------------------------

# print(a.shape)
# print(b.shape)

# d = a * b # Assuming a simple reshape method exists

# print(d.shape, d.data)  

# print(a.grad.shape, a.grad)  # Should print the gradient of a 
# print(b.grad.shape, b.grad)  # Should print the gradient of b 

# d.backward()

# print(a.grad.shape, a.grad)  # Should print the gradient of a 
# print(b.grad.shape, b.grad)  # Should print the gradient of b 

# print(" Multiplication operation verified")
# #------------------------------------------------------

# a1 = Tensor([[1, 2],
#             [3, 4],
#             [5, 6]], requires_grad=True)

# b1 = Tensor([[7],
#             [8]], requires_grad=True)

# print(a1.shape, a1.data)
# print(b1.shape, b1.data)

# e = a1 @ b1
# print(e.shape, e.data) 

# a1.zero_grad()
# b1.zero_grad()

# print(a1.grad.shape, a1.grad)  # Should print the gradient of a1 
# print(b1.grad.shape, b1.grad)  # Should print the gradient of b1 

# e.backward()

# print(a1.grad.shape, a1.grad)  # Should print the gradient of a1
# print(b1.grad.shape, b1.grad)  # Should print the gradient of b1 


# print(" Matrix multiplication operation verified")
#------------------------------------------------------------------------------------------------


# Load the California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

print(X.shape, y.shape)

# Normalize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to Tensors
X = Tensor(X, requires_grad=True)
y = Tensor(y.reshape(-1, 1), requires_grad=True)

print(X.shape, y.shape)
print(type(X), type(y))


# Model
class SimpleMLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 5)
        self.relu = ReLU()
        self.fc2 = Linear(5, 1)
        self.add_parameter(self.fc1)
        self.add_parameter(self.fc2)

    def forward(self, x):
        out = self.relu.forward(self.fc1.forward(x))
        out = self.fc2.forward(out)
        return out

model = SimpleMLP()
criterion = MSELoss()
print(model.parameters)
optimizer = SGD(model.parameters, lr=0.01, momentum=0)

# Training loop with mini-batch gradient descent
batch_size = 512
losses = []
num_batches = X.shape[0] // batch_size
# set seed
np.random.seed(42)

for epoch in range(1000):
    # Shuffle the data
    perm = np.random.permutation(X.shape[0])
    X_shuffled = X.data[perm]
    y_shuffled = y.data[perm]

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = Tensor(X_shuffled[start:end], requires_grad=False)
        y_batch = Tensor(y_shuffled[start:end], requires_grad=False)
        # print(X_batch.shape, y_batch.shape)

        optimizer.zero_grad()  # Reset gradients
        predictions = model.forward(X_batch)
        # print(predictions.shape)
        loss = criterion.forward(predictions, y_batch)
        loss.backward()
        optimizer.step(batch_size=batch_size)
        losses.append(loss.data.item())
    
    # Print average loss for the epoch
    avg_loss = np.mean(losses[-num_batches:])
    print(f"Epoch {epoch}, Loss: {avg_loss}")
   
print("-------------------")

import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch model for comparison
class SimpleMLP_Torch(nn.Module):
    def __init__(self):
        super(SimpleMLP_Torch, self).__init__()
        self.fc1 = nn.Linear(8, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

model_torch = SimpleMLP_Torch()
criterion_torch = nn.MSELoss()
optimizer_torch = optim.SGD(model_torch.parameters(), lr=0.01, momentum=0)

# Convert dataset to PyTorch tensors
X_torch = torch.tensor(X.data, dtype=torch.float32)
y_torch = torch.tensor(y.data, dtype=torch.float32)

# Training loop for PyTorch model with mini-batch gradient descent
losses_torch = []
# set seed
torch.manual_seed(42)

for epoch in range(1000):
    
    # Shuffle the data
    perm = torch.randperm(X_torch.size(0))
    X_shuffled = X_torch[perm]
    y_shuffled = y_torch[perm]

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        optimizer_torch.zero_grad()  # Reset gradients
        outputs = model_torch(X_batch)
        loss = criterion_torch(outputs, y_batch)
        loss.backward()
        optimizer_torch.step()
        losses_torch.append(loss.item())
    
    # Print average loss for the epoch
    avg_loss_torch = np.mean(losses_torch[-num_batches:])
    print(f"Epoch {epoch}, Loss: {avg_loss_torch}")

print("from scratch losses: ", losses)
print("torch losses: ", losses_torch)

# Compare losses
import matplotlib.pyplot as plt
plt.plot(losses, label='Custom Implementation', color='blue')
plt.plot(losses_torch, label='PyTorch Implementation', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
# save the comparison plot - write your code here - do not call any function
plt.savefig('pyAI/training_loss_comparison_plot-custom_vs_pytorch.png')
plt.show()



#================================================================================================

# def setup_data(dataset_loader, task_type='regression'):

#     data = dataset_loader()
#     X, y = preprocess_data(data, dataset_loader.__name__)

#     if task_type == 'classification':
#         y = y.astype(int)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
#     train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
#     test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     return train_loader, test_loader

# def setup_pytorch_data_loaders(train_loader, test_loader):
#     """ Convert custom DataLoader to PyTorch DataLoader for fair comparison. """

#     # Prepare containers for PyTorch-compatible tensors
#     train_features, train_labels = [], []
#     test_features, test_labels = [], []

#        # Process train_loader and test_loader
#     for loader, feature_list, label_list in [(train_loader, train_features, train_labels), (test_loader, test_features, test_labels)]:
#         for inputs, targets in loader:
#             try:
#                 # Convert inputs and targets to numpy arrays, ensuring type is float32 for inputs and appropriate for targets
#                 inputs_numpy = np.asarray(inputs.data, dtype=np.float32)
#                 targets_numpy = np.asarray(targets.data, dtype=np.float32 if targets.data.ndim > 1 else np.int64)

#                 # Check shapes and types
#                 if inputs_numpy.ndim == 1:
#                     inputs_numpy = inputs_numpy.reshape(1, -1)  # Reshape if single dimension array

#                 feature_list.append(torch.tensor(inputs_numpy))
#                 label_list.append(torch.tensor(targets_numpy))
#             except ValueError as e:
#                 logging.error(f"Error processing batch: {e}, inputs: {inputs.data}, targets: {targets.data}")
#                 raise

#     # Create PyTorch TensorDatasets from the list of tensors
#     train_dataset = TorchTensorDataset(torch.cat(train_features), torch.cat(train_labels))
#     test_dataset = TorchTensorDataset(torch.cat(test_features), torch.cat(test_labels))

#     # Create PyTorch DataLoaders from TensorDatasets
#     train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = TorchDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     return train_loader, test_loader