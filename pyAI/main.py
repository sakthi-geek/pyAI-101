import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy as TorchAccuracy
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset as TorchTensorDataset
import matplotlib.pyplot as plt
import os
import sys
import configparser
import logging
from argparse import ArgumentParser

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
print(project_root_dir)
sys.path.append(project_root_dir)

# Import your custom framework components
from pyAI.autograd.tensor import Tensor
from pyAI.nn.module import Module
from pyAI.nn.layers.linear import Linear
from pyAI.nn.activation import ReLU, Sigmoid
from pyAI.nn.loss import MSELoss, BCELoss, CrossEntropyLoss
from pyAI.optim.sgd import SGD
from pyAI.data.dataset import TensorDataset
from pyAI.data.dataloader import DataLoader
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pyAI.utils.metrics import Accuracy


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from file
config = configparser.ConfigParser()
config.read('pyAI/config.ini')

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=config.getint('DEFAULT', 'BATCH_SIZE', fallback=32))
parser.add_argument('--epochs', type=int, default=config.getint('DEFAULT', 'EPOCHS', fallback=100))
parser.add_argument('--learning_rate', type=float, default=config.getfloat('DEFAULT', 'LEARNING_RATE', fallback=0.01))
args = parser.parse_args()

# Centralized user-defined parameters
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
RANDOM_SEED = 42

print("BATCH_SIZE", BATCH_SIZE)
print("EPOCHS", EPOCHS)
print("LEARNING_RATE", LEARNING_RATE)
print("RANDOM_SEED", RANDOM_SEED)
      

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Define models for different tasks
class RegressionNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 16)
        self.relu = ReLU()
        self.fc2 = Linear(16, 1)
        self.add_parameter(self.fc1)
        self.add_parameter(self.fc2)

    def forward(self, x):
        x = self.relu.forward(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x

# PyTorch equivalent models
class TorchRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu.forward(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x


class BinaryClassificationNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(30, 16)  # Assuming 30 features in the dataset
        self.relu = ReLU()
        self.fc2 = Linear(16, 1)
        self.sigmoid = Sigmoid()
        self.add_parameter(self.fc1)
        self.add_parameter(self.fc2)

    def forward(self, x):
        x = self.relu.forward(self.fc1.forward(x))
        x = self.sigmoid.forward(self.fc2.forward(x))
        return x

class TorchBinaryClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu.forward(self.fc1.forward(x))
        x = self.sigmoid.forward(self.fc2.forward(x))
        return x

class MultiClassNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 10)  # Assuming 4 features in the dataset
        self.relu = ReLU()
        self.fc2 = Linear(10, 3)  # Assuming 3 classes
        self.add_parameter(self.fc1)
        self.add_parameter(self.fc2)

    def forward(self, x):
        x = self.relu.forward(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x

class TorchMultiClassNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.relu.forward(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x
    
def preprocess_data(data, dataset='fetch_california_housing', task_type="regression", scaler="standard_scaler", num_impute_strategy='median', 
                    cat_impute_strategy='constant', fill_value='missing'):
    
    tabular_datasets = ['fetch_california_housing', 'load_breast_cancer', 'load_iris']

    if dataset in tabular_datasets:

        X = data.data
        y = data.target

        print(type(X), type(y))
        print(X.shape, y.shape)
        print(data.feature_names)
        print(data.target_names)
        

        # Convert to DataFrame for easier manipulation
        X = pd.DataFrame(X, columns=data.feature_names)
        
        # Identify numerical and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=[object]).columns
        
        # Preprocessing for numerical data: imputation and scaling
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=num_impute_strategy)),                      # (strategy="mean")
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data: imputation and encoding
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=cat_impute_strategy, fill_value=fill_value)),  # (strategy="most_frequent")
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numeric_features),
                ('cat', cat_transformer, categorical_features)
            ])
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        print(y)

        if 'classification' in task_type:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)  # Convert to 0-based indices for classes

        print(y)

        if task_type == 'regression':
            y = y.reshape(-1, 1)  # Ensure y is [batch_size, 1] for regression task
        elif task_type == 'binary_classification':
            y = y.reshape(-1, 1)  # Ensure y is [batch_size, 1] for binary classification assuming BCELoss
        elif task_type == 'multi_class_classification':
            y = y.flatten()  # Ensure y is [batch_size] for multi-class classification assuming CrossEntropyLoss
        else:
            y = y.reshape(-1, 1)  # Default to regression task

        return X_processed, y

def prepare_dataset(data_loader, task_type='regression'):
    data = data_loader()
    X, y = preprocess_data(data, data_loader.__name__, task_type=task_type)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Convert to Tensors
    X_train = Tensor(X_train, requires_grad=True)
    y_train = Tensor(y_train, requires_grad=True)
    X_test = Tensor(X_test, requires_grad=True)
    y_test = Tensor(y_test, requires_grad=True)

    return X_train, X_test, y_train, y_test


def train(model, criterion, optimizer, X_train, y_train, epochs=EPOCHS):
    losses = []
    batch_size = BATCH_SIZE
    num_batches = X_train.shape[0] // batch_size
    
    for epoch in range(epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train.data[perm]
        y_shuffled = y_train.data[perm]

        total_loss = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = Tensor(X_shuffled[start:end], requires_grad=True)
            y_batch = Tensor(y_shuffled[start:end], requires_grad=True)

            optimizer.zero_grad()
            outputs = model.forward(X_batch)
            loss = criterion.forward(outputs, y_batch)
            loss.backward()
            optimizer.step(batch_size=batch_size)
            total_loss += loss.data.item()
            
        average_loss = total_loss / num_batches
        losses.append(average_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss:  {average_loss}')
    return losses

def evaluate_regression(model, criterion, X_test, y_test):
    X_test_tensor = Tensor(X_test.data, requires_grad=False)
    y_test_tensor = Tensor(y_test.data, requires_grad=False)
    
    predictions = model(X_test_tensor)
    
    mse = np.mean((predictions.data - y_test_tensor.data) ** 2)
    logging.info(f'Regression Loss (MSE): {mse}')
    
    return mse

def evaluate_binary_classification(model, criterion, X_test, y_test):
    X_test_tensor = Tensor(X_test.data, requires_grad=False)
    y_test_tensor = Tensor(y_test.data, requires_grad=False)
    
    outputs = model(X_test_tensor)
    loss = criterion(outputs, y_test_tensor)
    
    predicted_classes = (outputs.data > 0.5).astype(int)  # Assuming outputs are probabilities
    accuracy = np.mean(predicted_classes == y_test_tensor.data)
    logging.info(f'Validation Loss: {loss.data.item()}, Accuracy: {accuracy}')
    
    return loss.data.item(), accuracy

def evaluate_multi_class_classification(model, criterion, X_test, y_test):
    X_test_tensor = Tensor(X_test.data, requires_grad=False)
    y_test_tensor = Tensor(y_test.data, requires_grad=False)
    
    outputs = model(X_test_tensor)
    loss = criterion(outputs, y_test_tensor)
    
    predicted_classes = np.argmax(outputs.data, axis=1)
    accuracy = np.mean(predicted_classes == y_test_tensor.data)
    logging.info(f'Validation Loss: {loss.data.item()}, Accuracy: {accuracy}')
    
    return loss.data.item(), accuracy

def benchmark_pytorch(X_train, y_train, X_test, y_test, model, criterion, optimizer, epochs=100):
    
    X_train_torch = torch.tensor(X_train.data, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test.data, dtype=torch.float32)

    if criterion.__class__.__name__ == 'CrossEntropyLoss':
        y_train_torch = torch.tensor(y_train.data, dtype=torch.long)
        y_test_torch = torch.tensor(y_test.data, dtype=torch.long)
    else:
        y_train_torch = torch.tensor(y_train.data, dtype=torch.float32)  #expects float or long depending on the loss function
        y_test_torch = torch.tensor(y_test.data, dtype=torch.float32)   #expects float or long depending on the loss function

    losses = []
    batch_size = BATCH_SIZE
    num_batches = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train_torch.size(0))
        X_shuffled = X_train_torch[perm]
        y_shuffled = y_train_torch[perm]

        total_loss = 0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / num_batches
        losses.append(average_loss)

        if epoch % 10 == 0:
            print(f'[PyTorch] Epoch {epoch}, Loss: {average_loss}')

    return losses


def plot_results(custom_metrics, torch_metrics, labels, title, ylabel, save_path='plots'):
    # Ensure the directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure(figsize=(10, 5))
    for metric, label in zip([custom_metrics, torch_metrics], labels):
        plt.plot(metric, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_filename = os.path.join(save_path, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(plot_filename)
    plt.show()


def run_regression_task():

    # 1. Regression Task
    # Initialize datasets for regression using the California Housing dataset
    X_train, X_test, y_train, y_test = prepare_dataset(fetch_california_housing, task_type='regression')

    # Initialize models for Regression
    custom_model = RegressionNet()
    pytorch_model = TorchRegressionNet()

    # Initialize loss functions for Regression
    custom_criterion = MSELoss()
    pytorch_criterion = nn.MSELoss()

    # Initialize optimizers for Regression
    custom_optimizer = SGD(custom_model.parameters, lr=LEARNING_RATE, momentum=0.9)
    pytorch_optimizer = optim.SGD(pytorch_model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Train and evaluate custom model for Regression
    custom_losses = train(custom_model, custom_criterion, custom_optimizer, X_train, y_train, epochs=EPOCHS)
    custom_eval_loss = evaluate_regression(custom_model, custom_criterion, X_test, y_test)

    # Train and evaluate PyTorch model for Regression
    pytorch_losses = benchmark_pytorch(X_train, y_train, X_test, y_test, pytorch_model, pytorch_criterion, pytorch_optimizer, epochs=EPOCHS)

    # Plot and compare the results for Regression
    plot_results(custom_losses, pytorch_losses, ['Custom Implementation', 'PyTorch'], 'Regression Training Loss', 'Loss')

def run_binary_classification_task():

    # 2. Binary Classification Task
    # Initialize datasets for binary classification using the Breast Cancer dataset
    X_train, X_test, y_train, y_test = prepare_dataset(load_breast_cancer, task_type='binary_classification')

    # Initialize models for Binary Classification
    custom_model = BinaryClassificationNet()
    pytorch_model = TorchBinaryClassificationNet()

    # Initialize loss functions for Binary Classification
    custom_criterion = BCELoss()
    pytorch_criterion = nn.BCELoss()

    # Initialize optimizers for Binary Classification
    custom_optimizer = SGD(custom_model.parameters, lr=LEARNING_RATE, momentum=0.9)
    pytorch_optimizer = optim.SGD(pytorch_model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Train and evaluate custom model for Binary Classification
    custom_losses = train(custom_model, custom_criterion, custom_optimizer, X_train, y_train, epochs=EPOCHS)
    custom_eval_loss, custom_eval_accuracy = evaluate_binary_classification(custom_model, custom_criterion, X_test, y_test)

    # Train and evaluate PyTorch model for Binary Classification
    pytorch_losses = benchmark_pytorch(X_train, y_train, X_test, y_test, pytorch_model, pytorch_criterion, pytorch_optimizer, epochs=EPOCHS)

    # Plot and compare the results for Binary Classification
    plot_results(custom_losses, pytorch_losses, ['Custom Implementation', 'PyTorch'], 'Binary Classification Training Loss', 'Loss')

def run_multi_class_classification_task():

    # 3. Multi-Class Classification Task
    # Initialize datasets for multi-class classification using the Iris dataset
    X_train, X_test, y_train, y_test = prepare_dataset(load_iris, task_type='multi_class_classification')

    # Initialize models for Multi-Class Classification
    custom_model = MultiClassNet()
    pytorch_model = TorchMultiClassNet()

    # Initialize loss functions for Multi-Class Classification
    custom_criterion = CrossEntropyLoss()
    pytorch_criterion = nn.CrossEntropyLoss()

    # Initialize optimizers for Multi-Class Classification
    custom_optimizer = SGD(custom_model.parameters, lr=LEARNING_RATE, momentum=0.9) 
    pytorch_optimizer = optim.SGD(pytorch_model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Train and evaluate custom model for Multi-Class Classification
    custom_losses = train(custom_model, custom_criterion, custom_optimizer, X_train, y_train, epochs=EPOCHS)
    custom_eval_loss, custom_eval_accuracy = evaluate_multi_class_classification(custom_model, custom_criterion, X_test, y_test)

    # Train and evaluate PyTorch model for Multi-Class Classification
    pytorch_losses = benchmark_pytorch(X_train, y_train, X_test, y_test, pytorch_model, pytorch_criterion, pytorch_optimizer, epochs=EPOCHS)

    # Plot and compare the results for Multi-Class Classification
    plot_results(custom_losses, pytorch_losses, ['Custom Implementation', 'PyTorch'], 'Multi-Class Classification Training Loss', 'Loss')


def main():
    run_regression_task()
    run_binary_classification_task()
    run_multi_class_classification_task()
    # Final Output
    print("All tasks completed. Check the 'plots' directory for comparison plots.")


if __name__ == '__main__':
    main()

