import json
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging
from torchmetrics import Accuracy
from utils import evaluate, plot_learning_curve, plot_metrics, split_data, train
from base_model import BaseModel


class Perceptron(BaseModel):

    def __init__(self, input_dim, output_dim, device=None):

        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        x = self.fc(x)  # Linear transformation
        out = torch.sigmoid(x)  # Sigmoid activation
        return out

    def predict(self, x):
        probabilities = torch.sigmoid(self.forward(x))
        predicted_class = (
            probabilities > 0.5
        ).float()  # Convert probabilities to binary class
        return predicted_class


def preprocess_data(X_train, X_val, X_test):
    """
    Standardize features by removing the mean and scaling to unit variance
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
        config = config["Perceptron"]

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Important Hyperparameters
    SEED = config.get("hyperparameters", {}).get("seed", 42)
    BATCH_SIZE = config.get("hyperparameters", {}).get("batch_size", 32)
    LEARNING_RATE = config.get("hyperparameters", {}).get("learning_rate", 0.001)
    DEVICE = config.get("hyperparameters", {}).get("device", "cuda")
    VAL_SIZE = config.get("data", {}).get("val_size", 0.15)
    TEST_SIZE = config.get("data", {}).get("test_size", 0.15)

    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    # ------------------------- Breast Cancer Dataset -------------------------#
    # Binary classification problem

    # Load data
    data = load_breast_cancer()
    logging.info(f"Features: {data.feature_names}")
    logging.info(f"Target: {data.target_names}")
    train_data, val_data, test_data = split_data(
        data.data,
        data.target,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    # Preprocess data
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)

    print("Number of training samples:", X_train.shape[0])
    print("Number of validation samples:", X_val.shape[0])
    print("Number of test samples:", X_test.shape[0])
    print(X_train.shape, y_train.shape)
    print(X_train[:5], y_train[:5])

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(
        1
    )  # unsqueeze to add an extra dimension for output
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]  # Number of features
    output_dim = 1  # Binary classification

    # Model, criterion, and optimizer
    model = Perceptron(input_dim, output_dim, device=DEVICE).to(DEVICE)
    criterion = nn.BCELoss().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Initialize metric
    accuracy = Accuracy(task="binary").to(
        DEVICE
    )  # ["binary", "multiclass", "multilabel"]
    # precision = Precision()

    # Train the model
    train_losses, val_losses, train_metrics, val_metrics = train(
        model,
        train_loader,
        val_loader,
        criterion,
        [accuracy],
        optimizer,
        epochs=100,
        device=DEVICE,
    )

    # Evaluate the model
    loss, test_metrics = evaluate(
        model, test_loader, criterion, [accuracy], device=DEVICE
    )

    # Plot learning curve
    plot_learning_curve(train_losses, val_losses, title="Perceptron Learning Curve")

    # Plot Metrics
    plot_metrics(train_metrics, val_metrics, title="Perceptron Metrics Plot")


if __name__ == "__main__":

    main("03_deep_learning/key_architectures/config.json")
