import json
import pathlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchmetrics
from torchvision import datasets, transforms
import logging
from utils import (
    train,
    evaluate,
    plot_learning_curve,
    plot_metrics,
    save_model,
    load_model,
)
from base_model import BaseModel


class MLP(BaseModel):
    """
    Initializes the MLP model with given layer sizes.

    Args:
    layer_sizes (list of int): Defines the number of neurons in each layer including the input and output layers.
    device (torch.device, optional): The device on which to run the model (CPU or GPU).
    """

    def __init__(self, layer_sizes, device=None):

        super().__init__()

        # Creating neural network layers
        layers = []
        num_layers = len(layer_sizes) - 1
        for i in range(num_layers):
            # Add a fully connected layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add a ReLU activation function after each layer except the output layer
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        # Wrap all layers in a Sequential container
        self.layers = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image: from [B, 1, 28, 28] to [B, 784]
        return self.layers(x)  # Softmax is included in the CrossEntropyLoss

    def predict(self, x):
        probabilities = torch.softmax(self.forward(x), dim=1)  # Get class probabilities
        predicted_class = torch.argmax(probabilities, dim=1)  # Get predicted class
        return predicted_class


def preprocess_MNIST_data(batch_size, val_size=0.1, test_size=0.1):

    # Define the transform to normalize the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Normalize using the mean and std of MNIST
        ]
    )  # transforms.Lambda(lambda x: x.view(-1))  # Flatten the images

    # Load the MNIST dataset
    full_train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )  # Applied the transform before splitting - considered safe as they don't depend on the data split and are identical for every image
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    plot_MNIST_images(full_train_dataset, class_names=full_train_dataset.classes)

    # Split the full training dataset into training and validation sets
    val_size = int(val_size * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Setup DataLoaders with the collate function to apply transforms on-the-fly
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def plot_MNIST_images(dataset, class_names):
    fig = plt.figure(figsize=(16, 4))
    rows, cols = 2, 10

    for i in range(1, (rows * cols) + 1):
        rand_ind = torch.randint(0, len(dataset), size=[1]).item()
        img, label = dataset[rand_ind]
        img = (
            img.squeeze()
        )  # Squeeze the channel dimension or use img[0] if img is a tensor
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.numpy(), cmap="gray")
        plt.title(f"{class_names[label]}")
        plt.axis(False)
        plt.tight_layout()

    plt.show()


def main(config_path):
    with open(config_path) as f:
        config = json.load(f)["MLP"]

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
    EPOCHS = config.get("hyperparameters", {}).get("epochs", 100)
    LAYER_SIZES = config.get("hyperparameters", {}).get(
        "layer_sizes", [784, 128, 64, 10]
    )
    SKIP_TRAINING = config.get("skip_training", False)

    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    if not SKIP_TRAINING:

        # Load the MNIST dataset and preprocess it
        train_loader, val_loader, test_loader = preprocess_MNIST_data(
            BATCH_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE
        )

        print("Number of training samples:", len(train_loader.dataset))
        print("Number of validation samples:", len(val_loader.dataset))
        print("Number of test samples:", len(test_loader.dataset))
        print("Input shape:", train_loader.dataset[0][0].shape)
        print("Output:", train_loader.dataset[0][1])

        # Initialize the model, loss function, and optimizer
        model = MLP(LAYER_SIZES, device=DEVICE).to(DEVICE)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

        # Model summary
        logging.info(model)
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        # Initialize Metrics
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(DEVICE)
        f1_score = torchmetrics.F1Score(task="multiclass", num_classes=10).to(DEVICE)
        metrics = [accuracy, f1_score]

        # Training and evaluation
        train_losses, val_losses, train_metrics, val_metrics = train(
            model,
            train_loader,
            val_loader,
            criterion,
            metrics,
            optimizer,
            EPOCHS,
            early_stopping_patience=5,
            device=DEVICE,
        )

        test_loss, test_metrics = evaluate(
            model, test_loader, criterion, metrics, device=DEVICE
        )

        # Track metrics and save them to a file
        metrics = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        }
        with open("MLP_training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Save the trained model
        model_save_path = config.get("model_save_path", "model_artifacts/MLP.pth")
        pathlib.Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        save_model(model, model_save_path)
    else:
        # Load the trained model
        model_save_path = config.get("model_save_path", "model_artifacts/MLP.pth")

        model = MLP(LAYER_SIZES, device=DEVICE).to(DEVICE)
        model = load_model(model, model_save_path, device=DEVICE)

        # Predict on test data
        test_loader = preprocess_MNIST_data(
            BATCH_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE
        )[2]

        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            y_pred = model.predict(data)
            print("target:", target)
            print("predicted:", y_pred)
            break

        # Load saved metrics
        metrics_path = (
            "03_deep_learning/key_architectures/metrics/MLP_training_metrics.json"
        )
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        train_losses = metrics["train_loss"]
        val_losses = metrics["val_loss"]
        train_metrics = metrics["train_metrics"]
        val_metrics = metrics["val_metrics"]
        test_loss = metrics["test_loss"]
        test_metrics = metrics["test_metrics"]

    # Plot results
    plot_save_dir = config.get("plot_save_dir", "plots")
    print(f"Saving plots to {plot_save_dir}")

    plot_save_path = plot_save_dir + "/MLP_MNIST_Learning_Curve.png"
    pathlib.Path(plot_save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(
        train_losses, val_losses, title="MLP Learning Curve", save_path=plot_save_path
    )

    plot_save_path = plot_save_dir + "/MLP_MNIST_Metrics.png"
    plot_metrics(
        train_metrics, val_metrics, title="MLP Metrics", save_path=plot_save_path
    )


if __name__ == "__main__":
    main("03_deep_learning/key_architectures/config.json")
