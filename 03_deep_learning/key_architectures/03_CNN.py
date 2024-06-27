from datetime import datetime
import json
from math import prod
import pathlib
from matplotlib import pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils import (
    build_transforms,
    get_mean_std,
    load_checkpoint,
    plot_images,
    train,
    evaluate,
    plot_learning_curve,
    plot_metrics,
    save_model,
    load_model,
)
import logging
from base_model import BaseModel

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LeNet(BaseModel):  # Implementing LeNet-5 architecture

    def __init__(self, num_classes, input_channels=3, device=None):
        super().__init__()
        # Define the layers of the network - OUTPUT: (W-F+2P)/S + 1 where W is the input size, F is the filter size, P is the padding, and S is the stride

        # Convolutional layer with 6 filters of size 5x5 and stride of 1 - no padding - OUTPUT: (32-5)/1 + 1 = 28 (for CIFAR-10, input size is 32x32x3)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
        )
        # Max pooling with a 2x2 window and stride of 2 - OUTPUT: (28-2)/2 + 1 = 14
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Original LeNet-5 uses average pooling - max pooling is more common now
        # Second convolutional layer with 16 filters of size 5x5 and stride of 1 - no padding - OUTPUT: (14-5)/1 + 1 = 10
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        # Second max pooling layer with a 2x2 window and stride of 2 - OUTPUT: (10-2)/2 + 1 = 5
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Original LeNet-5 uses average pooling - max pooling is more common now
        # First fully connected layer, flattening included - INPUT feature size: 16 channels * 5x5 spatial dimensions = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  #
        # Second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # Output layer with a number of outputs matching the number of classes
        self.fc3 = nn.Linear(84, num_classes)
        self.to(device)

    def forward(self, x):

        # Convolutional layers
        x = self.conv1(x)
        x = torch.relu(
            x
        )  # Original LeNet-5 uses sigmoid activation - ReLU is more common now
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(
            x
        )  # Original LeNet-5 uses sigmoid activation - ReLU is more common now
        x = self.pool2(x)

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(
            x
        )  # Original LeNet-5 uses sigmoid activation - ReLU is more common now
        x = self.fc2(x)
        x = torch.relu(
            x
        )  # Original LeNet-5 uses sigmoid activation - ReLU is more common now
        x = self.fc3(
            x
        )  # No activation before CrossEntropyLoss as it expects raw logits

        return x

    def predict(self, x):
        probabilities = torch.softmax(self.forward(x), dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class



def preprocess_MNIST_data_for_LeNet(
    batch_size, val_size=0.1, test_size=0.1, augmentation=None
):

    # Load the MNIST dataset without any transforms to compute mean and std
    full_train_dataset_raw = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(full_train_dataset_raw, batch_size=batch_size, shuffle=False)
    mean, std_dev = get_mean_std(
        loader
    )  # Get mean and std dev for MNIST dataset - scalable to large datasets

    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std Dev: {std_dev}")

    plot_images(full_train_dataset_raw, class_names=full_train_dataset_raw.classes)

    # Define transformations for MNIST dataset for LeNet - LeNet expects 32x32 images but MNIST images are 28x28
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (32, 32)
            ),  # Resize MNIST images from 28x28 to 32x32 to match LeNet input
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(
                mean, std_dev
            ),  # Normalize using the mean and std of MNIST - [0.1307], [0.3081]
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (32, 32)
            ),  # Resize MNIST images from 28x28 to 32x32 to match LeNet input
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(
                mean, std_dev
            ),  # Normalize using the mean and std of MNIST - [0.1307], [0.3081]
        ]
    )

    # Load the MNIST dataset
    full_train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=train_transform
    )  # Applied the transform before splitting - considered safe as they don't depend on the data split and are identical for every image
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=test_transform
    )
    logging.info("Loaded MNIST dataset with LeNet transforms")

    plot_images(
        full_train_dataset,
        class_names=full_train_dataset.classes,
        denormalize=True,
        mean=mean,
        std_dev=std_dev,
    )

    # Split the dataset into training and validation sets
    val_size = int(val_size * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Create DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def preprocess_CIFAR10_data_for_LeNet(
    batch_size, val_size=0.15, test_size=0.2, augmentation=None
):

    # Load the CIFAR-10 dataset without any transforms to compute mean and std
    full_train_dataset_raw = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(full_train_dataset_raw, batch_size=batch_size, shuffle=False)
    mean, std_dev = get_mean_std(
        loader
    )  # Get mean and std dev for CIFAR-10 dataset - scalable to large datasets

    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std Dev: {std_dev}")

    plot_images(full_train_dataset_raw, class_names=full_train_dataset_raw.classes)

    # Define transformations for CIFAR-10 dataset with augmentation
    train_transform = build_transforms(
        normalization=(mean, std_dev), augmentation=augmentation
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean, std_dev
            ),  # Mean and std dev for CIFAR-10 dataset -[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        ]
    )

    # Load the CIFAR-10 dataset
    full_train_dataset = datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        "./data", train=False, download=True, transform=test_transform
    )
    logging.info("Loaded CIFAR-10 dataset with LeNet transforms and augmentation")

    # Split the dataset into training and validation sets
    val_size = int(val_size * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Create DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )  # Batch size of 1 for testing

    return train_loader, val_loader, test_loader



def main(config_path):

    with open(config_path) as f:
        config = json.load(f)["CNN"]

    # Important Hyperparameters
    SEED = config.get("hyperparameters", {}).get("seed", 42)
    BATCH_SIZE = config.get("hyperparameters", {}).get("batch_size", 32)
    LEARNING_RATE = config.get("hyperparameters", {}).get("learning_rate", 0.001)
    DEVICE = config.get("hyperparameters", {}).get("device", "cuda")
    EPOCHS = config.get("hyperparameters", {}).get("epochs", 100)
    NUM_CLASSES = config.get("hyperparameters", {}).get("num_classes", 10)
    MODEL = config.get("hyperparameters", {}).get("model", "LeNet")
    LOSS_FUNCTION = config.get("hyperparameters", {}).get(
        "loss_function", "CrossEntropyLoss"
    )
    OPTIMIZER = config.get("hyperparameters", {}).get("optimizer", "Adam")
    METRICS = config.get("hyperparameters", {}).get("metrics", ["Accuracy", "F1Score"])

    DATASET = config.get("data", {}).get("dataset", "CIFAR-10")
    DATA_DIR = config.get("data", {}).get("data_dir", "./data")
    VAL_SIZE = config.get("data", {}).get("val_size", 0.15)
    TEST_SIZE = config.get("data", {}).get("test_size", 0.15)
    AUGMENTATION = config.get("data", {}).get("augmentation", False)

    MODEL_SAVE_DIR = config.get(
        "model_save_path", "03_deep_learning/key_architectures/model_artifacts/cnn"
    )
    pathlib.Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    METRICS_SAVE_DIR = config.get(
        "metrics_save_dir", "03_deep_learning/key_architectures/metrics/cnn"
    )
    pathlib.Path(METRICS_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    PLOT_SAVE_DIR = config.get(
        "plot_save_dir", "03_deep_learning/key_architectures/plots/cnn"
    )
    pathlib.Path(PLOT_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Training from a checkpoint
    TRAIN_FROM_CHECKPOINT = config.get("train_from_checkpoint", False)
    MODEL_CHECKPOINT_DIR = config.get(
        "model_checkpoint_dir",
        "03_deep_learning/key_architectures/model_artifacts/cnn/checkpoints",
    )
    pathlib.Path(MODEL_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Skip training
    SKIP_TRAINING = config.get("skip_training", False)
    LOAD_SAVED_METRICS = config.get("load_saved_metrics", False)
    SAVED_METRICS_FILEPATH = config.get(
        "saved_metrics_filepath",
        "03_deep_learning/key_architectures/metrics/cnn/CIFAR-10_LeNet_training_metrics.json",
    )

    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    numpy.random.seed(SEED)

    if DATASET == "MNIST":
        input_channels = 1
        # Load and preprocess the MNIST dataset
        train_loader, val_loader, test_loader = preprocess_MNIST_data_for_LeNet(
            BATCH_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            augmentation=AUGMENTATION,
        )
        logging.info("Loaded MNIST dataset")

    elif DATASET == "CIFAR-10":
        input_channels = 3
        # Load and preprocess the CIFAR-10 dataset
        train_loader, val_loader, test_loader = preprocess_CIFAR10_data_for_LeNet(
            BATCH_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            augmentation=AUGMENTATION,
        )
        logging.info("Loaded CIFAR-10 dataset")

    print("Number of training samples:", len(train_loader.dataset))
    print("Number of validation samples:", len(val_loader.dataset))
    print("Number of test samples:", len(test_loader.dataset))
    print("Input shape:", train_loader.dataset[0][0].shape)
    print("Output:", train_loader.dataset[0][1])

    # Initialize the model
    if MODEL == "LeNet":
        model = LeNet(
            num_classes=NUM_CLASSES, input_channels=input_channels, device=DEVICE
        ).to(DEVICE)
        logging.info("Initialized LeNet-5 model")
    else:
        raise ValueError(f"Unknown CNN architecture: {MODEL}")

    # Loss function
    if LOSS_FUNCTION == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        logging.info("Using CrossEntropyLoss")
    else:
        raise ValueError(f"Unknown loss function: {LOSS_FUNCTION}")

    # Optimizer
    if OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        logging.info("Using Adam optimizer")
    elif OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        logging.info("Using SGD optimizer with momentum 0.9")
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    # Model summary
    logging.info(model)
    logging.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Initialize Metrics
    metrics = []
    if "Accuracy" in METRICS:
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(
            DEVICE
        )
        metrics.append(accuracy)
    if "F1Score" in METRICS:
        f1_score = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES).to(
            DEVICE
        )
        metrics.append(f1_score)
    if "AUROC" in METRICS:
        auroc = torchmetrics.AUROC(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        metrics.append(auroc)
    if "ConfusionMatrix" in METRICS:
        confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_CLASSES).to(
            DEVICE
        )
        metrics.append(confusion_matrix)
    logging.info(f"Metrics: {metrics}")

    if not SKIP_TRAINING and TRAIN_FROM_CHECKPOINT == False:

        checkpoint_path = MODEL_CHECKPOINT_DIR + "/{}_{}/".format(DATASET, MODEL)
        pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        # Training and evaluation
        train_losses, val_losses, train_metrics, val_metrics = train(
            model,
            train_loader,
            val_loader,
            criterion,
            metrics,
            optimizer,
            EPOCHS,
            early_stopping_patience=15,
            checkpoint_path=checkpoint_path,
            device=DEVICE,
        )
        logging.info("Training completed")

        test_loss, test_metrics = evaluate(
            model, test_loader, criterion, metrics, device=DEVICE
        )
        logging.info("Evaluation completed")

        # Track metrics and save them to a file
        metrics = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        }

        metrics_save_filepath = (
            METRICS_SAVE_DIR
            + "/{}_{}_training_metrics_{}.json".format(
                DATASET, MODEL, datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        )
        with open(metrics_save_filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics saved to file")

        model_save_path = MODEL_SAVE_DIR + "/{}_{}.pth".format(DATASET, MODEL)
        # Save the final trained model
        pathlib.Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        save_model(model, model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    else:
        if TRAIN_FROM_CHECKPOINT:
            # get the latest checkpoint file from the model checkpoint directory
            checkpoint_files = pathlib.Path(MODEL_CHECKPOINT_DIR).glob("*.pt")
            checkpoint_files = sorted(checkpoint_files, key=lambda x: x.stat().st_ctime)
            latest_checkpoint = checkpoint_files[-1]

            # Load the latest checkpoint
            load_checkpoint(model, optimizer, file_path=latest_checkpoint)
            logging.info(
                f"Model loaded from latest checkpoint file: {latest_checkpoint}"
            )

            checkpoint_path = MODEL_CHECKPOINT_DIR + "/{}_{}/".format(DATASET, MODEL)
            pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
            # Training and evaluation
            train_losses, val_losses, train_metrics, val_metrics = train(
                model,
                train_loader,
                val_loader,
                criterion,
                metrics,
                optimizer,
                EPOCHS,
                early_stopping_patience=15,
                checkpoint_path=checkpoint_path,
                device=DEVICE,
            )
            logging.info("Training completed")

            test_loss, test_metrics = evaluate(
                model, test_loader, criterion, metrics, device=DEVICE
            )
            logging.info("Evaluation completed")

            # Track metrics and save them to a file
            metrics = {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
            }

            train_metrics_filepath = (
                METRICS_SAVE_DIR + "/{}_{}_training_metrics.json".format(DATASET, MODEL)
            )
            with open("LeNet_checkpoint_training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info("Metrics saved to file")

        # ------------ TESTING --------------#

        model_save_path = MODEL_SAVE_DIR + "/{}_{}.pth".format(DATASET, MODEL)
        # Load the trained model
        model = LeNet(num_classes=NUM_CLASSES, device=DEVICE).to(DEVICE)
        model = load_model(model, model_save_path, device=DEVICE)

        # Predict on test data
        test_loader = preprocess_CIFAR10_data_for_LeNet(BATCH_SIZE, val_size=VAL_SIZE)[
            2
        ]

        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            y_pred = model.predict(data)
            print("target:", target)
            print("predicted:", y_pred)
            # ---- show images and predictions ----#
            plt.imshow(data[0].permute(1, 2, 0).cpu().numpy())
            plt.title(f"Predicted: {y_pred[0]} - Actual: {target[0]}")
            plt.axis("off")
            plt.show()
            break

        if LOAD_SAVED_METRICS:
            # Load saved metrics
            with open(SAVED_METRICS_FILEPATH, "r") as f:
                metrics = json.load(f)

            train_losses = metrics["train_loss"]
            val_losses = metrics["val_loss"]
            train_metrics = metrics["train_metrics"]
            val_metrics = metrics["val_metrics"]
            test_loss = metrics["test_loss"]
            test_metrics = metrics["test_metrics"]

    # Plot results
    print(f"Saving plots to {PLOT_SAVE_DIR}")

    plot_save_path = PLOT_SAVE_DIR + "/{}_{}_Learning_Curve_{}.png".format(
        DATASET, MODEL, datetime.now().strftime("%Y%m%d_%H%M")
    )
    pathlib.Path(plot_save_path).parent.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(
        train_losses, val_losses, title="LeNet Learning Curve", save_path=plot_save_path
    )

    plot_save_path = PLOT_SAVE_DIR + "/{}_{}.png".format(DATASET, MODEL)
    plot_metrics(
        train_metrics, val_metrics, title="{}_{}".format(DATASET, MODEL), save_path=plot_save_path
    )
    

if __name__ == "__main__":
    main("03_deep_learning/key_architectures/config.json")
