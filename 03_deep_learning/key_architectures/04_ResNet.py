from datetime import datetime
import json
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

from utils import (
    build_transforms,
    load_checkpoint,
    plot_images,
    plot_small_images,
    train,
    evaluate,
    plot_learning_curve,
    plot_metrics,
    save_model,
    load_model,
    get_mean_std,
)
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            out_channels
        )  # Batch normalization - to stabilize and accelerate training
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        # ReLU activation function - to introduce non-linearity
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, dataset="CIFAR-10", init_weights=True):
        super().__init__()
        
        if dataset == "CIFAR-10":
            self.in_channels = 16
            self.out_channels = [16, 32, 64, 128]
            self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        elif dataset == "ImageNet":
            self.in_channels = 64
            self.out_channels = [64, 128, 256, 512]
            self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        if dataset == "ImageNet":
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer definitions
        self.layer1 = self._make_layer(block, self.out_channels[0], layers[0])
        self.layer2 = self._make_layer(block, self.out_channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.out_channels[2], layers[2], stride=2)
        # Only create layer4 if it's specified in the layers list
        if len(layers) > 3:
            self.layer4 = self._make_layer(block, self.out_channels[3], layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.out_channels[3] * block.expansion, num_classes)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.out_channels[2] * block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if hasattr(self, 'maxpool'):
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if hasattr(self, 'layer4'):
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def predict(self, x):
        probabilities = torch.softmax(self.forward(x), dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class

#------- ImageNet ResNet Architectures -------#

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# ------- CIDAR-10 ResNet Architectures -------#

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)



def preprocess_CIFAR10_data_for_ResNet(
    batch_size, val_size=0.15, test_size=0.2, random_seed=42, augmentation=None,
    show_sample=False):

    # Load the CIFAR-10 dataset without any transforms to compute mean and std
    full_train_dataset_raw = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    NUM_CLASSES = len(full_train_dataset_raw.classes)

    n_train = len(full_train_dataset_raw)
    indices = list(range(n_train))
    split = int(np.floor(val_size * n_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    print(f"Number of training samples: {len(train_idx)}")
    print(f"Number of validation samples: {len(valid_idx)}")
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = DataLoader(full_train_dataset_raw, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    mean, std_dev = get_mean_std(
        train_loader
    )  # Get mean and std dev for CIFAR-10 dataset - scalable to large datasets

    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std Dev: {std_dev}")

    # Define transformations for CIFAR-10 dataset with augmentation
    train_transform = build_transforms(
        normalization=(mean, std_dev), augmentation=augmentation
    )
    val_test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )
    
    # Apply the transformations to the raw training and validation sets
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=val_test_transform)

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    print(f"Number of training samples: {len(train_loader.sampler)}")
    print(f"Number of validation samples: {len(val_loader.sampler)}")
    
    # Load and transform the test dataset
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size of 1 for testing

    class_names = train_dataset.classes
    if show_sample:
        plot_small_images(full_train_dataset_raw, class_names=full_train_dataset_raw.classes)
        sample_loader = DataLoader(
            full_train_dataset_raw, batch_size=9, shuffle=True,
            num_workers=0
        )
        for images, labels in sample_loader:
            print(images.shape, labels.shape)
            print(images.numpy().shape)
            X = images.numpy().transpose(0,2,3,1)
            plot_images(X, class_names, labels)

    return train_loader, val_loader, test_loader, NUM_CLASSES


#--------------------------------------------------------------------------------------

def main(config_path):
    with open(config_path) as f:
        config = json.load(f)["ResNet"]

    # Important Hyperparameters
    SEED = config.get("hyperparameters", {}).get("seed", 42)
    BATCH_SIZE = config.get("hyperparameters", {}).get("batch_size", 32)
    LEARNING_RATE = config.get("hyperparameters", {}).get("learning_rate", 0.005)
    DEVICE = config.get("hyperparameters", {}).get("device", "cuda")
    EPOCHS = config.get("hyperparameters", {}).get("epochs", 100)
    MODEL = config.get("hyperparameters", {}).get("model", "ResNet-110")
    LOSS_FUNCTION = config.get("hyperparameters", {}).get(
        "loss_function", "CrossEntropyLoss"
    )
    OPTIMIZER = config.get("hyperparameters", {}).get("optimizer", "Adam")
    METRICS = config.get("hyperparameters", {}).get("metrics", ["Accuracy", "AUROC"])

    DATASET = config.get("data", {}).get("dataset", "CIFAR-100")
    DATA_DIR = config.get("data", {}).get("data_dir", "./data")
    VAL_SIZE = config.get("data", {}).get("val_size", 0.15)
    TEST_SIZE = config.get("data", {}).get("test_size", 0.15)
    AUGMENTATION = config.get("data", {}).get("augmentation", False)

    MODEL_SAVE_DIR = config.get(
        "model_save_path", "03_deep_learning/key_architectures/model_artifacts/resnet"
    )
    pathlib.Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    METRICS_SAVE_DIR = config.get(
        "metrics_save_dir", "03_deep_learning/key_architectures/metrics/resnet"
    )
    pathlib.Path(METRICS_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    PLOT_SAVE_DIR = config.get(
        "plot_save_dir", "03_deep_learning/key_architectures/plots/resnet"
    )
    pathlib.Path(PLOT_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Training from a checkpoint
    TRAIN_FROM_CHECKPOINT = config.get("train_from_checkpoint", False)
    MODEL_CHECKPOINT_DIR = config.get(
        "model_checkpoint_dir",
        "03_deep_learning/key_architectures/model_artifacts/resnet/checkpoints",
    )
    pathlib.Path(MODEL_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Skip training
    SKIP_TRAINING = config.get("skip_training", False)
    LOAD_SAVED_METRICS = config.get("load_saved_metrics", False)
    SAVED_METRICS_FILEPATH = config.get(
        "saved_metrics_filepath",
        "03_deep_learning/key_architectures/metrics/resnet/CIFAR-100_ResNet_training_metrics.json",
    )

    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Load and preprocess the CIFAR-10 dataset
    train_loader, val_loader, test_loader, NUM_CLASSES = preprocess_CIFAR10_data_for_ResNet(
        BATCH_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, random_seed=SEED, augmentation=AUGMENTATION,
        show_sample=False
    )
    logging.info("Loaded CIFAR-10 dataset")

    # Get input size dynamically from input in the format -> [B,C,H,W]
    INPUT_SIZE = (BATCH_SIZE, train_loader.dataset[0][0].shape[0], 
                  train_loader.dataset[0][0].shape[1], train_loader.dataset[0][0].shape[2])

    print("Number of training samples:", len(train_loader.sampler))
    print("Number of validation samples:", len(val_loader.sampler))
    print("Number of test samples:", len(test_loader.dataset))
    print("Number of classes:", NUM_CLASSES)
    print("Input size:", INPUT_SIZE)
    print("Input shape:", train_loader.dataset[0][0].shape)
    print("Output:", train_loader.dataset[0][1])

    #--------------------------------------------------------------------------------------
    # Initialize the model
    if MODEL == "ResNet-110":
        model = resnet110().to(DEVICE)
        logging.info("Initialized ResNet-18 model")
    else:
        raise ValueError(f"Unknown ResNet architecture: {MODEL}")

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
    # save torchinfo summary
    # model_summary_filepath = MODEL_SAVE_DIR + "/{}_{}_summary.txt".format(DATASET, MODEL)
    # with open(model_summary_filepath, "w") as f:
    #     f.write(str(summary(model, input_size=INPUT_SIZE)))
    
    # Initialize Metrics
    metrics = []
    if "Accuracy" in METRICS:
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        metrics.append(accuracy)
    if "F1Score" in METRICS:
        f1_score = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
        metrics.append(f1_score)
    if "AUROC" in METRICS:
        auroc = torchmetrics.AUROC(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE)
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
            # Get the latest checkpoint file from the model checkpoint directory
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
            with open("ResNet_checkpoint_training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info("Metrics saved to file")

        # ------------ TESTING --------------#

        model_save_path = MODEL_SAVE_DIR + "/{}_{}.pth".format(DATASET, MODEL)
        # Load the trained model
        model = resnet110(device=DEVICE).to(DEVICE)
        model = load_model(model, model_save_path, device=DEVICE)

        # Predict on test data
        test_loader = preprocess_CIFAR10_data_for_ResNet(
            BATCH_SIZE, val_size=VAL_SIZE
        )[2]

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
