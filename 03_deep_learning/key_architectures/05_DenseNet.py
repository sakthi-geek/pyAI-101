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

class DenseLayer(nn.Module):
    def __init__(self, input_features, growth_rate, bottleneck_width):
        super().__init__()
        inter_channels = bottleneck_width * growth_rate
        
        # Bottleneck layers
        self.bn1 = nn.BatchNorm2d(input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_features, inter_channels, kernel_size=1, stride=1, bias=False)
        
        # Composite function (3x3 convolution)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        concatenated_features = torch.cat(x, 1)
        bottleneck_output = self.conv1(self.relu1(self.bn1(concatenated_features))) 
        new_features = self.conv2(self.relu2(self.bn2(bottleneck_output)))
        return new_features
    
class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_features, growth_rate, bottleneck_width):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(input_features + i * growth_rate, growth_rate, bottleneck_width)
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):              # Implementing DenseNet-BC (Bottleneck and Compression)
    def __init__(self, num_blocks, num_layers_per_block, growth_rate, reduction, num_classes, bottleneck_width=4, device="cuda"):
        super().__init__()
        num_features = 2 * growth_rate  # Initial number of features is twice the growth rate
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=7, stride=2, padding=3, bias=False)  # Initial convolution
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense Blocks and Transition Layers
        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        
        # Each dense block
        for i in range(num_blocks):
            block = DenseBlock(num_layers_per_block[i], num_features, growth_rate, bottleneck_width)
            self.dense_blocks.append(block)
            num_features += num_layers_per_block[i] * growth_rate
            
            if i != num_blocks - 1:  # No transition layer after the last block
                out_features = int(num_features * reduction)    # Apply compression to reduce the number of features
                transition = TransitionLayer(num_features, out_features)
                self.trans_layers.append(transition)
                num_features = out_features

        # Final batch norm  
        self.bn2 = nn.BatchNorm2d(num_features)

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        #Linear layer
        self.fc = nn.Linear(num_features, num_classes)

        self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        for dense_block, transition_layer in zip(self.dense_blocks, self.trans_layers):
            x = dense_block(x)
            x = transition_layer(x)
        # Last block without transition layer
        x = self.dense_blocks[-1](x)    # Last dense block
        x = self.bn2(x)                 # Final batch norm
        x = self.avg_pool(x)            # Global average pooling
        x = torch.flatten(x, 1)         # Flatten
        x = self.fc(x)                  # Linear layer
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

def densenet121(num_classes):
    return DenseNet(num_blocks=4, num_layers_per_block=[6, 12, 24, 16], growth_rate=32, reduction=0.5, num_classes=num_classes, bottleneck_width=4)

def densenet161(num_classes):
    return DenseNet(num_blocks=4, num_layers_per_block=[6, 12, 36, 24], growth_rate=48, reduction=0.5, num_classes=num_classes, bottleneck_width=4)

def densenet169(num_classes):
    return DenseNet(num_blocks=4, num_layers_per_block=[6, 12, 32, 32], growth_rate=32, reduction=0.5, num_classes=num_classes, bottleneck_width=4)

def densenet201(num_classes):
    return DenseNet(num_blocks=4, num_layers_per_block=[6, 12, 48, 32], growth_rate=32, reduction=0.5, num_classes=num_classes, bottleneck_width=4)

def densenet264(num_classes):
    return DenseNet(num_blocks=4, num_layers_per_block=[6, 12, 64, 48], growth_rate=32, reduction=0.5, num_classes=num_classes, bottleneck_width=4)


def preprocess_CIFAR10_data_for_DenseNet(
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
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = DataLoader(full_train_dataset_raw, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    mean, std_dev = get_mean_std(
        train_loader
    )  # Get mean and std dev for CIFAR-10 dataset 

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
        config = json.load(f)["DenseNet"]

    # Important Hyperparameters
    SEED = config.get("hyperparameters", {}).get("seed", 42)
    BATCH_SIZE = config.get("hyperparameters", {}).get("batch_size", None)
    LEARNING_RATE = config.get("hyperparameters", {}).get("learning_rate", None)
    EPOCHS = config.get("hyperparameters", {}).get("epochs", None)
    MODEL = config.get("hyperparameters", {}).get("model", None)
    LOSS_FUNCTION = config.get("hyperparameters", {}).get("loss_function", None)
    OPTIMIZER = config.get("hyperparameters", {}).get("optimizer", None)
    SCHEDULER = config.get("hyperparameters", {}).get("scheduler", None)
    METRICS = config.get("hyperparameters", {}).get("metrics", None)

    DATASET = config.get("data", {}).get("dataset", None)
    DATA_DIR = config.get("data", {}).get("data_dir", None)
    VAL_SIZE = config.get("data", {}).get("val_size", None)
    TEST_SIZE = config.get("data", {}).get("test_size", None)
    AUGMENTATION = config.get("data", {}).get("augmentation", False)

    DEVICE = config.get("device", None)
    MODEL_SAVE_DIR = config.get("model_save_dir", None)
    pathlib.Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    METRICS_SAVE_DIR = config.get("metrics_save_dir", None)
    pathlib.Path(METRICS_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    PLOT_SAVE_DIR = config.get("plot_save_dir", None)
    pathlib.Path(PLOT_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Training from a checkpoint
    TRAIN_FROM_CHECKPOINT = config.get("train_from_checkpoint", False)
    MODEL_CHECKPOINT_DIR = config.get("model_checkpoint_dir", None)
    pathlib.Path(MODEL_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Skip training
    SKIP_TRAINING = config.get("skip_training", False)
    LOAD_SAVED_METRICS = config.get("load_saved_metrics", False)
    SAVED_METRICS_FILEPATH = config.get("saved_metrics_filepath", None)

    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Load and preprocess the CIFAR-10 dataset
    train_loader, val_loader, test_loader, NUM_CLASSES = preprocess_CIFAR10_data_for_DenseNet(
        BATCH_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, random_seed=SEED, augmentation=AUGMENTATION,
        show_sample=False
    )
    logging.info("Loaded {} dataset")

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
    if MODEL == "DenseNet-169":
        model = densenet169(NUM_CLASSES).to(DEVICE)
        logging.info("Initialized {} model".format(MODEL))
    else:
        raise ValueError(f"Unknown architecture: {MODEL}")

    # Loss function
    if LOSS_FUNCTION == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        logging.info("Using {}".format(LOSS_FUNCTION)) 
    else:
        raise ValueError(f"Unknown loss function: {LOSS_FUNCTION}")

    # Optimizer
    if OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        logging.info("Using {} optimizer".format(OPTIMIZER))
    elif OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        logging.info("Using {} optimizer with momentum 0.9".format(OPTIMIZER))
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")
    
    # Scheduler
    if SCHEDULER == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        logging.info("Using {} scheduler".format(SCHEDULER))
    elif SCHEDULER == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
        logging.info("Using {} scheduler".format(SCHEDULER))
    else:
        scheduler = None
        logging.warning(f"Unknown scheduler: {SCHEDULER}. Training will continue without a scheduler.")
        
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
            scheduler=scheduler,
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
            with open(train_metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info("Metrics saved to file")

        # ------------ TESTING --------------#

        model_save_path = MODEL_SAVE_DIR + "/{}_{}.pth".format(DATASET, MODEL)
        # Load the trained model
        model = densenet169(NUM_CLASSES).to(DEVICE)
        model = load_model(model, model_save_path, device=DEVICE)

        # Predict on test data
        test_loader = preprocess_CIFAR10_data_for_DenseNet(
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
        train_losses, val_losses, title="{}_{}_Learning_Curve".format(DATASET, MODEL), save_path=plot_save_path
    )

    plot_save_path = PLOT_SAVE_DIR + "/{}_{}.png".format(DATASET, MODEL)
    plot_metrics(
        train_metrics, val_metrics, title="{}_{}".format(DATASET, MODEL), save_path=plot_save_path
    )


if __name__ == "__main__":
    main("03_deep_learning/key_architectures/config.json")
