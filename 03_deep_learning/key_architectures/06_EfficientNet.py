from datetime import datetime
import json
import math
import pathlib
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# MBConv Block: Mobile Inverted Residual Bottleneck Block
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block with squeeze and excitation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.drop_connect_rate = drop_connect_rate

        mid_channels = int(in_channels * expand_ratio)
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, 1, bias=False) if expand_ratio != 1 else nn.Identity()
        self.bn0 = nn.BatchNorm2d(mid_channels)
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.se = SqueezeExcitation(mid_channels, int(in_channels * se_ratio))
        self.project_conv = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bn0(x)
        x = F.relu6(x, inplace=True)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = F.relu6(x, inplace=True)
        x = self.se(x)
        x = self.project_conv(x)
        x = self.bn2(x)

        if self.use_res_connect:
            if self.training and self.drop_connect_rate > 0:
                x = drop_connect(x, self.drop_connect_rate, self.training)
            x += identity
        return x

# Squeeze and Excitation Layer
class SqueezeExcitation(nn.Module):
    """
    Squeeze and Excitation Layer for recalibrating channel-wise feature responses.
    """
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se_reduce = nn.Conv2d(in_channels, reduced_dim, 1)
        self.se_expand = nn.Conv2d(reduced_dim, in_channels, 1)

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, (1, 1))
        se = self.se_reduce(se)
        se = F.relu(se, inplace=True)
        se = self.se_expand(se)
        return x * torch.sigmoid(se)

def drop_connect(inputs, probability, training):
    """
    Drop connect implementation for regularization.
    """
    if not training:
        return inputs
    keep_prob = 1 - probability
    batch_size = inputs.shape[0]
    random_tensor = keep_prob + torch.rand((batch_size, 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

class EfficientNet(nn.Module):
    """
    EfficientNet model which scales depth, width, and resolution based on compound coefficients.
    """
    def __init__(self, model_config, resolution, dropout_rate=0.2, num_classes=1000):
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        width_coefficient = model_config['width_coefficient']
        depth_coefficient = model_config['depth_coefficient']
        resolution = model_config['resolution']

        # Setting up the initial convolution
        self.conv_stem = nn.Conv2d(3, round_filters(32, width_coefficient), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(round_filters(32, width_coefficient))

        # Setting up the MBConv blocks based on the external configuration
        self.blocks = nn.ModuleList([])
        for block_args in model_config['blocks']:
            input_filters = round_filters(block_args['input_filters'], width_coefficient)
            output_filters = round_filters(block_args['output_filters'], width_coefficient)
            repeats = round_repeats(block_args['n_repeats'], depth_coefficient)

            for _ in range(repeats):
                self.blocks.append(MBConvBlock(input_filters,
                                               output_filters,
                                               block_args['kernel_size'],
                                               block_args['stride'],
                                               block_args['expand_ratio'],
                                               block_args['se_ratio'],
                                               block_args['drop_connect_rate']))
                input_filters = output_filters  # Update input filters for subsequent blocks

        # Setting up the head
        self.conv_head = nn.Conv2d(round_filters(320, width_coefficient), round_filters(1280, width_coefficient), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(round_filters(1280, width_coefficient))
        self.fc = nn.Linear(round_filters(1280, width_coefficient), num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn0(x)
        x = F.relu6(x, inplace=True)

        for block in self.blocks:
            x = block(x)

        x = self.conv_head(x)
        x = self.bn1(x)
        x = F.relu6(x, inplace=True)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc(x)
        return x

def round_filters(filters, width_coefficient, divisor=8):
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(repeats * depth_coefficient))


def get_efficientnet(model_name="B0", dataset="CIFAR-10", num_classes=None):
    """Generates EfficientNet models adapted for different datasets and model sizes."""
    
    # EfficientNet B0 to B7 hyperparameters
    baseline_network_blocks = [
            {'input_filters': 32, 'output_filters': 16, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 1, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'n_repeats': 1},
            {'input_filters': 16, 'output_filters': 24, 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'n_repeats': 2},
            {'input_filters': 24, 'output_filters': 40, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'n_repeats': 2},
            {'input_filters': 40, 'output_filters': 80, 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'n_repeats': 3},
            {'input_filters': 80, 'output_filters': 112, 'kernel_size': 5, 'stride': 1, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'n_repeats': 3},
            {'input_filters': 112, 'output_filters': 192, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'n_repeats': 4},
            {'input_filters': 192, 'output_filters': 320, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 6, 'se_ratio': 0.25, 'drop_connect_rate': 0.2, 'n_repeats': 1}
        ]

    # Model configurations for EfficientNet B0 to B7
    model_configurations = {
        'B0': {
            'width_coefficient': 1.0, 'depth_coefficient': 1.0, 'resolution': 224, 'dropout_rate': 0.2, 'blocks': baseline_network_blocks
        },
        'B1': {
            'width_coefficient': 1.0, 'depth_coefficient': 1.1, 'resolution': 240, 'dropout_rate': 0.2, 'blocks': baseline_network_blocks
        },
        'B2': {
            'width_coefficient': 1.1, 'depth_coefficient': 1.2, 'resolution': 260, 'dropout_rate': 0.3, 'blocks': baseline_network_blocks
        },
        'B3': {
            'width_coefficient': 1.2, 'depth_coefficient': 1.4, 'resolution': 300, 'dropout_rate': 0.3, 'blocks': baseline_network_blocks
        },
        'B4': {
            'width_coefficient': 1.4, 'depth_coefficient': 1.8, 'resolution': 380, 'dropout_rate': 0.4, 'blocks': baseline_network_blocks
        },
        'B5': {
            'width_coefficient': 1.6, 'depth_coefficient': 2.2, 'resolution': 456, 'dropout_rate': 0.4, 'blocks': baseline_network_blocks
        },
        'B6': {
            'width_coefficient': 1.8, 'depth_coefficient': 2.6, 'resolution': 528, 'dropout_rate': 0.5, 'blocks': baseline_network_blocks
        },
        'B7': {
            'width_coefficient': 2.0, 'depth_coefficient': 3.1, 'resolution': 600, 'dropout_rate': 0.5, 'blocks': baseline_network_blocks
        }
    }

    # # Compound scaling method to determine the model hyperparameters
    # # Model configurations for EfficientNet B0 to B7
    # phi_values = {'B0': 0, 'B1': 0.5, 'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4, 'B6': 5, 'B7': 6}

    # alpha = 1.2  # Depth coefficient
    # beta = 1.1   # Width coefficient
    # gamma = 1.15 # Resolution coefficient

    # if model_name not in phi_values:
    #     raise ValueError(f"Unsupported model version: {model_name}. Choose from 'B0' to 'B7'.")

    # phi = phi_values[model_name]
    # width_coefficient = beta ** phi
    # depth_coefficient = alpha ** phi
    # resolution = int(224 * gamma ** phi)

    # comp_scaling_model_config = {}
    # for model, phi in phi_values.items():
    #     width_coefficient = round(beta ** phi, 2)
    #     depth_coefficient = round(alpha ** phi, 2)
    #     resolution = int(224 * gamma ** phi)

    #     comp_scaling_model_config[model] = {
    #         'width_coefficient': width_coefficient,
    #         'depth_coefficient': depth_coefficient,
    #         'resolution': resolution,
    #         'dropout_rate': 0.2 + 0.1 * phi,  # Example of scaling dropout
    #         'blocks': baseline_network_blocks
    #     }

    # pprint(comp_scaling_model_config)

    # # Dataset-specific configurations
    # dataset_params = {
    #     'CIFAR-10': (32, 10),
    #     'CIFAR-100': (32, 100),
    #     'ImageNet': (224, 1000),
    #     'Flowers': (224, 102)
    # }
    # if dataset not in dataset_params:
    #     raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets are 'cifar10', 'cifar100', 'imagenet', 'flowers'.")

    # resolution, classes = dataset_params[dataset]
    # num_classes = num_classes if num_classes is not None else classes

    if model_name not in model_configurations:
        raise ValueError(f"Unsupported model version: {model_name}. Choose from 'B0' to 'B7'.")
    
    width_coefficient, depth_coefficient, resolution, dropout_rate, _ = model_configurations[model_name].values() 

    model = EfficientNet(model_configurations["B0"], resolution, dropout_rate, num_classes)
    return model


def efficientnet_b0(num_classes=10):
    return get_efficientnet("B0", num_classes=num_classes)

def efficientnet_b1(num_classes=10):
    return get_efficientnet("B1", num_classes=num_classes)

def efficientnet_b2(num_classes=10):
    return get_efficientnet("B2", num_classes=num_classes)

def efficientnet_b3(num_classes=10):
    return get_efficientnet("B3", num_classes=num_classes)

def efficientnet_b4(num_classes=10):
    return get_efficientnet("B4", num_classes=num_classes)

def efficientnet_b5(num_classes=10):
    return get_efficientnet("B5", num_classes=num_classes)

def efficientnet_b6(num_classes=10):
    return get_efficientnet("B6", num_classes=num_classes)

def efficientnet_b7(num_classes=10):
    return get_efficientnet("B7", num_classes=num_classes)

#--------------------------------------------------------------------------------------

def preprocess_CIFAR10_data_for_EfficientNet(
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
        config = json.load(f)["EfficientNet"]

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
    train_loader, val_loader, test_loader, NUM_CLASSES = preprocess_CIFAR10_data_for_EfficientNet(
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
    if MODEL == "EfficientNet-B0":
        model = efficientnet_b0(NUM_CLASSES).to(DEVICE)
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
        model = efficientnet_b0(NUM_CLASSES).to(DEVICE)
        model = load_model(model, model_save_path, device=DEVICE)

        # Predict on test data
        test_loader = preprocess_CIFAR10_data_for_EfficientNet(
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
