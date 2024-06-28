from datetime import datetime
import os
import matplotlib.pyplot as plt
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import transforms
from torchinfo import summary
from torchviz import make_dot

# ----------------------------- Logging ---------------------------------#
# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------- Data Preparation ---------------------------------#
def get_mean_std(loader):
    # Vectors to hold the sum and square sum of all elements in each channel
    channel_sum, channel_sqr_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        channel_sqr_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channel_sum / num_batches
    std = (channel_sqr_sum / num_batches - mean**2) ** 0.5
    return mean, std

def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into training, validation, and testing sets.

    Args:
        X (np.array): Features.
        y (np.array): Labels.
        val_size (float): Fraction of data to include in the validation set.
        test_size (float): Fraction of data to include in the testing set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Adjust validation size to compensate for initial split
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size_adjusted, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_transforms(normalization=None, augmentation=None):
    """Builds torchvision transforms with optional augmentation and normalization."""

    transform_list = []

    if augmentation:
        aug_transforms = []
        for aug in augmentation:
            # ----- Augmentations to be applied on PIL images -----#
            if (
                aug["type"] == "RandomHorizontalFlip"
            ):  # p=0.5 means 50% probability of horizontal flipping
                aug_transforms.append(
                    transforms.RandomHorizontalFlip(
                        p=aug.get("params", {}).get("p", 0.5)
                    )
                )
                logging.info(
                    "Applied RandomHorizontalFlip with p={}".format(
                        aug.get("params", {}).get("p", 0.5)
                    )
                )

            elif (
                aug["type"] == "RandomVerticalFlip"
            ):  # p=0.5 means 50% probability of vertical flipping
                aug_transforms.append(
                    transforms.RandomVerticalFlip(p=aug.get("params", {}).get("p", 0.5))
                )
                logging.info(
                    "Applied RandomVerticalFlip with p={}".format(
                        aug.get("params", {}).get("p", 0.5)
                    )
                )

            elif (
                aug["type"] == "RandomRotation"
            ):  # degrees=10 means random rotation between -10 and 10 degrees
                aug_transforms.append(
                    transforms.RandomRotation(
                        degrees=aug.get("params", {}).get("degrees", 10)
                    )
                )
                logging.info(
                    "Applied RandomRotation with degrees={}".format(
                        aug.get("params", {}).get("degrees", 10)
                    )
                )

            elif (
                aug["type"] == "RandomResizedCrop"
            ):  # scale=(0.8, 1.0) means random crop between 80% and 100% of the original size - size=32 means output size is 32x32
                aug_transforms.append(
                    transforms.RandomResizedCrop(
                        size=32, scale=aug.get("params", {}).get("scale", (0.8, 1.0))
                    )
                )
                logging.info(
                    "Applied RandomResizedCrop with scale={}".format(
                        aug.get("params", {}).get("scale", (0.8, 1.0))
                    )
                )

            elif aug["type"] == "CenterCrop":
                aug_transforms.append(
                    transforms.CenterCrop(size=aug.get("params", {}).get("size", 32))
                )
                logging.info(
                    "Applied CenterCrop with size={}".format(
                        aug.get("params", {}).get("size", 32)
                    )
                )

            elif aug["type"] == "Resize":
                aug_transforms.append(
                    transforms.Resize(size=aug.get("params", {}).get("size", (32, 32)))
                )
                logging.info(
                    "Applied Resize with size={}".format(
                        aug.get("params", {}).get("size", (32, 32))
                    )
                )

            elif aug["type"] == "RandomPerspective":
                aug_transforms.append(
                    transforms.RandomPerspective(
                        distortion_scale=0.5, p=0.5, **aug.get("params", {})
                    )
                )
                logging.info(
                    "Applied RandomPerspective with distortion_scale=0.5, p=0.5"
                )

            elif aug["type"] == "ColorJitter":
                aug_transforms.append(transforms.ColorJitter(**aug.get("params", {})))
                logging.info(
                    "Applied ColorJitter with params={}".format(aug.get("params", {}))
                )

            elif aug["type"] == "RandomGrayscale":
                aug_transforms.append(
                    transforms.RandomGrayscale(p=aug.get("params", {}).get("p", 0.1))
                )
                logging.info(
                    "Applied RandomGrayscale with p={}".format(
                        aug.get("params", {}).get("p", 0.1)
                    )
                )

            elif aug["type"] == "RandomAffine":
                aug_transforms.append(
                    transforms.RandomAffine(degrees=0, **aug.get("params", {}))
                )
                logging.info(
                    "Applied RandomAffine with degrees=0, params={}".format(
                        aug.get("params", {})
                    )
                )

            elif aug["type"] == "GaussianBlur":
                aug_transforms.append(
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
                )
                logging.info(
                    "Applied GaussianBlur with kernel_size=(5, 9), sigma=(0.1, 5)"
                )

            elif aug["type"] == "Pad":
                aug_transforms.append(
                    transforms.Pad(
                        padding=aug.get("params", {}).get("padding", 4),
                        fill=aug.get("params", {}).get("fill", 0),
                        padding_mode=aug.get("params", {}).get(
                            "padding_mode", "constant"
                        ),
                    )
                )
                logging.info(
                    "Applied Pad with padding=4, fill=0, padding_mode='constant'"
                )

        # Apply augmentation first
        transform_list.extend(aug_transforms)
    
    transform_list.append(
            transforms.ToTensor()
        )  # Ensure ToTensor() is applied before Normalize()
    logging.info("Applied ToTensor()")

    if normalization:
        mean, std_dev = normalization
        transform_list.append(
            transforms.Normalize(mean, std_dev)
        )  # Normalize using the mean and std of the dataset
        logging.info(
            "Applied Normalize with mean={} and std_dev={}".format(mean, std_dev)
        )

    return transforms.Compose(transform_list)

def plot_small_images(dataset, class_names, fig_size=(16,4), rows=2, cols=10):
        
    fig = plt.figure(figsize=fig_size)

    for i in range(1, (rows * cols) + 1):
        img, label = dataset[i]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.permute(1, 2, 0))  # Permute the dimensions to [H, W, C]
        plt.title(class_names[label])
        plt.axis("off")
        plt.tight_layout()

    plt.show()

def plot_images(images, label_names, cls_true, cls_pred=None):
    """
    Plot images with true and predicted classes.
    """
    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            # plot img
            ax.imshow(images[i, :, :, :], interpolation='spline16')
        

            # show true & predicted classes
            cls_true_name = label_names[cls_true[i]]
            if cls_pred is None:
                xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
            else:
                cls_pred_name = label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(
                    cls_true_name, cls_pred_name
                )
            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
# ----------------------------- Model Helpers ---------------------------------#
def update_metrics(metrics, outputs, labels):
    for metric in metrics:
        metric.update(outputs, labels)


def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()


def save_checkpoint(model, optimizer, file_path=None):
    """Saves model and optimizer states to a file."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)


def load_checkpoint(model, optimizer, file_path=None):
    """Loads model and optimizer states from a file."""
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logging.info(f"Checkpoint loaded from {file_path}")
    return model, optimizer


def train_epoch(model, train_loader, criterion, optimizer, metrics, device):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        update_metrics(metrics, outputs, labels)

    avg_loss = total_loss / len(train_loader)

    metric_results = {}
    for metric in metrics:
        result = metric.compute()
        if isinstance(result, torch.Tensor) and result.numel() > 1:
            metric_results[metric.__class__.__name__] = result.float().mean().item()  # Mean over classes if multiple
        else:
            metric_results[metric.__class__.__name__] = result.item()
    reset_metrics(metrics)

    return avg_loss, metric_results


def validate_epoch(model, val_loader, criterion, metrics, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            update_metrics(metrics, outputs, labels)

    avg_loss = total_loss / len(val_loader)

    metric_results = {}
    for metric in metrics:
        result = metric.compute()
        if isinstance(result, torch.Tensor) and result.numel() > 1:
            metric_results[metric.__class__.__name__] = result.float().mean().item()  # Mean over classes if multiple
        else:
            metric_results[metric.__class__.__name__] = result.item()
    reset_metrics(metrics)

    return avg_loss, metric_results


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    metrics,
    optimizer,
    epochs=100,
    scheduler=None,
    early_stopping_patience=None,
    checkpoint_path=None,
    device=None,
):
    """
    Train the model with the specified parameters and optionally validate on a validation set.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader, optional): DataLoader for validation data. Default is None.
        criterion (nn.Module): Loss function.
        metrics (list of torchmetrics.Metric, optional): List of metric objects to evaluate during training.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of epochs to train for.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Default is None.
        early_stopping_patience (int, optional): Number of epochs with no improvement to stop training. Default is None.
        checkpoint_path (str, optional): Path to save the model checkpoint. Default is None.
        device (torch.device, optional): Device to run training on. Defaults to model's device.

    Returns:
        tuple: (train_losses, val_losses, train_metrics, val_metrics) if val_loader is not None, else (train_losses, train_metrics).
    """
    model.to(device)
    train_losses = []
    val_losses = []
    train_metrics = {metric.__class__.__name__: [] for metric in metrics}
    val_metrics = {metric.__class__.__name__: [] for metric in metrics}
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        avg_train_loss, train_metric_results = train_epoch(
            model, train_loader, criterion, optimizer, metrics, device
        )
        train_losses.append(avg_train_loss)
        for metric_name, result in train_metric_results.items():
            train_metrics[metric_name].append(result)

        if epoch % 5 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")
            for metric_name, result in train_metric_results.items():
                logging.info(
                    f"Epoch {epoch+1}/{epochs}, Training {metric_name}: {result}"
                )

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_train_loss)
            else:
                scheduler.step()

        if val_loader:
            avg_val_loss, val_metric_results = validate_epoch(
                model, val_loader, criterion, metrics, device
            )
            val_losses.append(avg_val_loss)
            for metric_name, result in val_metric_results.items():
                val_metrics[metric_name].append(result)

            if epoch % 5 == 0:
                logging.info(
                    f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}"
                )
                for metric_name, result in val_metric_results.items():
                    logging.info(
                        f"Epoch {epoch+1}/{epochs}, Validation {metric_name}: {result}"
                    )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                epochs_no_improve = 0
                if checkpoint_path and epoch % 1 == 0:
                    final_checkpoint_path = (
                        checkpoint_path
                        + "checkpoint_{}_{}.pt".format(
                            epoch + 1, round(avg_val_loss, 4)
                        )
                    )
                    save_checkpoint(model, optimizer, file_path=final_checkpoint_path)
                    logging.info(f"Checkpoint saved at {final_checkpoint_path}")
            else:
                epochs_no_improve += 1
                if (
                    early_stopping_patience
                    and epochs_no_improve >= early_stopping_patience
                ):
                    logging.info("Early stopping triggered.")
                    break

    return train_losses, val_losses, train_metrics, val_metrics


def evaluate(model, test_loader, criterion, metrics, device=None):
    """
    Evaluate the model's performance on a test set using a given metric.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for testing data.
        criterion (nn.Module): Loss function.
        metric (list of torchmetrics.Metric, optional): List of metric objects to evaluate performance.
        device (torch.device, optional): Device to run evaluation on. Defaults to model's device.

    Returns:
        tuple: (test loss, metric result).
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    test_metrics = {metric.__class__.__name__: [] for metric in metrics}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            update_metrics(metrics, outputs, labels)

    avg_test_loss = total_loss / len(test_loader)
    for metric in metrics:
        result = metric.compute()
        if isinstance(result, torch.Tensor) and result.numel() > 1:
            metric_result = result.float().mean().item()  # Mean over classes if multiple
        else:
            metric_result = result.item()
        test_metrics[metric.__class__.__name__] = metric_result
        metric.reset()
        logging.info(f"Test {metric.__class__.__name__}: {metric_result}")
    logging.info(f"Test Loss: {avg_test_loss}")

    return avg_test_loss, test_metrics


def save_model(model, path):
    """
    Save the model parameters to a file.

    Args:
        model (nn.Module): The model to save.
        path (str): Path to the file where model parameters will be saved.
    """
    torch.save(model.state_dict(), path)


def load_model(model, model_path, device=None):
    """
    Load the model parameters from a file.

    Args:
        model_path (str): Path to the file where model parameters are saved.
        device (torch.device, optional): Device to load the model on. Defaults to CPU.

    Returns:
        nn.Module: The model with loaded parameters.
    """
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    logging.info(
        f"Model loaded from {model_path} and moved to {device} and set to evaluation mode."
    )
    return model


# ----------------------------- Visualizations ---------------------------------#


def plot_learning_curve(
    train_losses, val_losses=None, title="Learning Curve", save_path=None
):
    """
    Plot the learning curve for training and validation losses.

    Args:
        train_losses (list): List of training losses.
        val_losses (list, optional): List of validation losses. Default is None.
        title (str, optional): Title of the plot. Default is "Learning Curve".
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Learning curve saved at {save_path}")
    plt.show()


def plot_metrics(train_metrics, val_metrics, title="Metrics Plot", save_path=None):
    """
    Plot the metrics for training and validation sets.

    Args:
        train_metrics (dict): Dict of training metrics where values of each metric is a list.
        val_metrics (dict): Dict of validation metrics where values of each metric is a list.
        title (str, optional): Title of the plot. Default is "Metrics Plot".
        save_path (str, optional): If provided, the plot will be saved to this path.
    """

    # Ensure the metrics from both train and val are the same and in the same order
    assert (
        train_metrics.keys() == val_metrics.keys()
    ), "Training and validation metrics do not match"
       
    for metric_name in train_metrics.keys():
        plt.figure(figsize=(10, 5))
        plt.plot(train_metrics[metric_name], label=f"Training {metric_name}")  # 'o-'
        plt.plot(val_metrics[metric_name], label=f"Validation {metric_name}")  # 's--'

        plt.title(title + "_" + metric_name)
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)

        # If a save path is provided, save the figure
        if save_path:
            file_name = save_path.split(os.sep)[-1].split(".")[0]
            file_name = file_name + "{}_{}.png".format(metric_name, datetime.now().strftime("%Y%m%d_%H%M"))
            save_path = os.path.join(os.sep.join(save_path.split(os.sep)[:-1]), file_name)
            plt.savefig(save_path)
            logging.info(f"Metrics plot saved at {save_path}")
        plt.show()


def plot_confusion_matrix(
    cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues, save_path=None
):
    """
    Plot the confusion matrix.

    Args:
        cm (np.array): Confusion matrix.
        classes (list): List of class names.
        title (str, optional): Title of the plot. Default is "Confusion Matrix".
        cmap (matplotlib.colors.Colormap, optional): Colormap to use in the plot. Default is plt.cm.Blues.
        save_path (str, optional): If provided, the plot will be saved to this path.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_classification_report(
    report, title="Classification Report", cmap=plt.cm.Blues, save_path=None
):
    """
    Plot the classification report.

    Args:
        report (dict): Classification report.
        title (str, optional): Title of the plot. Default is "Classification Report".
        cmap (matplotlib.colors.Colormap, optional): Colormap to use in the plot. Default is plt.cm.Blues.
        save_path (str, optional): If provided, the plot will be saved to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.table(
        cellText=[[str(report[k][j]) for j in range(4)] for k in report.keys()],
        rowLabels=report.keys(),
        colLabels=["precision", "recall", "f1-score", "support"],
        cellLoc="center",
        loc="center",
        cellColours=plt.cm.Blues(np.full((len(report), 4), 0.1)),
        colColours=plt.cm.Blues(np.full(4, 0.3)),
        rowColours=plt.cm.Blues(np.full(len(report), 0.3)),
    )
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_roc_curve(fpr, tpr, title="ROC Curve", save_path=None):
    """
    Plot the ROC curve.

    Args:
        fpr (np.array): False positive rate.
        tpr (np.array): True positive rate.
        title (str, optional): Title of the plot. Default is "ROC Curve".
        save_path (str, optional): If provided, the plot will be saved to this path.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_model_architecture(model, input_size, device="cuda"):
    """
    Plot and summarize the architecture of a PyTorch model.

    Parameters:
    model (nn.Module): The PyTorch model to visualize.
    input_size (tuple): The size of the input tensor (e.g., (3, 224, 224) for an image).
    device (str): The device to use ('cuda' or 'cpu').

    Returns:
    None
    """
    # Move the model to the device
    model.to(device)

    # Print a summary of the model
    print("Model Summary:")
    summary(model, input_size=input_size)

    # Create a dummy input tensor with the specified size
    dummy_input = torch.randn(1, *input_size).to(device)

    # Generate the model graph
    model_graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))

    # Render the graph to a PDF file and display it
    model_graph.render("model_architecture", format="pdf")
    print("Model architecture graph saved as 'model_architecture.pdf'")
