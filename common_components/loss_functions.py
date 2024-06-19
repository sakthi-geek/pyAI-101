"""
loss_functions.py

This module provides various loss functions commonly used in machine learning and deep learning models.
Each function is implemented from scratch to help learners understand the underlying mechanics.
"""

import numpy as np

class LossFunction:
    """
    Base class for all loss functions.
    """
    def forward(self, y_true, y_pred):
        raise NotImplementedError("Forward method not implemented!")

    def backward(self, y_true, y_pred):
        raise NotImplementedError("Backward method not implemented!")

#-------------------------------------------------------------------------------------------

class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss function.
    
    L(y, y_hat) = 1/N * sum((y - y_hat)^2)
    """
    def forward(self, y_true, y_pred):
        """
        Forward pass of the MSE loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed MSE loss.
        """
        self.loss = np.mean((y_true - y_pred) ** 2)
        return self.loss

    def backward(self, y_true, y_pred):
        """
        Backward pass (derivative) of the MSE loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Gradient of the MSE loss function.
        """
        return (2 / y_true.size) * (y_pred - y_true)

#-------------------------------------------------------------------------------------------

class MeanAbsoluteError(LossFunction):
    """
    Mean Absolute Error (MAE) loss function.
    
    L(y, y_hat) = 1/N * sum(|y - y_hat|)
    """
    def forward(self, y_true, y_pred):
        """
        Forward pass of the MAE loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed MAE loss.
        """
        self.loss = np.mean(np.abs(y_true - y_pred))
        return self.loss

    def backward(self, y_true, y_pred):
        """
        Backward pass (derivative) of the MAE loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Gradient of the MAE loss function.
        """
        return np.where(y_pred > y_true, 1, -1) / y_true.size

#-------------------------------------------------------------------------------------------

class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross-Entropy (BCE) loss function.
    
    L(y, y_hat) = -1/N * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
    """
    def forward(self, y_true, y_pred):
        """
        Forward pass of the BCE loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels (between 0 and 1).

        Returns:
            float: Computed BCE loss.
        """
        self.loss = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return self.loss

    def backward(self, y_true, y_pred):
        """
        Backward pass (derivative) of the BCE loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels (between 0 and 1).

        Returns:
            np.ndarray: Gradient of the BCE loss function.
        """
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15)

#-------------------------------------------------------------------------------------------

class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross-Entropy (CCE) loss function.
    
    L(y, y_hat) = -1/N * sum(y*log(y_hat))
    """
    def forward(self, y_true, y_pred):
        """
        Forward pass of the CCE loss function.

        Args:
            y_true (np.ndarray): True labels (one-hot encoded).
            y_pred (np.ndarray): Predicted labels (probabilities).

        Returns:
            float: Computed CCE loss.
        """
        self.loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
        return self.loss

    def backward(self, y_true, y_pred):
        """
        Backward pass (derivative) of the CCE loss function.

        Args:
            y_true (np.ndarray): True labels (one-hot encoded).
            y_pred (np.ndarray): Predicted labels (probabilities).

        Returns:
            np.ndarray: Gradient of the CCE loss function.
        """
        return -y_true / (y_pred + 1e-15)

#-------------------------------------------------------------------------------------------

class HuberLoss(LossFunction):
    """
    Huber loss function.
    
    L(y, y_hat) = 1/N * sum(0.5 * (y - y_hat)^2) if |y - y_hat| <= delta
                 1/N * sum(delta * |y - y_hat| - 0.5 * delta^2) otherwise
    """
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_true, y_pred):
        """
        Forward pass of the Huber loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed Huber loss.
        """
        diff = np.abs(y_true - y_pred)
        self.loss = np.mean(np.where(diff <= self.delta, 0.5 * diff ** 2, self.delta * diff - 0.5 * self.delta ** 2))
        return self.loss

    def backward(self, y_true, y_pred):
        """
        Backward pass (derivative) of the Huber loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Gradient of the Huber loss function.
        """
        diff = y_true - y_pred
        return np.where(np.abs(diff) <= self.delta, diff, np.sign(diff) * self.delta)
    
#-------------------------------------------------------------------------------------------

class HingeLoss(LossFunction):
    """
    Hinge loss function.
    
    L(y, y_hat) = 1/N * sum(max(0, 1 - y*y_hat))
    """
    def forward(self, y_true, y_pred):
        """
        Forward pass of the Hinge loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed Hinge loss.
        """
        self.loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
        return self.loss

    def backward(self, y_true, y_pred):
        """
        Backward pass (derivative) of the Hinge loss function.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Gradient of the Hinge loss function.
        """
        return -y_true * (y_true * y_pred < 1) / y_true.size
    
#-------------------------------------------------------------------------------------------

#================================================================================================

# Example usage:
if __name__ == "__main__":
    y_true_reg = np.array([1.5, 2.0, 3.5])
    y_pred_reg = np.array([1.0, 2.5, 3.0])

    mse = MeanSquaredError()
    print("MSE Loss forward:\n", mse.forward(y_true_reg, y_pred_reg))
    print("MSE Loss backward:\n", mse.backward(y_true_reg, y_pred_reg))

    mae = MeanAbsoluteError()
    print("MAE Loss forward:\n", mae.forward(y_true_reg, y_pred_reg))
    print("MAE Loss backward:\n", mae.backward(y_true_reg, y_pred_reg))

    y_true_bin = np.array([1, 0, 1])
    y_pred_bin = np.array([0.9, 0.1, 0.8])

    bce = BinaryCrossEntropy()
    print("BCE Loss forward:\n", bce.forward(y_true_bin, y_pred_bin))
    print("BCE Loss backward:\n", bce.backward(y_true_bin, y_pred_bin))

    y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred_cat = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])

    cce = CategoricalCrossEntropy()
    print("CCE Loss forward:\n", cce.forward(y_true_cat, y_pred_cat))
    print("CCE Loss backward:\n", cce.backward(y_true_cat, y_pred_cat))

    huber = HuberLoss(delta=1.0)
    print("Huber Loss forward:\n", huber.forward(y_true_reg, y_pred_reg))
    print("Huber Loss backward:\n", huber.backward(y_true_reg, y_pred_reg))

    hinge = HingeLoss()
    print("Hinge Loss forward:\n", hinge.forward(y_true_bin, y_pred_bin))
    print("Hinge Loss backward:\n", hinge.backward(y_true_bin, y_pred_bin))

