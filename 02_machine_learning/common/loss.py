import numpy as np
from pyAI.nn.module import Module


#-------------------------------------------------------------------------------------------

class MSELoss (Module):  # MeanSquaredError # used for regression tasks
    """
    Mean Squared Error (MSE) loss function.
    
    L(y, y_pred) = 1/N * sum((y - y_pred)^2)
    """
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

#-------------------------------------------------------------------------------------------

class MAELoss(Module):   # L1Loss # used for regression tasks
    """
    Mean Absolute Error (MAE) loss function.
    
    L(y, y_pred) = 1/N * sum(|y - y_pred|)
    """
    def forward(self, y_true, y_pred):
        self.loss = np.mean(np.abs(y_true - y_pred))
        return self.loss

    def backward(self, y_true, y_pred):
        return np.where(y_pred > y_true, 1, -1) / y_true.size

#-------------------------------------------------------------------------------------------

class BCELoss(Module):  # BinaryCrossEntropy # used for binary classification tasks
    """
    Binary Cross-Entropy (BCE) loss function.
    
    L(y, y_pred) = -1/N * sum(y*log(y_pred) + (1-y)*log(1-y_pred))
    """
    def forward(self, y_true, y_pred):
        self.loss = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return self.loss

    def backward(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15)

#-------------------------------------------------------------------------------------------

class CrossEntropyLoss(Module):  # CategoricalCrossEntropy    # used for multi-class classification tasks

    """
    Combined Softmax activation and Categorical Cross-Entropy (CCE) loss function.
    
    Softmax: f(x) = exp(x) / sum(exp(x))
    CCE Loss: L(y, y_hat) = -sum(y*log(y_hat))
    """
    def forward(self, y_true, y_pred):
        # Softmax
        exps = np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True))
        self.softmax_output = exps / np.sum(exps, axis=-1, keepdims=True)
        
        # Categorical Cross-Entropy
        self.y_true = y_true
        self.loss = -np.sum(y_true * np.log(self.softmax_output + 1e-15)) / y_true.shape[0]
        return self.loss

    def backward(self, y_true, y_pred):
        # The gradient for softmax combined with cross-entropy
        return (self.softmax_output - y_true) / y_true.shape[0]

#-------------------------------------------------------------------------------------------

class L1Loss(Module):   # MeanAbsoluteError # used for regression tasks
    """
    L1 loss function.
    
    L(y, y_pred) = 1/N * sum(|y - y_pred|)
    """
    def forward(self, y_true, y_pred):
        self.loss = np.mean(np.abs(y_true - y_pred))
        return self.loss

    def backward(self, y_true, y_pred):
        return np.where(y_pred > y_true, 1, -1) / y_true.size
    
#-------------------------------------------------------------------------------------------

class HuberLoss(Module):   # Smooth L1 loss   # used for regression tasks, more robust and less sensitive to outliers than MSE
    """
    Huber loss function.
    
    L(y, y_pred) = 1/N * sum(0.5 * (y - y_pred)^2) if |y - y_pred| <= delta
                 1/N * sum(delta * |y - y_pred| - 0.5 * delta^2) otherwise
    """
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_true, y_pred):
        diff = np.abs(y_true - y_pred)
        self.loss = np.mean(np.where(diff <= self.delta, 0.5 * diff ** 2, self.delta * diff - 0.5 * self.delta ** 2))
        return self.loss

    def backward(self, y_true, y_pred):
        diff = y_true - y_pred
        return np.where(np.abs(diff) <= self.delta, diff, np.sign(diff) * self.delta)
    
#-------------------------------------------------------------------------------------------

class HingeLoss(Module):  # used for binary classification tasks - commonly used in SVMs 
    """
    Hinge loss function.
    
    L(y, y_pred) = 1/N * sum(max(0, 1 - y*y_pred))
    """
    def forward(self, y_true, y_pred):
        self.loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
        return self.loss

    def backward(self, y_true, y_pred):
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

