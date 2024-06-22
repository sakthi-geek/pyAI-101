from pyAI.autograd.tensor import Tensor
from pyAI.nn.module import Module
import numpy as np

class Loss(Module):
    """
    Base class for all loss functions.
    """
    def forward(self, predictions, targets):
        raise NotImplementedError("Each loss must define its forward pass.")

    def backward(self, grad_output=None):
        raise NotImplementedError("Each loss must define its backward pass.")

#----------------------------------------------------------------------------------------------------

class MSELoss(Loss):
    """
    Mean Squared Error Loss: Computes the mean of squares of errors between predictions and actual targets.
    """
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        # print("predictions.data", predictions.data)
        # print("targets.data", targets.data)
        loss = np.mean((predictions.data - targets.data) ** 2)
        loss = Tensor(loss)
        loss.set_grad_fn(self)
        return loss

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = np.ones_like(self.predictions.data)
        if not isinstance(grad_output, Tensor):
            grad_output = Tensor(grad_output)
        
        # print("grad_output", grad_output.shape, grad_output)
        # print(self.predictions.data.shape, self.targets.data.shape, self.targets.data.size)

        # Compute the gradient of the loss with respect to predictions
        grad_predictions  = 2 * (self.predictions.data - self.targets.data) / self.targets.data.size

        # print("grad_predictions", grad_output.data.shape, grad_predictions.shape, grad_predictions)
        grad_output_data = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        self.predictions.backward(Tensor(grad_output_data  * grad_predictions))

#----------------------------------------------------------------------------------------------------

class BCELoss(Loss):
    """
    Binary Cross Entropy Loss: Computes the binary cross entropy loss between predictions and actual targets.
    """
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        preds_clipped = np.clip(predictions.data, 1e-7, 1 - 1e-7)  # Avoid log(0)
        # Calculate BCE loss
        loss = -np.mean(targets.data * np.log(preds_clipped) + (1 - targets.data) * np.log(1 - preds_clipped))
        loss = Tensor(loss)
        loss.set_grad_fn(self)
        return loss

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = 1   # Default gradient that multiplies the loss gradient

        preds_clipped = np.clip(self.predictions.data, 1e-7, 1 - 1e-7)
        # Calculate gradient with respect to predictions
        grad_predictions = -(self.targets.data / preds_clipped - (1 - self.targets.data) / (1 - preds_clipped)) 
        
        grad_output_data = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        self.predictions.backward(Tensor(grad_output_data * grad_predictions))

#----------------------------------------------------------------------------------------------------

class CrossEntropyLoss(Loss):
    """
    Cross Entropy Loss: Computes the cross entropy loss between predictions and actual targets.
    """
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        # Ensure targets are of integer type for array indexing
        target_indices = targets.data.astype(int)

        # Compute softmax
        # Compute the maximum logit to subtract for numerical stability (Log-Sum-Exp trick)
        max_logits = np.max(predictions.data, axis=1, keepdims=True)
        # Subtract max logit for stability and exponentiate
        exps = np.exp(predictions.data - max_logits)

        # Compute the sum of exps for each sample and log it for softmax denominator
        sum_exps = np.sum(exps, axis=1, keepdims=True)
        log_sum_exps = np.log(sum_exps)

        # Compute log softmax
        log_softmax = predictions.data - max_logits - log_sum_exps
        # Gather the log softmax based on target indices
        # np.arange(targets.data.shape[0]) creates an array of indices [0, 1, ..., N-1], where N is batch size
        # targets.data selects the log softmax of the correct class

        log_softmax_correct_class = log_softmax[np.arange(target_indices.shape[0]), target_indices]

        # Compute the negative log likelihood loss
        loss = -np.mean(log_softmax_correct_class)
        loss = Tensor(loss)
        loss.set_grad_fn(self)
        return loss

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = 1  # Scalar multiplication with the gradient if not provided

        # Convert targets to integer indices if they are not already
        target_indices = self.targets.data.astype(int)

        # Calculate softmax again
        max_logits = np.max(self.predictions.data, axis=1, keepdims=True)
        exps = np.exp(self.predictions.data - max_logits)
        sum_exps = np.sum(exps, axis=1, keepdims=True)
        softmax = exps / sum_exps

        # Gradient of softmax with cross entropy loss
        # Subtract 1 from the softmax scores of the correct class
        softmax[np.arange(target_indices.shape[0]), target_indices] -= 1

        # Average over the batch
        grad_predictions = softmax / self.targets.data.shape[0]

        grad_output_data = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        
        self.predictions.backward(Tensor(grad_output_data * grad_predictions))

#----------------------------------------------------------------------------------------------------



