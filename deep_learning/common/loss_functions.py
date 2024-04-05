import numpy as np

class MSELoss:
    def forward(self, predictions, targets):
        """
        Forward pass for MSE loss.
        """
        self.difference = predictions - targets
        return np.mean(self.difference ** 2)

    def backward(self):
        """
        Backward pass for MSE loss.
        """
        return 2 * self.difference / np.size(self.difference)


class CrossEntropyLoss:
    def forward(self, logits, targets):
        """
        Forward pass for Cross-Entropy loss with logits.
        """
        self.softmax = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.softmax /= np.sum(self.softmax, axis=1, keepdims=True)
        self.targets = targets
        return -np.sum(targets * np.log(self.softmax + 1e-15)) / logits.shape[0]

    def backward(self):
        """
        Backward pass for the Cross-Entropy loss.
        """
        return (self.softmax - self.targets) / self.targets.shape[0]


class BinaryCrossEntropyLoss:
    def forward(self, logits, targets):
        """
        Forward pass for Binary Cross-Entropy loss.
        """
        self.predictions = 1 / (1 + np.exp(-logits))
        self.targets = targets
        return -np.mean(targets * np.log(self.predictions + 1e-15) + (1 - targets) * np.log(1 - self.predictions + 1e-15))

    def backward(self):
        """
        Backward pass for Binary Cross-Entropy loss.
        """
        return (self.predictions - self.targets) / (self.predictions * (1 - self.predictions) * np.size(self.targets))


class L1Loss: # Absolute error
    def forward(self, predictions, targets):
        """
        Compute the L1 Loss between predictions and targets.
        """
        self.difference = predictions - targets
        return np.mean(np.abs(self.difference))

    def backward(self):
        """
        Compute the gradient of L1 Loss with respect to the predictions.
        """
        return np.where(self.difference > 0, 1, -1) / self.difference.size


class SoftmaxCrossEntropyLoss:
    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, logits, targets):
        """
        Forward pass for the combined softmax and cross-entropy loss.
        """
        self.softmax_output = self.softmax(logits)
        self.targets = targets
        # Using clip to prevent NaNs in log
        clipped_softmax_output = np.clip(self.softmax_output, 1e-15, 1 - 1e-15)
        return -np.sum(targets * np.log(clipped_softmax_output)) / logits.shape[0]

    def backward(self):
        """
        Backward pass for softmax cross-entropy loss.
        """
        return (self.softmax_output - self.targets) / self.targets.shape[0]

