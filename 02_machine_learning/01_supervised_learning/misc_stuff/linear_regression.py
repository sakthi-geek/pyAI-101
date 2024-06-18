import numpy as np
import scikit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MSELoss:
    """Mean Squared Error Loss Function."""
    @staticmethod
    def compute_loss(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def compute_gradient(y_pred, y_true):
        return y_pred - y_true

class GradientDescentOptimizer:
    """Basic Gradient Descent Optimizer."""
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, parameter, gradient):
        return parameter - self.learning_rate * gradient

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True, normalize=False, regularization=None, alpha=1.0, early_stopping=False, tolerance=1e-4):
        """
        Initialize LinearRegression model.
        :param learning_rate: Learning rate for gradient descent.
        :param n_iterations: Number of iterations for gradient descent.
        :param fit_intercept: Boolean, whether to fit the intercept term.
        :param normalize: Boolean, whether to standardize the data.
        :param regularization: Type of regularization ('l1', 'l2', or None).
        :param alpha: Regularization strength (only for 'l1' and 'l2' regularization).
        :param early_stopping: Boolean, whether to use early stopping.
        :param tolerance: Minimum improvement in loss for stopping early.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.regularization = regularization
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

        self.scaler = StandardScaler() if normalize else None
        self.loss_function = MSELoss()
        self.optimizer = GradientDescentOptimizer(learning_rate)

    def _initialize_parameters(self, n_features):
        """Initialize the model parameters."""
        self.weights = np.zeros(n_features)
        self.bias = 0 if self.fit_intercept else None

    def _linear_function(self, X):
        """Compute the linear function of the input."""
        return np.dot(X, self.weights) + (self.bias if self.fit_intercept else 0)

    def _apply_regularization(self, dw, n_samples):
        """Apply regularization to the weight gradients."""
        if self.regularization == 'l1':
            dw += (self.alpha * np.sign(self.weights)) / n_samples
        elif self.regularization == 'l2':
            dw += (self.alpha * self.weights) / n_samples
        return dw

    def _compute_gradients(self, X, y, y_pred):
        """Compute gradients for gradient descent."""
        n_samples = len(y)
        error = self.loss_function.compute_gradient(y_pred, y)

        dw = np.dot(X.T, error) / n_samples
        dw = self._apply_regularization(dw, n_samples)
        db = np.sum(error) / n_samples if self.fit_intercept else 0

        return dw, db

    def fit(self, X, y):
        """Train the Linear Regression model using gradient descent."""
        if self.scaler:
            X = self.scaler.fit_transform(X)

        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        prev_loss = float('inf')
        for _ in range(self.n_iterations):
            y_pred = self._linear_function(X)
            dw, db = self._compute_gradients(X, y, y_pred)

            self.weights = self.optimizer.update(self.weights, dw)
            if self.fit_intercept:
                self.bias = self.optimizer.update(self.bias, db)

            # Early stopping
            current_loss = self.loss_function.compute_loss(y_pred, y)
            if self.early_stopping and abs(prev_loss - current_loss) < self.tolerance:
                break
            prev_loss = current_loss

        return self

    def predict(self, X):
        """Predict target values for given data using trained model."""
        if self.scaler:
            X = self.scaler.transform(X)
        return self._linear_function(X)

    def score(self, X, y, metric='r2'):
        """Calculate R2 score or another metric for the predictions."""
        y_pred = self.predict(X)

        if metric == 'r2':
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            return 1 - (ss_res / ss_total)
        elif metric == 'mse':
            return mean_squared_error(y, y_pred)
        elif metric == 'mae':
            return mean_absolute_error(y, y_pred)
        else:
            raise ValueError("Unsupported metric. Use 'r2', 'mse', or 'mae'.")

