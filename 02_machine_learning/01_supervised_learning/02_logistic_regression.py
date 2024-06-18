import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score

class CrossEntropyLoss:
    """Cross Entropy Loss Function for Logistic Regression."""
    @staticmethod
    def compute_loss(y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def compute_gradient(y_pred, y_true):
        return y_pred - y_true

class GradientDescentOptimizer:
    """Basic Gradient Descent Optimizer."""
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, parameter, gradient):
        return parameter - self.learning_rate * gradient

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, init_strategy='zeros', 
                 fit_intercept=True, normalize=False, regularization=None, alpha=1.0, 
                 early_stopping=False, tolerance=1e-4):
        """
        Initialize LogisticRegression model.
        :param learning_rate: Learning rate for gradient descent.
        :param epochs: Number of iterations for gradient descent.
        :param init_strategy: Strategy for initializing weights ('zeros' or 'random').
        :param fit_intercept: Boolean, whether to fit the intercept term.
        :param normalize: Boolean, whether to standardize the data.
        :param regularization: Type of regularization ('l1', 'l2', or None).
        :param alpha: Regularization strength (only for 'l1' and 'l2' regularization).
        :param early_stopping: Boolean, whether to use early stopping.
        :param tolerance: Minimum improvement in loss for stopping early.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.init_strategy = init_strategy
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.regularization = regularization
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

        self.scaler = StandardScaler() if normalize else None
        self.loss_function = CrossEntropyLoss()
        self.optimizer = GradientDescentOptimizer(learning_rate)

    def _initialize_parameters(self, n_features):
        """Initialize the model parameters."""
        if self.init_strategy == 'zeros':
            self.weights = np.zeros(n_features)
            self.bias = 0 if self.fit_intercept else None
        elif self.init_strategy == 'random':
            self.weights = np.random.randn(n_features)
            self.bias = np.random.randn() if self.fit_intercept else None
        else:
            raise ValueError("Invalid initialization strategy")

    def _linear_function(self, X):
        """Compute the linear function of the input."""
        return np.dot(X, self.weights) + (self.bias if self.fit_intercept else 0)

    def _sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

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
        """Train the Logistic Regression model using gradient descent."""
        if self.scaler:
            X = self.scaler.fit_transform(X)

        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        prev_loss = float('inf')
        for _ in range(self.epochs):
            linear_output = self._linear_function(X)
            y_pred = self._sigmoid(linear_output)
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

    def predict_proba(self, X):
        """Predict probability estimates for given data using trained model."""
        if self.scaler:
            X = self.scaler.transform(X)
        linear_output = self._linear_function(X)
        return self._sigmoid(linear_output)

    def predict(self, X, threshold=0.5):
        """Predict binary target values for given data using trained model."""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score(self, X, y, metric='accuracy'):
        """Calculate accuracy or another metric for the predictions."""
        y_pred = self.predict(X)

        if metric == 'accuracy':
            return accuracy_score(y, y_pred)
        elif metric == 'mse':
            return mean_squared_error(y, y_pred)
        else:
            raise ValueError("Unsupported metric. Use 'accuracy' or 'mse'.")

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return (X - self.mean_) / self.scale_

    def transform(self, X):
        return (X - self.mean_) / self.scale_

def mean_squared_error(y_true, y_pred):
    """Calculate the Mean Squared Error (MSE) between true and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

def generate_data(n_samples=100):
    """Generate synthetic binary classification data for testing."""
    X = np.random.rand(n_samples, 1)
    y = (3 * X.squeeze() + 4 + np.random.randn(n_samples) * 0.5 > 4.5).astype(int)
    return X, y

def main():
    # Generate data
    X, y = generate_data()
    
    # From-scratch implementation
    lr = LogisticRegression(learning_rate=0.01, epochs=1000, init_strategy='zeros')
    lr.fit(X, y)
    y_pred_scratch = lr.predict(X)
    accuracy_scratch = lr.score(X, y, metric='accuracy')
    
    # Scikit-learn implementation
    sklearn_lr = SklearnLogisticRegression()
    sklearn_lr.fit(X, y)
    y_pred_sklearn = sklearn_lr.predict(X)
    accuracy_sklearn = accuracy_score(y, y_pred_sklearn)
    
    # Print results
    print(f"From-scratch implementation accuracy: {accuracy_scratch}")
    print(f"Scikit-learn implementation accuracy: {accuracy_sklearn}")

if __name__ == "__main__":
    main()
