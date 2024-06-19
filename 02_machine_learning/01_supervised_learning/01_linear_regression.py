import sys
import os

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------

import numpy as np
from common_components.loss_functions import MeanSquaredError
from common_components.optimizers import SGD
from common_components.evaluation_metrics import MeanSquaredError as MSEMetric, RSquared
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import time
import cProfile

class LinearRegression:
    """
    Linear Regression model.

    This implementation follows a similar structure to scikit-learn, with methods like fit(), predict(), and score().
    """
    def __init__(self, learning_rate=0.01, epochs=1000, optimizer=None, loss_function=None):
        """
        Initialize the Linear Regression model.

        Args:
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.01.
            epochs (int): Number of training epochs. Defaults to 1000.
            optimizer (Optimizer): Optimizer for updating the weights. Defaults to SGD.
            loss_function (LossFunction): Loss function for training. Defaults to MeanSquaredError.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer if optimizer else SGD(learning_rate=self.learning_rate)
        self.loss_function = loss_function if loss_function else MeanSquaredError()
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        """
        Fit the Linear Regression model to the training data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = self._predict(X)

            # Compute loss
            loss = self.loss_function.forward(y, y_pred)
            self.losses.append(loss)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights = self.optimizer.update(self.weights, dw)
            self.bias -= self.learning_rate * db

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        """
        Predict target values using the trained Linear Regression model.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        return self._predict(X)

    def score(self, X, y):
        """
        Evaluate the model using the R-squared metric.

        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            y (np.ndarray): True values of shape (n_samples,).

        Returns:
            float: R-squared score.
        """
        y_pred = self.predict(X)
        r_squared = RSquared().compute(y, y_pred)
        return r_squared

    def _predict(self, X):
        """
        Internal method to compute the linear prediction.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Linear prediction of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias


def test_and_benchmark():
    # Load and prepare the California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the scratch Linear Regression model
    scratch_model = LinearRegression(learning_rate=0.01, epochs=1000)
    start_time = time.time()
    scratch_model.fit(X_train, y_train)
    scratch_train_time = time.time() - start_time

    # Predict and evaluate the scratch model
    scratch_predictions = scratch_model.predict(X_test)
    scratch_r_squared = scratch_model.score(X_test, y_test)
    scratch_mse = MSEMetric().compute(y_test, scratch_predictions)

    print(f"\nScratch Model - R-squared: {scratch_r_squared:.4f}, MSE: {scratch_mse:.4f}, Training Time: {scratch_train_time:.4f}s")

    # Initialize and train the scikit-learn Linear Regression model
    sklearn_model = SklearnLinearRegression()
    start_time = time.time()
    sklearn_model.fit(X_train, y_train)
    sklearn_train_time = time.time() - start_time

    # Predict and evaluate the scikit-learn model
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_r_squared = sklearn_model.score(X_test, y_test)
    sklearn_mse = MSEMetric().compute(y_test, sklearn_predictions)

    print(f"Scikit-learn Model - R-squared: {sklearn_r_squared:.4f}, MSE: {sklearn_mse:.4f}, Training Time: {sklearn_train_time:.4f}s")

    # # Profiling the scratch model training
    # cProfile.run('scratch_model.fit(X_train, y_train)', 'linear_regression_profile.prof')

if __name__ == "__main__":
    test_and_benchmark()
