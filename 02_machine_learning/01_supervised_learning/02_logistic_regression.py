import sys
import os

import torch

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------

import numpy as np
from common_components.activation_functions import Sigmoid
from common_components.loss_functions import BinaryCrossEntropy
from common_components.optimizers import SGD
from common_components.evaluation_metrics import Accuracy, Precision, Recall, F1Score, LogLoss
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import time
import cProfile

class LogisticRegression:
    """
    Logistic Regression model.

    This implementation follows a similar structure to scikit-learn, with methods like fit(), predict(), and score().
    """
    def __init__(self, learning_rate=0.01, epochs=1000, optimizer=None, loss_function=None, threshold=0.5):
        """
        Initialize the Logistic Regression model.

        Args:
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.01.
            epochs (int): Number of training epochs. Defaults to 1000.
            optimizer (Optimizer): Optimizer for updating the weights. Defaults to SGD.
            loss_function (LossFunction): Loss function for training. Defaults to BinaryCrossEntropy.
            threshold (float): Decision threshold for classification. Defaults to 0.5.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer if optimizer else SGD(learning_rate=self.learning_rate)
        self.loss_function = loss_function if loss_function else BinaryCrossEntropy()
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.losses = []
        self.activation = Sigmoid()

    def fit(self, X, y):
        """
        Fit the Logistic Regression model to the training data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            linear_model = self._linear_combination(X)
            y_pred = self.activation.forward(linear_model)

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

    def predict_proba(self, X):
        """
        Predict probabilities using the trained Logistic Regression model.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples,).
        """
        linear_model = self._linear_combination(X)
        return self.activation.forward(linear_model)

    def predict(self, X):
        """
        Predict binary target values using the trained Logistic Regression model.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted binary target values of shape (n_samples,).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)

    def score(self, X, y):
        """
        Evaluate the model using the accuracy metric.

        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            y (np.ndarray): True values of shape (n_samples,).

        Returns:
            float: Accuracy score.
        """
        y_pred = self.predict(X)
        accuracy = Accuracy().compute(y, y_pred)
        return accuracy

    def _linear_combination(self, X):
        """
        Internal method to compute the linear combination of input features and weights.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Linear combination of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias


def test_and_benchmark():
    # Load and prepare the Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the scratch Logistic Regression model
    scratch_model = LogisticRegression(learning_rate=0.01, epochs=1000)
    start_time = time.time()
    scratch_model.fit(X_train, y_train)
    scratch_train_time = time.time() - start_time

    # Predict and evaluate the scratch model
    scratch_predictions = scratch_model.predict(X_test)
    scratch_accuracy = scratch_model.score(X_test, y_test)
    scratch_precision = Precision().compute(y_test, scratch_predictions)
    scratch_recall = Recall().compute(y_test, scratch_predictions)
    scratch_f1 = F1Score().compute(y_test, scratch_predictions)
    scratch_log_loss = LogLoss().compute(y_test, scratch_model.predict_proba(X_test))

    print(f"\nScratch Model - Accuracy: {scratch_accuracy:.4f}, Precision: {scratch_precision:.4f}, Recall: {scratch_recall:.4f}, F1 Score: {scratch_f1:.4f}, Log Loss: {scratch_log_loss:.4f}, Training Time: {scratch_train_time:.4f}s")

    # Initialize and train the scikit-learn Logistic Regression model
    sklearn_model = SklearnLogisticRegression(max_iter=1000)
    start_time = time.time()
    sklearn_model.fit(X_train, y_train)
    sklearn_train_time = time.time() - start_time

    # Predict and evaluate the scikit-learn model
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_accuracy = sklearn_model.score(X_test, y_test)
    sklearn_precision = Precision().compute(y_test, sklearn_predictions)
    sklearn_recall = Recall().compute(y_test, sklearn_predictions)
    sklearn_f1 = F1Score().compute(y_test, sklearn_predictions)
    sklearn_log_loss = LogLoss().compute(y_test, sklearn_model.predict_proba(X_test)[:, 1])

    print(f"Scikit-learn Model - Accuracy: {sklearn_accuracy:.4f}, Precision: {sklearn_precision:.4f}, Recall: {sklearn_recall:.4f}, F1 Score: {sklearn_f1:.4f}, Log Loss: {sklearn_log_loss:.4f}, Training Time: {sklearn_train_time:.4f}s")

    # # Profiling the scratch model training
    # cProfile.run('scratch_model.fit(X_train, y_train)', 'logistic_regression_profile.prof')

if __name__ == "__main__":
    test_and_benchmark()
