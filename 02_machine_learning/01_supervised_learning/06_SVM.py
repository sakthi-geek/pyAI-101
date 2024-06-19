import sys
import os

import torch

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SklearnSVC
from common_components.evaluation_metrics import Accuracy, Precision, Recall, F1Score, confusion_matrix, ClassificationReport, RSquared
from common_components.loss_functions import HingeLoss
from common_components.optimizers import SGD
import time
import cProfile

class SVM:
    """
    Support Vector Machine (SVM) classifier.

    This implementation follows a similar structure to scikit-learn, with methods like fit(), predict(), and score().
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        """
        Initialize the SVM model.

        Args:
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
            lambda_param (float): Regularization parameter. Defaults to 0.01.
            epochs (int): Number of training epochs. Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the SVM model to the training data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y <= 0, -1, 1)

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        """
        Predict target values using the trained SVM model.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)

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


def test_and_benchmark():
    # Load and prepare the Iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # Binary classification for simplicity
    X, y = X[y != 2], y[y != 2]

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the scratch SVM model
    scratch_model = SVM(learning_rate=0.001, lambda_param=0.01, epochs=1000)
    start_time = time.time()
    scratch_model.fit(X_train, y_train)
    scratch_train_time = time.time() - start_time

    # Predict and evaluate the scratch model
    scratch_predictions = scratch_model.predict(X_test)
    scratch_accuracy = scratch_model.score(X_test, y_test)
    scratch_precision = Precision().compute(y_test, scratch_predictions)
    scratch_recall = Recall().compute(y_test, scratch_predictions)
    scratch_f1 = F1Score().compute(y_test, scratch_predictions)

    print(f"\nScratch Model - Accuracy: {scratch_accuracy:.4f}, Precision: {scratch_precision:.4f}, Recall: {scratch_recall:.4f}, F1 Score: {scratch_f1:.4f}, Training Time: {scratch_train_time:.4f}s")

    # Initialize and train the scikit-learn SVM model
    sklearn_model = SklearnSVC(kernel='linear', C=1.0)
    start_time = time.time()
    sklearn_model.fit(X_train, y_train)
    sklearn_train_time = time.time() - start_time

    # Predict and evaluate the scikit-learn model
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_accuracy = sklearn_model.score(X_test, y_test)
    sklearn_precision = Precision().compute(y_test, sklearn_predictions)
    sklearn_recall = Recall().compute(y_test, sklearn_predictions)
    sklearn_f1 = F1Score().compute(y_test, sklearn_predictions)

    print(f"Scikit-learn Model - Accuracy: {sklearn_accuracy:.4f}, Precision: {sklearn_precision:.4f}, Recall: {sklearn_recall:.4f}, F1 Score: {sklearn_f1:.4f}, Training Time: {sklearn_train_time:.4f}s")

    # # Profiling the scratch model prediction
    # cProfile.run('scratch_model.predict(X_test)', 'svm_profile.prof')

if __name__ == "__main__":
    test_and_benchmark()
