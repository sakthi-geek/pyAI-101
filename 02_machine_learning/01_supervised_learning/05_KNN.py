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
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from common_components.evaluation_metrics import Accuracy, Precision, Recall, F1Score
from collections import Counter
import time
import cProfile

class KNN:
    """
    K-Nearest Neighbors (KNN) classifier.

    This implementation follows a similar structure to scikit-learn, with methods like fit(), predict(), and score().
    """
    def __init__(self, k=3):
        """
        Initialize the KNN model.

        Args:
            k (int): Number of neighbors to consider. Defaults to 3.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fit the KNN model to the training data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict target values using the trained KNN model.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples,).
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predict the class label for a single sample.

        Args:
            x (np.ndarray): Input sample of shape (n_features,).

        Returns:
            int: Predicted class label.
        """
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

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

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the scratch KNN model
    scratch_model = KNN(k=3)
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

    # Initialize and train the scikit-learn KNN model
    sklearn_model = SklearnKNeighborsClassifier(n_neighbors=3)
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
    # cProfile.run('scratch_model.predict(X_test)', 'knn_profile.prof')

if __name__ == "__main__":
    test_and_benchmark()
