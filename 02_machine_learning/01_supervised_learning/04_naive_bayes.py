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
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from common.metrics import Accuracy, Precision, Recall, F1Score, LogLoss
import time
import cProfile

class NaiveBayes:
    """
    Naive Bayes classifier.

    This implementation follows a similar structure to scikit-learn, with methods like fit(), predict(), and score().
    """
    def fit(self, X, y):
        """
        Fit the Naive Bayes model to the training data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate mean, var, and prior for each class
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predict target values using the trained Naive Bayes model.

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
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function for a given class and sample.

        Args:
            class_idx (int): Index of the class.
            x (np.ndarray): Input sample of shape (n_features,).

        Returns:
            np.ndarray: Probability density function values.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

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

    # Initialize and train the scratch Naive Bayes model
    scratch_model = NaiveBayes()
    start_time = time.time()
    scratch_model.fit(X_train, y_train)
    scratch_train_time = time.time() - start_time

    # Predict and evaluate the scratch model
    scratch_predictions = scratch_model.predict(X_test)
    scratch_accuracy = scratch_model.score(X_test, y_test)
    scratch_precision = Precision().compute(y_test, scratch_predictions)
    scratch_recall = Recall().compute(y_test, scratch_predictions)
    scratch_f1 = F1Score().compute(y_test, scratch_predictions)
    scratch_log_loss = LogLoss().compute(y_test, scratch_model.predict(X_test))

    print(f"\nScratch Model - Accuracy: {scratch_accuracy:.4f}, Precision: {scratch_precision:.4f}, Recall: {scratch_recall:.4f}, F1 Score: {scratch_f1:.4f}, Log Loss: {scratch_log_loss:.4f}, Training Time: {scratch_train_time:.4f}s")

    # Initialize and train the scikit-learn Naive Bayes model
    sklearn_model = SklearnGaussianNB()
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
    # cProfile.run('scratch_model.fit(X_train, y_train)', 'naive_bayes_profile.prof')

if __name__ == "__main__":
    test_and_benchmark()
