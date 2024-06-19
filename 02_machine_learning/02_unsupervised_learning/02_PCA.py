import sys
import os

import torch

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------

import numpy as np
import time
import cProfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
# from common_components.evaluation_metrics import ExplainedVariance

class PCA:
    """
    Principal Component Analysis (PCA) implementation from scratch.
    """

    def __init__(self, n_components):
        """
        Initialize PCA with the number of principal components.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the model with X by computing the principal components.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).
        """
        # Step 1: Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Project the data onto the principal components.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)

    def explained_variance_ratio(self, X):
        """
        Compute the explained variance ratio for each principal component.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Explained variance ratio for each principal component.
        """
        X_centered = X - self.mean
        total_variance = np.var(X_centered, axis=0).sum()
        explained_variances = np.var(self.transform(X), axis=0)
        return explained_variances / total_variance


def test_and_benchmark():
    # Load a well-known dataset (IRIS)
    data = load_iris()
    X = data.data
    y = data.target

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the custom PCA model
    custom_pca = PCA(n_components=2)
    start_time = time.time()
    X_train_pca = custom_pca.fit_transform(X_train)
    custom_train_time = time.time() - start_time

    # Transform the test set
    X_test_pca = custom_pca.transform(X_test)
    custom_evr = custom_pca.explained_variance_ratio(X_test)

    # Print results for the custom PCA model
    print(f"\nCustom PCA - Explained Variance Ratio: {custom_evr}, Training Time: {custom_train_time:.4f}s")

    # Initialize and train the scikit-learn PCA model
    sklearn_pca = SklearnPCA(n_components=2)
    start_time = time.time()
    sklearn_pca.fit(X_train)
    sklearn_train_time = time.time() - start_time

    # Transform the test set
    X_test_pca_sklearn = sklearn_pca.transform(X_test)
    sklearn_evr = sklearn_pca.explained_variance_ratio_

    # Print results for the scikit-learn PCA model
    print(f"Scikit-learn PCA - Explained Variance Ratio: {sklearn_evr}, Training Time: {sklearn_train_time:.4f}s")

    # # Profiling the custom model training
    # cProfile.run('custom_pca.fit_transform(X_train)', 'pca_profile.prof')

    # Plot comparison of the PCA results
    plt.figure(figsize=(14, 6))

    # Custom PCA plot
    plt.subplot(1, 2, 1)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Custom PCA')
    plt.colorbar()

    # Scikit-learn PCA plot
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_pca_sklearn[:, 0], X_test_pca_sklearn[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scikit-learn PCA')
    plt.colorbar()

    plt.suptitle('PCA Comparison: Custom Implementation vs. Scikit-learn')
    plt.show()


if __name__ == "__main__":
    test_and_benchmark()

