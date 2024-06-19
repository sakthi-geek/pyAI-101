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
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cProfile
from common_components.evaluation_metrics import MeanSquaredError as MSEMetric

class KMeans:
    """
    K-Means Clustering algorithm.
    
    This implementation follows a similar structure to scikit-learn, with methods like fit(), predict(), and score().
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        Initialize the K-Means model.

        Args:
            n_clusters (int): Number of clusters. Defaults to 3.
            max_iter (int): Maximum number of iterations. Defaults to 300.
            tol (float): Tolerance for convergence. Defaults to 1e-4.
            random_state (int): Random seed. Defaults to None.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        """
        if self.random_state:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for i in range(self.max_iter):
            # Assign clusters
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # Compute new centroids
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            
            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break
                
            self.centroids = new_centroids

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted cluster labels of shape (n_samples,).
        """
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def score(self, X):
        """
        Compute the mean squared error between points and their corresponding cluster centers.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            float: Mean squared error score.
        """
        labels = self.predict(X)
        mse_metric = MSEMetric()
        mse = np.mean([mse_metric.compute(X[labels == i], self.centroids[i]) for i in range(self.n_clusters)])
        return mse

    def _compute_distances(self, X):
        """
        Compute the distance between each sample and each centroid.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Distance matrix of shape (n_samples, n_clusters).
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return distances


def test_and_benchmark():
    # Generate a synthetic dataset
    X, y = make_blobs(n_samples=1000, centers=3, n_features=5, random_state=42)

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the scratch K-Means model
    scratch_model = KMeans(n_clusters=3, max_iter=300, tol=1e-4, random_state=42)
    start_time = time.time()
    scratch_model.fit(X_train)
    scratch_train_time = time.time() - start_time

    # Predict and evaluate the scratch model
    scratch_predictions = scratch_model.predict(X_test)
    scratch_silhouette_score = silhouette_score(X_test, scratch_predictions)

    print(f"\nScratch Model - Silhouette Score: {scratch_silhouette_score:.4f}, Training Time: {scratch_train_time:.4f}s")

    # Initialize and train the scikit-learn K-Means model
    sklearn_model = SklearnKMeans(n_clusters=3, max_iter=300, tol=1e-4, random_state=42)
    start_time = time.time()
    sklearn_model.fit(X_train)
    sklearn_train_time = time.time() - start_time

    # Predict and evaluate the scikit-learn model
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_silhouette_score = silhouette_score(X_test, sklearn_predictions)

    print(f"Scikit-learn Model - Silhouette Score: {sklearn_silhouette_score:.4f}, Training Time: {sklearn_train_time:.4f}s")

    # # Profiling the scratch model training
    # cProfile.run('scratch_model.fit(X_train)', 'k_means_clustering_profile.prof')


if __name__ == "__main__":
    test_and_benchmark()
