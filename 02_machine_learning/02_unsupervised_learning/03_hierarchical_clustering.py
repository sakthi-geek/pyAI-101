import sys
import os
import time

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import cProfile
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
# from common_components.evaluation_metrics import EvaluationMetric


class HierarchicalClustering:
    """
    Hierarchical Clustering Algorithm.
    """
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.linkage_matrix = None

    def fit(self, X):
        """
        Fit the hierarchical clustering on the data.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            self: Fitted estimator.
        """
        self.X = X
        self.distance_matrix = self._compute_distance_matrix(X)
        self.linkage_matrix = linkage(self.distance_matrix, method='single')
        self.labels_ = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust')
        return self

    def _compute_distance_matrix(self, X):
        """
        Compute the distance matrix.

        Args:
            X (np.ndarray): Data to cluster.

        Returns:
            np.ndarray: Distance matrix.
        """
        m = X.shape[0]
        distance_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
                distance_matrix[j, i] = distance_matrix[i, j]
        return distance_matrix

    def plot_dendrogram(self):
        """
        Plot the dendrogram.
        """
        dendrogram(self.linkage_matrix)
        plt.title("Dendrogram")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.show()


def test_and_benchmark():
    # Load a well-known dataset (IRIS)
    data = load_iris()
    X = data.data
    y = data.target

    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Initialize and train the custom hierarchical clustering model
    custom_hc = HierarchicalClustering(n_clusters=3)
    start_time = time.time()
    custom_hc.fit(X)
    custom_train_time = time.time() - start_time

    # Plot Dendrogram
    custom_hc.plot_dendrogram()

    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(14, 6))

    # Custom Hierarchical Clustering plot
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=custom_hc.labels_, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Custom Hierarchical Clustering')
    plt.colorbar()

    # Initialize and train the scikit-learn hierarchical clustering model
    sklearn_hc = AgglomerativeClustering(n_clusters=3)
    start_time = time.time()
    sklearn_labels = sklearn_hc.fit_predict(X)
    sklearn_train_time = time.time() - start_time

    # scikit-learn Hierarchical Clustering plot
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=sklearn_labels, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('scikit-learn Hierarchical Clustering')
    plt.colorbar()

    plt.suptitle('Hierarchical Clustering: Custom Implementation vs. scikit-learn')
    plt.show()

    # Print training times for comparison
    print(f"Custom Hierarchical Clustering - Training Time: {custom_train_time:.4f}s")
    print(f"scikit-learn Hierarchical Clustering - Training Time: {sklearn_train_time:.4f}s")

    # # Profiling the custom model training
    # cProfile.run('custom_hc.fit(X)', 'hc_profile.prof')


if __name__ == "__main__":
    test_and_benchmark()
