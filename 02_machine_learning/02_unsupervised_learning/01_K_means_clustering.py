import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import pairwise_distances_argmin

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        """
        Initialize the KMeans model.

        Parameters:
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance to declare convergence.
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        """
        Train the KMeans model.

        Parameters:
        X (ndarray): Feature matrix.
        """
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)

            # Compute new centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        """
        Assign clusters based on current centroids.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        ndarray: Assigned cluster labels.
        """
        return pairwise_distances_argmin(X, self.centroids)

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        ndarray: Predicted cluster labels.
        """
        return self._assign_clusters(X)

def generate_data(n_samples=300, n_features=2, n_clusters=3):
    """
    Generate synthetic data for testing.

    Parameters:
    n_samples (int): Number of samples to generate.
    n_features (int): Number of features.
    n_clusters (int): Number of clusters.

    Returns:
    tuple: Feature matrix and true labels.
    """
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    return X, y

def main():
    # Generate data
    X, y = generate_data()

    # From-scratch implementation
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    y_pred_scratch = kmeans.predict(X)

    # Scikit-learn implementation
    sklearn_kmeans = SklearnKMeans(n_clusters=3, random_state=42)
    sklearn_kmeans.fit(X)
    y_pred_sklearn = sklearn_kmeans.predict(X)

    # Print results
    print(f"From-scratch implementation centroids:\n{kmeans.centroids}")
    print(f"Scikit-learn implementation centroids:\n{sklearn_kmeans.cluster_centers_}")

if __name__ == "__main__":
    main()
