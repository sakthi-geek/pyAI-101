import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

class PCA:
    def __init__(self, n_components):
        """
        Initialize PCA model.

        Parameters:
        n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Parameters:
        X (ndarray): Data matrix with shape (n_samples, n_features).
        """
        # Step 1: Standardize the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Transform the data to the principal component space.

        Parameters:
        X (ndarray): Data matrix with shape (n_samples, n_features).

        Returns:
        ndarray: Transformed data matrix with shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Fit the PCA model to the data and transform it.

        Parameters:
        X (ndarray): Data matrix with shape (n_samples, n_features).

        Returns:
        ndarray: Transformed data matrix with shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)

def generate_data(n_samples=100, n_features=3):
    """
    Generate synthetic data for testing.

    Parameters:
    n_samples (int): Number of samples to generate.
    n_features (int): Number of features.

    Returns:
    ndarray: Generated data matrix with shape (n_samples, n_features).
    """
    np.random.seed(42)
    return np.random.randn(n_samples, n_features)

def main():
    # Generate data
    X = generate_data()

    # From-scratch implementation
    pca = PCA(n_components=2)
    X_transformed_scratch = pca.fit_transform(X)

    # Scikit-learn implementation
    sklearn_pca = SklearnPCA(n_components=2)
    X_transformed_sklearn = sklearn_pca.fit_transform(X)

    # Compare results
    print("From-scratch implementation results:\n", X_transformed_scratch[:5])
    print("Scikit-learn implementation results:\n", X_transformed_sklearn[:5])

if __name__ == "__main__":
    main()
