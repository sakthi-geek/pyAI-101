import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, k=3):
        """
        Initialize the KNN model.

        Parameters:
        k (int): Number of neighbors to use for classification.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Train the KNN model.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Label array.
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
        x1 (ndarray): First point.
        x2 (ndarray): Second point.

        Returns:
        float: Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict(self, x):
        """
        Predict the class for a single sample.

        Parameters:
        x (ndarray): Feature vector.

        Returns:
        int: Predicted class.
        """
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def predict(self, X):
        """
        Predict the classes for the input samples.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        ndarray: Predicted classes.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def score(self, X, y):
        """
        Calculate the accuracy of the predictions.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): True labels.

        Returns:
        float: Accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def generate_data(n_samples=100):
    """
    Generate synthetic data for testing.

    Parameters:
    n_samples (int): Number of samples to generate.

    Returns:
    tuple: Feature matrix and label array.
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = np.array([0 if x[0] + x[1] < 0 else 1 for x in X])
    return X, y

def main():
    # Generate data
    X, y = generate_data()

    # From-scratch implementation
    knn = KNN(k=3)
    knn.fit(X, y)
    accuracy_scratch = knn.score(X, y)

    # Scikit-learn implementation
    sklearn_knn = SklearnKNN(n_neighbors=3)
    sklearn_knn.fit(X, y)
    accuracy_sklearn = sklearn_knn.score(X, y)

    # Print results
    print(f"From-scratch implementation accuracy: {accuracy_scratch}")
    print(f"Scikit-learn implementation accuracy: {accuracy_sklearn}")

if __name__ == "__main__":
    main()
