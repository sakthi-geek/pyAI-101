import numpy as np
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def __init__(self):
        """
        Initialize the Naive Bayes model.
        """
        self.classes = None
        self.means = None
        self.variances = None
        self.priors = None

    def fit(self, X, y):
        """
        Train the Naive Bayes model.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Label array.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.means = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variances = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[idx, :] = X_c.mean(axis=0)
            self.variances[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function for a given class and feature.

        Parameters:
        class_idx (int): Index of the class.
        x (float): Feature value.

        Returns:
        float: Probability density.
        """
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict(self, x):
        """
        Predict the class for a single sample.

        Parameters:
        x (ndarray): Feature vector.

        Returns:
        int: Predicted class.
        """
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

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
    nb = NaiveBayes()
    nb.fit(X, y)
    accuracy_scratch = nb.score(X, y)

    # Scikit-learn implementation
    sklearn_nb = SklearnGaussianNB()
    sklearn_nb.fit(X, y)
    accuracy_sklearn = sklearn_nb.score(X, y)

    # Print results
    print(f"From-scratch implementation accuracy: {accuracy_scratch}")
    print(f"Scikit-learn implementation accuracy: {accuracy_sklearn}")

if __name__ == "__main__":
    main()
