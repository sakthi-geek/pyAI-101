import numpy as np
from sklearn.svm import SVC as SklearnSVC
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the Support Vector Machine model.

        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        lambda_param (float): Regularization parameter.
        n_iters (int): Number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the SVM model using gradient descent.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Label array.
        """
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to {-1, 1}

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        """
        Predict the classes of the input samples.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        ndarray: Predicted classes.
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

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
    y = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 > 1, 1, 0)
    return X, y

def main():
    # Generate data
    X, y = generate_data()

    # From-scratch implementation
    svm = SVM()
    svm.fit(X, y)
    accuracy_scratch = svm.score(X, y)

    # Scikit-learn implementation
    sklearn_svm = SklearnSVC()
    sklearn_svm.fit(X, y)
    accuracy_sklearn = sklearn_svm.score(X, y)

    # Print results
    print(f"From-scratch implementation accuracy: {accuracy_scratch}")
    print(f"Scikit-learn implementation accuracy: {accuracy_sklearn}")

if __name__ == "__main__":
    main()
