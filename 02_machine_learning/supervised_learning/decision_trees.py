import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.metrics import accuracy_score

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        """
        Initialize the Decision Tree model.
        :param max_depth: Maximum depth of the tree.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        """
        Calculate Gini Impurity for a list of labels.
        :param y: Array of labels.
        :return: Gini Impurity.
        """
        m = len(y)
        if m == 0:
            return 0
        unique_classes, counts = np.unique(y, return_counts=True)
        probs = counts / m
        gini = 1 - np.sum(probs ** 2)
        return gini

    def _best_split(self, X, y):
        """
        Find the best split for the data.
        :param X: Feature matrix.
        :param y: Label array.
        :return: Best split parameters (feature index, threshold, left and right splits).
        """
        m, n = X.shape
        best_gini = float('inf')
        best_split = {}
        
        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                left_gini = self._gini(y[left_indices])
                right_gini = self._gini(y[right_indices])
                gini = (len(left_indices) / m) * left_gini + (len(right_indices) / m) * right_gini
                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
        return best_split

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        :param X: Feature matrix.
        :param y: Label array.
        :param depth: Current depth of the tree.
        :return: DecisionTreeNode.
        """
        m, n = X.shape
        if m >= self.min_samples_split and depth < self.max_depth:
            best_split = self._best_split(X, y)
            if best_split:
                left_subtree = self._build_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
                right_subtree = self._build_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)
                return DecisionTreeNode(feature_index=best_split['feature_index'], threshold=best_split['threshold'], left=left_subtree, right=right_subtree)
        
        leaf_value = self._leaf_value(y)
        return DecisionTreeNode(value=leaf_value)

    def _leaf_value(self, y):
        """
        Determine the value of a leaf node.
        :param y: Array of labels.
        :return: Leaf value.
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        return unique_classes[np.argmax(counts)]

    def fit(self, X, y):
        """
        Build the decision tree using the training data.
        :param X: Feature matrix.
        :param y: Label array.
        """
        self.root = self._build_tree(X, y)

    def _predict_sample(self, x, tree):
        """
        Predict the class of a single sample.
        :param x: Feature vector of a single sample.
        :param tree: Decision tree.
        :return: Predicted class.
        """
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._predict_sample(x, tree.left)
        else:
            return self._predict_sample(x, tree.right)

    def predict(self, X):
        """
        Predict the classes of the input samples.
        :param X: Feature matrix.
        :return: Predicted classes.
        """
        return np.array([self._predict_sample(x, self.root) for x in X])

    def score(self, X, y):
        """
        Calculate the accuracy of the predictions.
        :param X: Feature matrix.
        :param y: True labels.
        :return: Accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def generate_data(n_samples=100):
    """
    Generate synthetic data for testing.
    :param n_samples: Number of samples to generate.
    :return: Feature matrix and label array.
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    return X, y

def main():
    # Generate data
    X, y = generate_data()

    # From-scratch implementation
    dt = DecisionTree(max_depth=3)
    dt.fit(X, y)
    accuracy_scratch = dt.score(X, y)

    # Scikit-learn implementation
    sklearn_dt = SklearnDecisionTreeClassifier(max_depth=3)
    sklearn_dt.fit(X, y)
    accuracy_sklearn = sklearn_dt.score(X, y)

    # Print results
    print(f"From-scratch implementation accuracy: {accuracy_scratch}")
    print(f"Scikit-learn implementation accuracy: {accuracy_sklearn}")

if __name__ == "__main__":
    main()
