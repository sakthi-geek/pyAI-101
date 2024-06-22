import sys
import os

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------
"""
decision_trees.py

This module implements the Decision Tree algorithm from scratch.
"""

import numpy as np
from collections import Counter

class Node:
    """
    Class representing a node in the decision tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    """
    Decision Tree classifier.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        """
        Initialize the DecisionTree.

        Args:
            max_depth (int): The maximum depth of the tree.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            max_features (str): The number of features to consider when looking for the best split.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
        """
        self.n_features = X.shape[1] if not self.max_features else min(self.max_features, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def predict(self, X):
        """
        Predict the class labels for the input features.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the grown tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or 
            n_samples < self.min_samples_split or 
            n_samples < self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """
        Find the best split for the data.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            feat_idxs (np.ndarray): The indices of the features to consider for the split.

        Returns:
            (int, float): The index of the best feature and the best threshold value.
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        """
        Calculate the information gain of a split.

        Args:
            y (np.ndarray): The target values.
            X_column (np.ndarray): The feature values.
            split_thresh (float): The threshold value for the split.

        Returns:
            float: The information gain of the split.
        """
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n, n_left, n_right = len(y), len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        """
        Split the data based on the threshold.

        Args:
            X_column (np.ndarray): The feature values.
            split_thresh (float): The threshold value for the split.

        Returns:
            (np.ndarray, np.ndarray): The indices of the left and right splits.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculate the entropy of the target values.

        Args:
            y (np.ndarray): The target values.

        Returns:
            float: The entropy of the target values.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Determine the most common label in the target values.

        Args:
            y (np.ndarray): The target values.

        Returns:
            int: The most common label.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction.

        Args:
            x (np.ndarray): The input feature values.
            node (Node): The current node in the tree.

        Returns:
            int: The predicted class label.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from pyAI.utils.metrics import Accuracy

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train custom DecisionTree
    dt = DecisionTree(max_depth=10)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # Evaluate model
    accuracy = Accuracy().compute(y_test, y_pred)
    print(f"DecisionTree Accuracy: {accuracy:.4f}")
