import sys
import os

import torch

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _gini(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = len(y[y == cls]) / len(y)
            gini -= p ** 2
        return gini

    def _entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p = len(y[y == cls]) / len(y)
            entropy -= p * np.log2(p)
        return entropy

    def _criterion_function(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError("Criterion should be either 'gini' or 'entropy'")

    def _split(self, X, y, index, value):
        left_mask = X[:, index] < value
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y):
        best_index, best_value, best_score, best_splits = None, None, float('inf'), None
        for index in range(X.shape[1]):
            for value in np.unique(X[:, index]):
                splits = self._split(X, y, index, value)
                left_y, right_y = splits[1], splits[3]
                score = self._criterion_function(left_y) * len(left_y) + self._criterion_function(right_y) * len(right_y)
                if score < best_score:
                    best_index, best_value, best_score, best_splits = index, value, score, splits
        return best_index, best_value, best_splits

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth == self.max_depth:
            return np.bincount(y).argmax()
        index, value, splits = self._best_split(X, y)
        left_tree = self._build_tree(splits[0], splits[1], depth + 1)
        right_tree = self._build_tree(splits[2], splits[3], depth + 1)
        return (index, value, left_tree, right_tree)

    def _predict(self, inputs, tree):
        if not isinstance(tree, tuple):
            return tree
        index, value, left_tree, right_tree = tree
        if inputs[index] < value:
            return self._predict(inputs, left_tree)
        else:
            return self._predict(inputs, right_tree)


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='mse'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _criterion_function(self, y):
        if self.criterion == 'mse':
            return self._mse(y)
        else:
            raise ValueError("Criterion should be 'mse'")

    def _split(self, X, y, index, value):
        left_mask = X[:, index] < value
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y):
        best_index, best_value, best_score, best_splits = None, None, float('inf'), None
        for index in range(X.shape[1]):
            for value in np.unique(X[:, index]):
                X_left, y_left, X_right, y_right = self._split(X, y, index, value)
                score = (self._criterion_function(y_left) * len(y_left) + 
                         self._criterion_function(y_right) * len(y_right))
                if score < best_score:
                    best_index, best_value, best_score = index, value, score
                    best_splits = (X_left, y_left, X_right, y_right)
        return best_index, best_value, best_splits

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)
        index, value, splits = self._best_split(X, y)
        if splits is None:
            return np.mean(y)
        left_tree = self._build_tree(splits[0], splits[1], depth + 1)
        right_tree = self._build_tree(splits[2], splits[3], depth + 1)
        return (index, value, left_tree, right_tree)

    def _predict(self, inputs, tree):
        if not isinstance(tree, tuple):
            return tree
        index, value, left_tree, right_tree = tree
        if inputs[index] < value:
            return self._predict(inputs, left_tree)
        else:
            return self._predict(inputs, right_tree)

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor as SklearnDTR
    import matplotlib.pyplot as plt
    import time

    class MeanSquaredError:
        def compute(self, y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

    class RSquared:
        def compute(self, y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)

    # Load dataset
    data = load_boston()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train custom Decision Tree Regressor
    dtr = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
    start_time = time.time()
    dtr.fit(X_train, y_train)
    end_time = time.time()
    print(f"Custom Decision Tree Regressor training time: {end_time - start_time} seconds")

    # Predictions
    y_pred_train = dtr.predict(X_train)
    y_pred_test = dtr.predict(X_test)

    # Evaluate custom model
    mse = MeanSquaredError()
    rsquared = RSquared()
    print(f"Custom Decision Tree Regressor MSE (Train): {mse.compute(y_train, y_pred_train)}")
    print(f"Custom Decision Tree Regressor MSE (Test): {mse.compute(y_test, y_pred_test)}")
    print(f"Custom Decision Tree Regressor R-squared (Train): {rsquared.compute(y_train, y_pred_train)}")
    print(f"Custom Decision Tree Regressor R-squared (Test): {rsquared.compute(y_test, y_pred_test)}")

    # Train sklearn Decision Tree Regressor
    sklearn_dtr = SklearnDTR(max_depth=5, min_samples_split=10, random_state=42)
    start_time = time.time()
    sklearn_dtr.fit(X_train, y_train)
    end_time = time.time()
    print(f"Sklearn Decision Tree Regressor training time: {end_time - start_time} seconds")

    # Predictions
    sklearn_y_pred_train = sklearn_dtr.predict(X_train)
    sklearn_y_pred_test = sklearn_dtr.predict(X_test)

    # Evaluate sklearn model
    print(f"Sklearn Decision Tree Regressor MSE (Train): {mse.compute(y_train, sklearn_y_pred_train)}")
    print(f"Sklearn Decision Tree Regressor MSE (Test): {mse.compute(y_test, sklearn_y_pred_test)}")
    print(f"Sklearn Decision Tree Regressor R-squared (Train): {rsquared.compute(y_train, sklearn_y_pred_train)}")
    print(f"Sklearn Decision Tree Regressor R-squared (Test): {rsquared.compute(y_test, sklearn_y_pred_test)}")

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred_test, label='Custom DTR Predictions')
    plt.plot(sklearn_y_pred_test, label='Sklearn DTR Predictions')
    plt.legend()
    plt.title('Decision Tree Regressor Comparison')
    plt.show()