import sys
import os

import torch

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------

import numpy as np
from common.optimizer import SGD
from common.metrics import Accuracy
# from common_components.loss_functions import CrossEntropyLoss
from decision_trees import DecisionTree

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, criterion='gini', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                DecisionTree(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    criterion=criterion
                )
            )

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for tree in self.trees:
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return Accuracy().compute(y, y_pred)

# Testing and benchmarking
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import time

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate custom RandomForest
    custom_rf = RandomForest(n_estimators=100, max_depth=10, random_state=42)
    start_time = time.time()
    custom_rf.fit(X_train, y_train)
    custom_train_time = time.time() - start_time
    custom_accuracy = custom_rf.score(X_test, y_test)

    # Train and evaluate sklearn RandomForest
    sklearn_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    start_time = time.time()
    sklearn_rf.fit(X_train, y_train)
    sklearn_train_time = time.time() - start_time
    sklearn_accuracy = sklearn_rf.score(X_test, y_test)

    # Display results
    print(f"Custom RandomForest accuracy: {custom_accuracy:.4f}, training time: {custom_train_time:.4f}s")
    print(f"Sklearn RandomForest accuracy: {sklearn_accuracy:.4f}, training time: {sklearn_train_time:.4f}s")

    # Plot comparison
    labels = ['Custom RandomForest', 'Sklearn RandomForest']
    accuracy = [custom_accuracy, sklearn_accuracy]
    train_time = [custom_train_time, sklearn_train_time]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.bar(labels, accuracy, color='g')
    ax2.bar(labels, train_time, color='b', alpha=0.6)

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Training Time (s)', color='b')

    plt.title('Comparison of Custom RandomForest and Sklearn RandomForest')
    plt.show()
