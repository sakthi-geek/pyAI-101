import sys
import os

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
print(project_root_dir)

# Add the root directory of the project to the Python path
sys.path.append(project_root_dir)
#-------------------------------------------------------------------------------------------

import numpy as np
from decision_trees import DecisionTreeRegressor
from common_components.evaluation_metrics import MeanSquaredError, RSquared
from common_components.optimizers import SGD
import matplotlib.pyplot as plt

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.f0 = None

    def fit(self, X, y):
        self.f0 = np.mean(y)
        residuals = y - self.f0

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return RSquared().compute(y, y_pred)


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
    import time

    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train custom Gradient Boosting Regressor
    gb = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
    start_time = time.time()
    gb.fit(X_train, y_train)
    end_time = time.time()
    print(f"Custom Gradient Boosting Regressor training time: {end_time - start_time} seconds")

    # Predictions
    y_pred_train = gb.predict(X_train)
    y_pred_test = gb.predict(X_test)

    # Evaluate custom model
    mse = MeanSquaredError()
    rsquared = RSquared()
    print(f"Custom Gradient Boosting Regressor MSE (Train): {mse.compute(y_train, y_pred_train)}")
    print(f"Custom Gradient Boosting Regressor MSE (Test): {mse.compute(y_test, y_pred_test)}")
    print(f"Custom Gradient Boosting Regressor R-squared (Train): {rsquared.compute(y_train, y_pred_train)}")
    print(f"Custom Gradient Boosting Regressor R-squared (Test): {rsquared.compute(y_test, y_pred_test)}")

    # Train sklearn Gradient Boosting Regressor
    sklearn_gb = SklearnGBR(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42)
    start_time = time.time()
    sklearn_gb.fit(X_train, y_train)
    end_time = time.time()
    print(f"Sklearn Gradient Boosting Regressor training time: {end_time - start_time} seconds")

    # Predictions
    sklearn_y_pred_train = sklearn_gb.predict(X_train)
    sklearn_y_pred_test = sklearn_gb.predict(X_test)

    # Evaluate sklearn model
    print(f"Sklearn Gradient Boosting Regressor MSE (Train): {mse.compute(y_train, sklearn_y_pred_train)}")
    print(f"Sklearn Gradient Boosting Regressor MSE (Test): {mse.compute(y_test, sklearn_y_pred_test)}")
    print(f"Sklearn Gradient Boosting Regressor R-squared (Train): {rsquared.compute(y_train, sklearn_y_pred_train)}")
    print(f"Sklearn Gradient Boosting Regressor R-squared (Test): {rsquared.compute(y_test, sklearn_y_pred_test)}")

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred_test, label='Custom GB Predictions')
    plt.plot(sklearn_y_pred_test, label='Sklearn GB Predictions')
    plt.legend()
    plt.title('Gradient Boosting Regressor Comparison')
    plt.show()