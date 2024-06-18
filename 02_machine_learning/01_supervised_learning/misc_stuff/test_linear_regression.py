import unittest
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing."""
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([6, 8, 10, 12])

    def test_fit_predict(self):
        """Test the custom LinearRegression model against scikit-learn."""
        model = LinearRegression(learning_rate=0.01, n_iterations=1000, normalize=True, regularization='l2', alpha=0.1)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        custom_mse = mean_squared_error(self.y, predictions)

        sklearn_lr = SklearnLinearRegression()
        sklearn_lr.fit(StandardScaler().fit_transform(self.X), self.y)
        sklearn_predictions = sklearn_lr.predict(StandardScaler().fit_transform(self.X))
        sklearn_mse = mean_squared_error(self.y, sklearn_predictions)

        self.assertAlmostEqual(custom_mse, sklearn_mse, delta=0.01)

    def test_regularization(self):
        """Test L1 and L2 regularization effects."""
        model_l1 = LinearRegression(regularization='l1', alpha=0.1)
        model_l1.fit(self.X, self.y)
        predictions_l1 = model_l1.predict(self.X)
        l1_mse = mean_squared_error(self.y, predictions_l1)

        model_l2 = LinearRegression(regularization='l2', alpha=0.1)
        model_l2.fit(self.X, self.y)
        predictions_l2 = model_l2.predict(self.X)
        l2_mse = mean_squared_error(self.y, predictions_l2)

        self.assertLessEqual(l1_mse, l2_mse)

    def test_edge_cases(self):
        """Test edge cases like zero variance features and small learning rates."""
        zero_variance_X = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        zero_variance_y = np.array([5, 5, 5, 5])

        model = LinearRegression()
        model.fit(zero_variance_X, zero_variance_y)
        predictions = model.predict(zero_variance_X)
        self.assertTrue(np.allclose(predictions, zero_variance_y, atol=1e-5))

        model_small_lr = LinearRegression(learning_rate=1e-6, n_iterations=1000)
        model_small_lr.fit(self.X, self.y)
        predictions_small_lr = model_small_lr.predict(self.X)
        small_lr_mse = mean_squared_error(self.y, predictions_small_lr)
        self.assertLessEqual(small_lr_mse, 1.0)

    def test_metrics(self):
        """Test various evaluation metrics."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        r2 = r2_score(self.y, predictions)
        mse = mean_squared_error(self.y, predictions)
        mae = mean_absolute_error(self.y, predictions)

        self.assertAlmostEqual(r2, 1.0, delta=0.01)
        self.assertAlmostEqual(mse, 0.0, delta=0.01)
        self.assertAlmostEqual(mae, 0.0, delta=0.01)

if __name__ == '__main__':
    unittest.main()



# tests/test_logistic_regression.py
import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from logistic_regression import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        """Set up sample data for testing."""
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [1, 1], [2, 2], [3, 3], [4, 4]])
        self.y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        self.X_multi = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [1, 1], [2, 2], [3, 3], [4, 4]])
        self.y_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1])

    def test_fit_predict(self):
        """Test the custom LogisticRegression model against scikit-learn."""
        model = LogisticRegression(learning_rate=0.01, n_iterations=1000, normalize=True, early_stopping=True, tolerance=1e-4)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        custom_accuracy = accuracy_score(self.y, predictions)

        sklearn_lr = SklearnLogisticRegression()
        sklearn_lr.fit(self.X, self.y)
        sklearn_predictions = sklearn_lr.predict(self.X)
        sklearn_accuracy = accuracy_score(self.y, sklearn_predictions)

        self.assertAlmostEqual(custom_accuracy, sklearn_accuracy, delta=0.01)

    def test_regularization(self):
        """Test L1 and L2 regularization effects."""
        model_l1 = LogisticRegression(learning_rate=0.01, regularization='l1', alpha=0.1)
        model_l1.fit(self.X, self.y)
        predictions_l1 = model_l1.predict(self.X)
        self.assertAlmostEqual(accuracy_score(self.y, predictions_l1), 1.0, delta=0.01)

        model_l2 = LogisticRegression(learning_rate=0.01, regularization='l2', alpha=0.1)
        model_l2.fit(self.X, self.y)
        predictions_l2 = model_l2.predict(self.X)
        self.assertAlmostEqual(accuracy_score(self.y, predictions_l2), 1.0, delta=0.01)

    def test_edge_cases(self):
        """Test edge cases like zero variance features and small learning rates."""
        zero_variance_X = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        zero_variance_y = np.array([1, 1, 1, 1])

        model = LogisticRegression()
        model.fit(zero_variance_X, zero_variance_y)
        predictions = model.predict(zero_variance_X)
        self.assertTrue(np.allclose(predictions, zero_vari

ance_y, atol=1e-5))

python

    model_small_lr = LogisticRegression(learning_rate=1e-6, n_iterations=1000)
    model_small_lr.fit(self.X, self.y)
    predictions_small_lr = model_small_lr.predict(self.X)
    small_lr_accuracy = accuracy_score(self.y, predictions_small_lr)
    self.assertLessEqual(small_lr_accuracy, 1.0)

def test_metrics(self):
    """Test various evaluation metrics."""
    model = LogisticRegression()
    model.fit(self.X, self.y)
    predictions = model.predict(self.X)
    accuracy = accuracy_score(self.y, predictions)
    log_loss_value = log_loss(self.y, model.predict_proba(self.X))

    self.assertAlmostEqual(accuracy, 1.0, delta=0.01)
    self.assertAlmostEqual(log_loss_value, 0.0, delta=0.01)

def test_multiclass_support(self):
    """Test for multiclass support."""
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000, multi_class='multinomial')
    model.fit(self.X_multi, self.y_multi)
    predictions = model.predict(self.X_multi)
    custom_accuracy = accuracy_score(self.y_multi, predictions)

    sklearn_lr = SklearnLogisticRegression(multi_class='multinomial')
    sklearn_lr.fit(self.X_multi, self.y_multi)
    sklearn_predictions = sklearn_lr.predict(self.X_multi)
    sklearn_accuracy = accuracy_score(self.y_multi, sklearn_predictions)

    self.assertAlmostEqual(custom_accuracy, sklearn_accuracy, delta=0.01)

if name == 'main':
unittest.main()