"""
evaluation_metrics.py

This module provides various evaluation metrics commonly used in machine learning and deep learning models.
Each metric is implemented from scratch to help learners understand the underlying mechanics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EvaluationMetric:
    """
    Base class for all evaluation metrics.
    """
    def compute(self, y_true, y_pred):
        raise NotImplementedError("Compute method not implemented!")

#-------------------------------------------------------------------------------------------

class MeanSquaredError(EvaluationMetric):
    """
    Mean Squared Error (MSE) evaluation metric.
    
    MSE = 1/N * sum((y - y_hat)^2)
    """
    def compute(self, y_true, y_pred):
        """
        Compute the MSE.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed MSE value.
        """
        return np.mean((y_true - y_pred) ** 2)

#-------------------------------------------------------------------------------------------

class MeanAbsoluteError(EvaluationMetric):
    """
    Mean Absolute Error (MAE) evaluation metric.
    
    MAE = 1/N * sum(|y - y_hat|)
    """
    def compute(self, y_true, y_pred):
        """
        Compute the MAE.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed MAE value.
        """
        return np.mean(np.abs(y_true - y_pred))

#-------------------------------------------------------------------------------------------

class RootMeanSquaredError(EvaluationMetric):
    """
    Root Mean Squared Error (RMSE) evaluation metric.
    
    RMSE = sqrt(1/N * sum((y - y_hat)^2))
    """
    def compute(self, y_true, y_pred):
        """
        Compute the RMSE.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed RMSE value.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

#-------------------------------------------------------------------------------------------

class Accuracy(EvaluationMetric):
    """
    Accuracy evaluation metric.
    
    Accuracy = (number of correct predictions) / (total number of predictions)
    """
    def compute(self, y_true, y_pred):
        """
        Compute the accuracy.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed accuracy value.
        """
        return np.mean(y_true == y_pred)

#-------------------------------------------------------------------------------------------

class Precision(EvaluationMetric):
    """
    Precision evaluation metric.
    
    Precision = (true positives) / (true positives + false positives)
    """
    def compute(self, y_true, y_pred):
        """
        Compute the precision.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed precision value.
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / (predicted_positives + 1e-15)

#-------------------------------------------------------------------------------------------

class Recall(EvaluationMetric):
    """
    Recall evaluation metric.
    
    Recall = (true positives) / (true positives + false negatives)
    """
    def compute(self, y_true, y_pred):
        """
        Compute the recall.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed recall value.
        """
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / (actual_positives + 1e-15)

#-------------------------------------------------------------------------------------------

class F1Score(EvaluationMetric):
    """
    F1 Score evaluation metric.
    
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    def compute(self, y_true, y_pred):
        """
        Compute the F1 score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed F1 score value.
        """
        precision = Precision().compute(y_true, y_pred)
        recall = Recall().compute(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall + 1e-15)

#-------------------------------------------------------------------------------------------

class MeanIoU(EvaluationMetric):
    """
    Mean Intersection over Union (IoU) evaluation metric.
    
    IoU = (Intersection) / (Union)
    Mean IoU = 1/N * sum(IoU)
    """
    def compute(self, y_true, y_pred):
        """
        Compute the mean IoU.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed mean IoU value.
        """
        intersection = np.sum((y_true == 1) & (y_pred == 1))
        union = np.sum((y_true == 1) | (y_pred == 1))
        return intersection / (union + 1e-15)
    
#-------------------------------------------------------------------------------------------

class confusion_matrix(EvaluationMetric):
    """
    Confusion Matrix evaluation metric.
    
    """
    def compute(self, y_true, y_pred):
        """
        Compute the confusion matrix.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed confusion matrix value.
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return TP, FP, TN, FN
    
    def display(self, y_true, y_pred):
        """
        Display the confusion matrix.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            dataframe: Computed confusion matrix value.
        """
        TP, FP, TN, FN = confusion_matrix().compute(y_true, y_pred)
        # display dataframe similar to sklearn confusion matrix
        return pd.DataFrame([[TN, FP], [FN, TP]], columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
         
#-------------------------------------------------------------------------------------------

class ClassificationReport(EvaluationMetric):
    """
    Classification Report evaluation metric.
    
    """
    def compute(self, y_true, y_pred):
        """
        Compute the classification report.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed classification report value.
        """
        TP, FP, TN, FN = confusion_matrix().compute(y_true, y_pred)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1
    
    def display(self, y_true, y_pred):
        """
        Display the classification report.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            dataframe: Computed classification report value.
        """
        precision, recall, f1 = ClassificationReport().compute(y_true, y_pred)
        return pd.DataFrame([[precision, recall, f1]], columns=['Precision', 'Recall', 'F1 Score'], index=['Metrics'])
    
#-------------------------------------------------------------------------------------------

class ROCcurve(EvaluationMetric):
    """
    ROC curve and ROC AUC score evaluation metrics.
    
    """
    def compute_roc_curve(self, y_true, y_pred):
        """
        Compute the ROC curve.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            tuple: Computed false positive rate and true positive rate values.
        """

        # sort the true values and predicted values
        y_true_sorted = np.asarray([x for _, x in sorted(zip(y_pred, y_true))])

        # calculate the cumulative sum of true values
        cum_true = np.cumsum(y_true_sorted)

        # calculate the cumulative sum of false values
        cum_false = np.cumsum(1 - y_true_sorted)

        # calculate the total number of true values
        total_true = np.sum(y_true_sorted)

        # calculate the total number of false values
        total_false = len(y_true_sorted) - total_true

        # calculate the true positive rate
        tpr = cum_true / total_true

        # calculate the false positive rate
        fpr = cum_false / total_false

        return fpr, tpr
    
    def compute_roc_auc_score(self, y_true, y_pred):
        """
        Compute the ROC AUC score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed ROC AUC score value.
        """
        fpr, tpr = ROCcurve().compute_roc_curve(y_true, y_pred)

        # calculate the ROC AUC score
        roc_auc_score = np.trapz(tpr, fpr)

        return roc_auc_score
    
    def plot_roc_curve(self, y_true, y_pred):
        """
        Plot the ROC curve.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            plot: ROC curve plot.
        """
        fpr, tpr = ROCcurve().compute_roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

#-------------------------------------------------------------------------------------------

class GiniCoefficient(EvaluationMetric):
    """
    Gini Coefficient evaluation metric.
    
    """
    def compute(self, y_true, y_pred):
        """
        Compute the Gini Coefficient.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed Gini Coefficient value.
        """
        
        # calculate the ROC AUC score
        roc_auc_score = ROCcurve().compute_roc_auc_score(y_true, y_pred)

        # calculate the Gini Coefficient
        gini_coefficient = 2 * roc_auc_score - 1

        return gini_coefficient

#-------------------------------------------------------------------------------------------

class LogLoss(EvaluationMetric):
    """
    Log Loss evaluation metric.
    
    """
    def compute(self, y_true, y_pred):
        """
        Compute the Log Loss.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed Log Loss value.
        """

        # clip the predicted values to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # calculate the log loss
        log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return log_loss
    
#-------------------------------------------------------------------------------------------

class MeanSquaredLogError(EvaluationMetric):
    """
    Mean Squared Log Error evaluation metric.
    
    """
    def compute(self, y_true, y_pred):
        """
        Compute the Mean Squared Log Error.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed Mean Squared Log Error value.
        """

        # calculate the log of the true values
        log_y_true = np.log1p(y_true)

        # calculate the log of the predicted values
        log_y_pred = np.log1p(y_pred)

        # calculate the mean squared log error
        msle = np.mean((log_y_true - log_y_pred) ** 2)

        return msle
    
#-------------------------------------------------------------------------------------------

class RSquared(EvaluationMetric):
    """
    R-Squared evaluation metric.
    
    """
    def compute(self, y_true, y_pred):
        """
        Compute the R-Squared.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Computed R-Squared value.
        """

        #calculate the mean of the true values
        y_mean = np.mean(y_true)

        #calculate the total sum of squares
        ss_total = np.sum((y_true - y_mean) ** 2)

        #calculate the residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)

        #calculate the R-Squared
        r_squared = 1 - (ss_res / ss_total)

        return r_squared


#===========================================================================================================

# Example usage:
if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    mse = MeanSquaredError()
    print("MSE:", mse.compute(y_true, y_pred))

    mae = MeanAbsoluteError()
    print("MAE:", mae.compute(y_true, y_pred))

    rmse = RootMeanSquaredError()
    print("RMSE:", rmse.compute(y_true, y_pred))

    accuracy = Accuracy()
    print("Accuracy:", accuracy.compute(y_true, y_pred))

    precision = Precision()
    print("Precision:", precision.compute(y_true, y_pred))

    recall = Recall()
    print("Recall:", recall.compute(y_true, y_pred))

    f1 = F1Score()
    print("F1 Score:", f1.compute(y_true, y_pred))

    mean_iou = MeanIoU()
    print("Mean IoU:", mean_iou.compute(y_true, y_pred))

    confusion = confusion_matrix()
    print("Confusion Matrix:", confusion.compute(y_true, y_pred))

    classification_report = ClassificationReport()
    print("Classification Report:", classification_report.compute(y_true, y_pred))

    roc_curve = ROCcurve()
    print("ROC AUC Score:", roc_curve.compute_roc_auc_score(y_true, y_pred))

    gini_coefficient = GiniCoefficient()
    print("Gini Coefficient:", gini_coefficient.compute(y_true, y_pred))

    log_loss = LogLoss()
    print("Log Loss:", log_loss.compute(y_true, y_pred))

    msle = MeanSquaredLogError()
    print("Mean Squared Log Error:", msle.compute(y_true, y_pred))

    r_squared = RSquared()
    print("R-Squared:", r_squared.compute(y_true, y_pred))

    roc_curve.plot_roc_curve(y_true, y_pred)

#-------------------------------------------------------------------------------------------