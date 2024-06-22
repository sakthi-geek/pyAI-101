import numpy as np

class Metric:
    def reset(self):
        raise NotImplementedError

    def update(self, preds, target):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

#-----------------------------------------------------------------------

class MeanSquaredError(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_squared_error = 0
        self.total = 0

    def update(self, preds, target):
        self.sum_squared_error += np.sum((preds - target) ** 2)
        self.total += target.size

    def compute(self):
        return self.sum_squared_error / self.total

class MeanAbsoluteError(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_absolute_error = 0
        self.total = 0

    def update(self, preds, target):
        self.sum_absolute_error += np.sum(np.abs(preds - target))
        self.total += target.size

    def compute(self):
        return self.sum_absolute_error / self.total

class RootMeanSquaredError(Metric):
    def __init__(self):
        self.mse = MeanSquaredError()

    def reset(self):
        self.mse.reset()

    def update(self, preds, target):
        self.mse.update(preds, target)

    def compute(self):
        return np.sqrt(self.mse.compute())

class Accuracy(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, preds, target):
        preds = np.argmax(preds, axis=1)
        self.correct += np.sum(preds == target)
        self.total += target.size

    def compute(self):
        return self.correct / self.total

class Precision(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.true_positives = 0
        self.predicted_positives = 0

    def update(self, preds, target):
        preds = np.argmax(preds, axis=1)
        self.true_positives += np.sum((preds == 1) & (target == 1))
        self.predicted_positives += np.sum(preds == 1)

    def compute(self):
        return self.true_positives / self.predicted_positives if self.predicted_positives > 0 else 0

class Recall(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.true_positives = 0
        self.actual_positives = 0

    def update(self, preds, target):
        preds = np.argmax(preds, axis=1)
        self.true_positives += np.sum((preds == 1) & (target == 1))
        self.actual_positives += np.sum(target == 1)

    def compute(self):
        return self.true_positives / self.actual_positives if self.actual_positives > 0 else 0

class F1Score(Metric):
    def __init__(self):
        self.precision = Precision()
        self.recall = Recall()

    def reset(self):
        self.precision.reset()
        self.recall.reset()

    def update(self, preds, target):
        self.precision.update(preds, target)
        self.recall.update(preds, target)

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

class MeanIoU(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

    def update(self, preds, target):
        preds = np.argmax(preds, axis=1)
        for cls in range(self.num_classes):
            self.intersection[cls] += np.sum((preds == cls) & (target == cls))
            self.union[cls] += np.sum((preds == cls) | (target == cls))

    def compute(self):
        iou = self.intersection / (self.union + 1e-10)  # Avoid division by zero
        return np.mean(iou)

class ConfusionMatrix(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, preds, target):
        preds = np.argmax(preds, axis=1)
        for t, p in zip(target, preds):
            self.conf_matrix[t, p] += 1

    def compute(self):
        return self.conf_matrix

class ClassificationReport(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.conf_matrix = ConfusionMatrix(num_classes)

    def reset(self):
        self.conf_matrix.reset()

    def update(self, preds, target):
        self.conf_matrix.update(preds, target)

    def compute(self):
        conf_matrix = self.conf_matrix.compute()
        report = {}
        for cls in range(self.num_classes):
            tp = conf_matrix[cls, cls]
            fp = np.sum(conf_matrix[:, cls]) - tp
            fn = np.sum(conf_matrix[cls, :]) - tp
            tn = np.sum(conf_matrix) - (tp + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            report[cls] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': tp + fn}
        return report

class ROCCurve(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, preds, target):
        self.preds.append(preds)
        self.targets.append(target)

    def compute(self):
        preds = np.concatenate(self.preds)
        targets = np.concatenate(self.targets)
        fpr, tpr, thresholds = [], [], []
        thresholds_uniq = np.unique(preds)
        thresholds_uniq = np.append(thresholds_uniq, thresholds_uniq[-1] + 1)  # To include max value
        for threshold in thresholds_uniq:
            tp = np.sum((preds >= threshold) & (targets == 1))
            fp = np.sum((preds >= threshold) & (targets == 0))
            fn = np.sum((preds < threshold) & (targets == 1))
            tn = np.sum((preds < threshold) & (targets == 0))
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))
            thresholds.append(threshold)
        return np.array(fpr), np.array(tpr), np.array(thresholds)

class ROCAUC(Metric):
    def __init__(self):
        self.roc_curve = ROCCurve()

    def reset(self):
        self.roc_curve.reset()

    def update(self, preds, target):
        self.roc_curve.update(preds, target)

    def compute(self):
        fpr, tpr, _ = self.roc_curve.compute()
        return np.trapz(tpr, fpr)

class GiniCoefficient(Metric):
    def __init__(self):
        self.roc_auc = ROCAUC()

    def reset(self):
        self.roc_auc.reset()

    def update(self, preds, target):
        self.roc_auc.update(preds, target)

    def compute(self):
        auc = self.roc_auc.compute()
        return 2 * auc - 1

class LogLoss(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_log_loss = 0
        self.total = 0

    def update(self, preds, target):
        preds = np.clip(preds, 1e-10, 1 - 1e-10)  # Avoid log(0)
        self.sum_log_loss += -np.sum(target * np.log(preds) + (1 - target) * np.log(1 - preds))
        self.total += target.size

    def compute(self):
        return self.sum_log_loss / self.total

class MeanSquaredLogError(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_squared_log_error = 0
        self.total = 0

    def update(self, preds, target):
        self.sum_squared_log_error += np.sum((np.log1p(preds) - np.log1p(target)) ** 2)
        self.total += target.size

    def compute(self):
        return self.sum_squared_log_error / self.total

class RSquared(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_sum_of_squares = 0
        self.residual_sum_of_squares = 0

    def update(self, preds, target):
        self.total_sum_of_squares += np.sum((target - np.mean(target)) ** 2)
        self.residual_sum_of_squares += np.sum((target - preds) ** 2)

    def compute(self):
        return 1 - (self.residual_sum_of_squares / self.total_sum_of_squares)
