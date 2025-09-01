import numpy as np


def accuracy(predicted_labels, true_labels):
    return np.sum(predicted_labels == true_labels) / len(true_labels)


def to_binary(labels):
    return np.where(labels <= 0, 0, 1)


def confusion_matrix(predicted_labels, true_labels):
    predicted_labels, true_labels = to_binary(predicted_labels), to_binary(true_labels)
    tp = np.sum((predicted_labels == 1) & (true_labels == 1))           # true positives
    tn = np.sum((predicted_labels == 0) & (true_labels == 0))           # true negatives
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))           # false positives
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))           # false negatives
    return np.array([[tn, fp],
                     [fn, tp]])


def precision(predicted_labels, true_labels):
    cm = confusion_matrix(predicted_labels, true_labels)
    tp, fp = cm[1, 1], cm[0, 1]
    return tp / (tp + fp) if (tp + fp) != 0 else 0.0


def recall(predicted_labels, true_labels):
    cm = confusion_matrix(predicted_labels, true_labels)
    tp, fn = cm[1, 1], cm[1, 0]
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


def f1_score(predicted_labels, true_labels):
    p = precision(predicted_labels, true_labels)
    r = recall(predicted_labels, true_labels)
    return 2 * (p * r) / (p + r) if (p + r) != 0 else 0.0
