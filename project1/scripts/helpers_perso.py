# -*- coding: utf-8 -*-
"""some personal helper functions for project 1."""
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, method_train, method_loss, *args):
    """
    run on given fold
    use this to do cross validation (run over every k-fold in main)
    return the loss of the given method.
    method must return weights and loss.
    args must be the correct arguments for the mehtod
    """
    test_idx = k_indices[k]
    train_idx = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_idx = train_idx.reshape(-1)
    y_test = y[test_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    x_train = x[train_idx]
    
    w, train_loss = method_train(y_train, x_train, *args)
    test_loss = method_loss(y_test, x_train, w)
    
    return w, train_loss, test_loss


def confusion_mat(tp, tn, fp, fn):
    """
    Inputs are the labels y and the predictions.
    Creates a confusion matrix (y-axes: actual class, x-class: predicted class)
    Size of 2x2.
    """
    conf_mtx = np.array([[tp, fn], [fp, tn]])
    return conf_mtx
    

def calc_rates(y, p):
    tn = sum((p == 0) & (p == y))  # prediction 0 and the same as the label -> true negative
    tp = sum((p == 1) & (p == y))  # prediction 1 and the same as the label -> true positive
    fn = sum((p == 0) & (p != y))  # prediction 0 and not the same as the label -> false negative
    fp = sum((p == 1) & (p != y))  # prediction 1 and not the same as the label -> false positive
    return tp, tn, fp, fn


def conf_matrix(tp, tn, fp, fn):
    conf_mtx = np.array([[tp, fn], [fp, tn]])
    return conf_mtx


def recall(tp, fn):
    tpr = tp / (tp + fn)
    return tpr


def precision(tp, fp):
    ppv = tp / (tp + fp)
    return ppv


def f_score(beta, recall, precision):
    """
    Two commonly used values for β are 2, which weighs recall higher than precision, and 0.5, which weighs recall lower than precision.
    :param beta:
    :param recall:
    :param precision:
    :return:
    """
    f_s = (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))
    return f_s
    
    
    
    

    
