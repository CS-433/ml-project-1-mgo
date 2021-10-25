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


def cross_validation(y, x, k_indices, k, method_train, method_loss *args):
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
    test_loss = method_test(y_test, x_train, w)
    
    return w, train_loss, test_loss
    
    
    
    
    

    
