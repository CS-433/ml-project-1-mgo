# -*- coding: utf-8 -*-
"""some personal helper functions for project 1."""
import numpy as np
from tqdm import tqdm
from implementations import *
from proj1_helpers import *
import matplotlib.pyplot as plt


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None or std_x is None:
        mean_x = np.nanmean(x, axis=0)
        x = x - mean_x
        std_x = np.nanstd(x, axis=0)
        x = x / std_x
    else:
        x = x - mean_x
        x = x / std_x
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio. Into train and validation"""
    # same seed
    np.random.seed(seed)
    # generate random indices
    no_ex = y.shape[0]
    indices = np.random.permutation(no_ex)
    idx_split = int(np.floor(ratio * no_ex))
    idx_train = indices[: idx_split]
    idx_val = indices[idx_split:]
    # create split
    x_train = x[idx_train]
    x_val = x[idx_val]
    y_train = y[idx_train]
    y_val = y[idx_val]
    return x_train, x_val, y_train, y_val


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


def buildpoly(x, degree):
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly



def vis_conf_mtx(conf_mtx):
    """
    Visualize the confusion matrix
    :param conf_mtx:
    :return:
    """

    fig, ax = plt.subplots()
    im = ax.imshow(conf_mtx)
    s = conf_mtx.shape[0]
    # We want to show all ticks...
    ax.set_xticks(np.arange(s))
    ax.set_yticks(np.arange(s))
    # ... and label them with the respective list entries
    ax.set_xticklabels([1, 0])
    ax.set_yticklabels([1, 0])

    # Loop over data dimensions and create text annotations.
    for i in range(s):
        for j in range(s):
            text = ax.text(j, i, conf_mtx[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Predictions")
    fig.tight_layout()
    plt.show()

def grid_search(y_val, tX_val, val_iter, method_train, method_loss, y_train, tX_train, max_iters, *args):
    """
    Degree
    Gamma
    Lambda
    """
    degrees, gammas, lambdas_= args
    
    for degree in degrees:
        for gamma in gammas:
            for lambda_ in lambdas_:
                print("In: Degree {}, Gamma {}, Lambda {}".format(degree, gamma, lambda_))
                if degree != 1:
                    tX_val_p = buildpoly(tX_val, degree)
                    tX_train_p = buildpoly(tX_train, degree)
                else:
                    tX_val_p, tX_train_p = tX_val, tX_train
                _, _ = trainer_val(y_val, tX_val_p, method_train, method_loss, y_train, tX_train_p, val_iter, max_iters, gamma, lambda_)
                print("-"*100)
                
    


def trainer_val(y_val, tX_val, method_train, method_loss, y, tx, val_iter = 1, max_iters = 1, gamma = 0.0000001, lambda_ = None):
    """
    
    """
    iter_count, best_val, best_weight = 0, np.inf, 0
    train_losses, val_losses, weights = [], [], []
    w=np.zeros(tx.shape[1])
    runs = max_iters // val_iter
    if runs < (max_iters / val_iter):
        # run validation after the last training run
        runs += 1
    for run in range(runs):
        max_iters -= val_iter
        if max_iters >= 0:
            # run training till validation
            if gamma is None and lambda_ is None:
                # least squares
                loss, w = method_train(y, tx)
            elif gamma is None:
                # ridge regression
                loss, w = method_train(y, tx, lambda_)
            elif lambda_ is None:
                # least squares GD
                # least squares SGD
                # logistic regression
                loss, w = method_train(y, tx, w, val_iter, gamma)
            else:
                # reg logistic regression
                loss, w = method_train(y, tx, lambda_ , w, val_iter, gamma)
            iter_count += val_iter
        else: 
            # run the rest iterations
            max_iters += val_iter
            if gamma is None and lambda_ is None:
                # least squares
                loss, w = method_train(y, tx)
            elif gamma is None:
                # ridge regression
                loss, w = method_train(y, tx, lambda_)
            elif lambda_ is None:
                # least squares GD
                # least squares SGD
                # logistic regression
                loss, w = method_train(y, tx, w, val_iter, gamma)
            else:
                # reg logistic regression
                loss, w = method_train(y, tx, lambda_ , w, val_iter, gamma)
            iter_count += max_iters
        # validate
        val_loss = method_loss(y_val, tX_val, w)
        train_losses.append(loss)
        val_losses.append(val_loss)
        weights.append(w)
        if best_val > val_loss:
            best_val = val_loss
            best_weight = w
            best_iter = iter_count
        print("In run: {}, trained. Train loss: {}, Val loss: {}.".format(run, loss, val_loss))
    print("Best weights from iteration {}".format(best_iter))
    # Estimating the predictions on the validation set
    pred_val = predict_labels(best_weight, tX_val)
    # Confusion matrix
    tp, tn, fp, fn = calc_rates(y_val, pred_val)
    vis_conf_mtx(conf_matrix(tp, tn, fp, fn))
    # Recall, Precision, F2-Score, Accruacy
    f_score(recall(tp, fn), precision(tp, fp))
    accruacy(tp, tn, fp, fn)
    return loss, best_weight


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
    print("TP: {}, TN: {}, FP: {}, FN: {}".format(tp, tn, fp, fn))
    return tp, tn, fp, fn


def conf_matrix(tp, tn, fp, fn):
    conf_mtx = np.array([[tp, fn], [fp, tn]])
    return conf_mtx


def recall(tp, fn):
    tpr = tp / (tp + fn)
    print("Recall: {}".format(tpr))
    return tpr


def precision(tp, fp):
    ppv = tp / (tp + fp)
    print("Precision: {}".format(ppv))
    return ppv


def accruacy(tp, tn, fp, fn):
    ppv = (tp + tn) / (tp + tn + fp + fn)
    print("Accruacy: {}".format(ppv))
    return ppv


def f_score(recall, precision, beta=2):
    """
    Two commonly used values for β are 2, which weighs recall higher than precision, and 0.5, which weighs recall lower than precision.
    :param beta:
    :param recall:
    :param precision:
    :return:
    """
    f_s = (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))
    print("F_{} score: {}".format(beta, f_s))
    return f_s
    
    
    
 

    
