# -*- coding: utf-8 -*-
"""ML Methods functions."""

import numpy as np

def mse(e):
    """Calculate mse for the given vector e.

    Args:
        e: error vector 

    Returns:
        mse value for the given vector.
    """
    return 1/2*np.mean(e**2)
    
def mae(e):
    """Calculate mae for the given vector e.

    Args:
        e: error vector 

    Returns:
        mae value for the given vector.
    """
    return np.mean(np.abs(e))

def compute_loss_mse(y, tx, w):
    """Compute the loss function using mse.

    Args:
        y: expected results
        tx : inputs
        w : weights

    Returns:
        mse value for the model.
    """
    e = y - tx.dot(w)
    return mse(e)
    
def compute_loss_mae(y, tx, w):
    """Compute the loss function using mae.

    Args:
        y: expected results
        tx : inputs
        w : weights

    Returns:
        msae value for the model.
    """
    e = y - tx.dot(w)
    return mae(e)    
    

    
    
    
def compute_GD(y, tx, gamma):
    """Compute the gradient descent
    
    Args:
        y: expected values
        tx: inputs
        gamma: step-size > 0
        
    Returns:
       gradient
    """
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(err)
    return grad
    
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    
    Args:
        y: 
        tx: 
        initial_w:
        max_iters:
        gamma:
        
    Returns:
       
    
    """
    w = initial_w
    for n in range(max_iters):
        grad = compute_GD(y, tx, w)
        w = w - gamma * grad
    loss = calculate_loss_mse(y, tx, w)
    return loss, w
    
    

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
    
    
    
    Args:
        y: 
        tx: 
        initial_w:
        max_iters:
        gamma:
        
    Returns:
       
    
    """
    w = initial_w
    data_size = len(y)
    
    for n in range(max_iters):
        # as batch_size and num_batch are both 1, a batch creator (given from the labs) will only given
        # back one sample (the FIRST sample), given the batch creator shuffles the dataset before it 
        # gives back the one sample, this is equal to just pick a random sample
        # --->>> this means to maximize the possibility to train with every sample the max iteration
        # should be a lot higher then the data_size
        i = np.random.randint(low=0, high=data_size, size=1)
        grad = compute_GD(y[i], tx[i], w)
        w = w - gamma * grad
    loss = calculate_loss_mse(y, tx, w)
    return loss, w
    
    
    
    
def least_squares(y, tx):
    """Compute the least squares solution.

    Args:
        y: 
        tx: 

    Returns:
        optimal weights and loss with normal equation.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss_mse(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
     """Ridge regression using normal equations.

    Args:
        y: expected results
        tx: inputs
        lambda_: regularization parameter

    Returns:
        optimal weights and loss with normal equation.
    """
    (N, p) = tx.shape
    lambda1 = 2 * N * lambda_
    a = tx.T.dot(tx) + lambda1 * np.eye(p)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss_mse(y, tx, w)
    return w, mse
    
    
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
     """Logistic regression using gradient descent or SGD
    
    Args:
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       
    
    """
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    
    
    
def reg_logistic_regression(y, tx, initial_w, max_iters, gamma):
     """Regularized Logistic regression using gradient descent or SGD
    
    Args:
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       
    
    """
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
