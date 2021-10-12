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
     # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************

    
    

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
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************    
    
    
    
    
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
        y: 
        tx: 
        lambda_:

    Returns:
        optimal weights and loss with normal equation.
    """
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    
    
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
     """Logistic regression using gradient descent or SGD
    
    Args:
        y: 
        tx: 
        initial_w:
        max_iters:
        gamma:
        
    Returns:
       
    
    """
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    
    
    
def reg_logistic_regression(y, tx, initial_w, max_iters, gamma):
     """Regularized Logistic regression using gradient descent or SGD
    
    Args:
        y: 
        tx: 
        initial_w:
        max_iters:
        gamma:
        
    Returns:
       
    
    """
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
