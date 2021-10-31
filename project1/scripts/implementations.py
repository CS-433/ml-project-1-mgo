# -*- coding: utf-8 -*-
"""ML Methods functions."""

import numpy as np
from tqdm import tqdm

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
    
def compute_GD(y, tx, w):
    """Compute the gradient descent
    
    Args:
        y: expected values
        tx: inputs
        w: weights
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
    for n in tqdm(range(max_iters), desc="In step"):
        grad = compute_GD(y, tx, w)
        w = w - gamma * grad
    loss = compute_loss_mse(y, tx, w)
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
    
    for n in tqdm(range(max_iters)):
        # as batch_size and num_batch are both 1, a batch creator (given from the labs) will only given
        # back one sample (the FIRST sample), given the batch creator shuffles the dataset before it 
        # gives back the one sample, this is equal to just pick a random sample
        # --->>> this means to maximize the possibility to train with every sample the max iteration
        # should be a lot higher then the data_size
        i = np.random.randint(low=0, high=data_size, size=1)
        grad = compute_GD(y[i], tx[i], w)
        w = w - gamma * grad
    loss = compute_loss_mse(y, tx, w)
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
    return mse, w


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
    return mse, w


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1 / (1 + np.exp(-t))


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = (tx.T).dot(pred - y)
    return grad


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    return - (y.T).dot(np.log(pred)) - (1 - y).T.dot(np.log(1 - pred))


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
    w = initial_w
    for iter in range(max_iters):
        """
        Do one step of gradient descent using logistic regression.
        Return the loss and the updated w.
        """
        grad = calculate_gradient(y, tx, w)
        w -= gamma * grad
    loss = calculate_loss(y, tx, w)
    return loss, w
    
    
    
    
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Regularized Logistic regression using gradient descent or SGD
    
    Args:
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       
    
    """
    w = initial_w
    for iter in range(max_iters):
        """return the loss and gradient."""
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        """
        Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w.
        """
        w -= gamma * gradient
    loss = calculate_loss(y, tx, w) + lambda_ * np.linalg.norm(w)**2
    return loss, w
    

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = sigmoid(tx.dot(w))
    S = pred * (1 - pred)
    return (S * (tx.T))@tx


def learning_by_newton_method(y, tx, initial_w, max_iters, gamma):
    """Newton's method
    
    Args:
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       
    
    """
    w = initial_w
    for iter in range(max_iters):
        """
        Do one step of Newton's method.
        Return the loss and updated w.
        """
        gradient = calculate_gradient(y, tx, w)
        hessian = calculate_hessian(y, tx, w)
        w -= gamma * np.linalg.solve(hessian, gradient)
    loss = calculate_loss(y, tx, w)
    return loss, w
    
    
