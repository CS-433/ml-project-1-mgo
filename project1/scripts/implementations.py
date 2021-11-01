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

def rmse(e):
    return np.sqrt(mse(e))

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

def compute_loss_rmse(y, tx, w):
    """Compute the loss function using rmse.

    Args:
        y: expected results
        tx : inputs
        w : weights

    Returns:
        mse value for the model.
    """
    e = y - tx.dot(w)
    return rmse(e)
    
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
                
    Returns:
       gradient
    """
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(err)
    return grad
    
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    
    Args:
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       loss: last loss
       w: last weights
           
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
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       loss: last loss
       w: last weights
    
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
        y: expected results
        tx: inputs
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
    #print("pred", pred)
    grad = (tx.T).dot(pred - y)
    #print("grad", grad)
    return grad


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    pred[pred == 1] -= np.finfo(float).eps
    pred[pred == 0] += np.finfo(float).eps
    #print("pred2", pred)
    loss = - (y.T).dot(np.log(pred)) - (1 - y).T.dot(np.log(1 - pred))
    #print("loss", loss)
    return loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD    
    Args:
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       loss: last loss
       w: last weights     
    """
    w = initial_w
    for iter in tqdm(range(max_iters), desc="In step"):
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
       loss: last loss
       w: last weights     
    """
    w = initial_w
    for iter in tqdm(range(max_iters), desc="In step"):
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
       loss: last loss
       w: last weights     
    """
    w = initial_w
    for iter in tqdm(range(max_iters)):
        """
        Do one step of Newton's method.
        Return the loss and updated w.
        """
        gradient = calculate_gradient(y, tx, w)
        hessian = calculate_hessian(y, tx, w)
        w -= gamma * np.linalg.solve(hessian, gradient)
    loss = calculate_loss(y, tx, w)
    return loss, w
    
################################################################################
# Fixing the numpy overflow for logistic methods
# Following and using code from http://fa.bianp.net/blog/2019/evaluate_logistic/
# Theoretical background follows Mächler, Martin (2012) “Accurately Computing log(1- exp(-|a|)) Assessed by the Rmpfr package”. The Comprehensive R Archive Network.

def stable_calculate_gradient(y, tx, w):
    a = tx.dot(w)
    # Taken from webiste!!!
    """Compute sigmoid(x) - b component-wise."""
    idx = a < 0
    sig = np.zeros_like(a)
    exp_a = np.exp(a[idx])
    y_idx = y[idx]
    sig[idx] = ((1 - y_idx) * exp_a - y_idx) / (1 + exp_a)
    exp_nx = np.exp(-a[~idx])
    y_nidx = y[~idx]
    sig[~idx] = ((1 - y_nidx) - y_nidx * exp_nx) / (1 + exp_nx)
    grad = tx.T.dot(sig) / tx.shape[0]
    return grad

def logsig(x):
    # Taken from webiste!!!
    """Compute the log-sigmoid function component-wise."""
    ls = np.zeros_like(x)
    idx0 = x < -33
    ls[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    ls[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    ls[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    ls[idx3] = -np.exp(-x[idx3])
    return ls

def stable_calculate_loss(y, tx, w):
    # Taken from webiste!!!
    """Logistic loss, numerically stable implementation."""
    a = np.dot(tx, w)
    y = np.asarray(y)
    return np.mean((1 - y) * a - logsig(a))    
    
def stable_reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Stable Regularized Logistic regression using gradient descent or SGD
    
    Args:
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       loss: last loss
       w: last weights     
    """
    w = initial_w
    for iter in tqdm(range(max_iters), desc="In step"):
        """return the loss and gradient."""
        gradient = stable_calculate_gradient(y, tx, w) + 2 * lambda_ * w
        """
        Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w.
        """
        w -= gamma * gradient
    loss = stable_calculate_loss(y, tx, w) + lambda_ * np.linalg.norm(w)**2
    return loss, w

def stable_logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Stable logistic regression using gradient descent or SGD    
    Args:
        y: expected results
        tx: inputs
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
        
    Returns:
       loss: last loss
       w: last weights    
    """
    w = initial_w
    for iter in tqdm(range(max_iters), desc="In step"):
        """
        Do one step of gradient descent using logistic regression.
        Return the loss and the updated w.
        """
        grad = stable_calculate_gradient(y, tx, w)
        w -= gamma * grad
    loss = stable_calculate_loss(y, tx, w)
    return loss, w
    

