"""
This is the script to reproduce our best prediction results:
Overview:
    1. Load datasets
    2. Splitt up the data by feature 22
    3. Set NaN
    4. Log normalize on Skewed data
        Features which are skewed (see notebook): 0,1,2,3,4,5,7,8,9,10,13,16,19,21,23,26,29
    5. Standardize data
    6. Remove NaN
    7. Polynominal expansion
    8. Train three seperate models of Ridge Regression on the datasets
        Best parameters for Ridge Regression (found out with grid_search):
            degree: 9
            lambda: 0.001
    9. Predict
"""
# -*- coding: utf-8 -*-
import numpy as np
from implementations import *
from helpers_perso import *
from proj1_helpers import *
from zipfile import ZipFile

# skewed features
log_norm_idx = [0,1,2,3,4,5,7,8,9,10,13,16,19,21,23,26,29]
# found best parameters with cross validation
lambda_ = 0.001
degree = 9

# drop features, from data description
drop_0 = [4, 5, 6, 8, 9, 23, 24, 25, 26, 27, 28]
drop_1 = [4, 5, 6, 26, 27, 28]

def load_data(data_path='../data/'):
    """
    Load the datasets: train and test
    """
    DATA_TRAIN_PATH = data_path + 'train.csv'
    DATA_TEST_PATH = data_path + 'test.csv'
    with ZipFile(DATA_TRAIN_PATH + '.zip', 'r') as zip:
        zip.extractall('../data')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Size of the train set: {}.".format(tX.shape))
    
    with ZipFile(DATA_TEST_PATH + '.zip', 'r') as zip:
        zip.extractall('../data')
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print("Size of the test set: {}.".format(tX_test.shape))
    return y, tX, ids, tX_test, ids_test
    
def log_norm_skewed(tX, log_norm_idx):
    """
    log1p on skewed features
    """
    tX[:, log_norm_idx] = np.log1p(tX[:, log_norm_idx])
    return tX

def split_by_feature(tX, y=None, feature22=None):
    """
    Split the data up by categorical feature 22
    """
    # Split up the dataset by feature 22 by 0, 1 and >1
    tX_0 = tX[feature22 == 0]
    tX_1 = tX[feature22 == 1]
    tX_2 = tX[feature22 > 1]
    # Drop the undefined features
    #tX_0 = np.delete(tX_0, drop_0, axis=1)
    #tX_1 = np.delete(tX_1, drop_1, axis=1)
    print("Shape 0: {}, Shape 1: {}, Shape 2: {}".format(tX_0.shape, tX_1.shape, tX_2.shape))
    if y is not None:
        y_0 = y[feature22 == 0]
        y_1 = y[feature22 == 1]
        y_2 = y[feature22 > 1]
        print("Shape 0: {}, Shape 1: {}, Shape 2: {}".format(y_0.shape, y_1.shape, y_2.shape)) 
        return tX_0, tX_1, tX_2, y_0, y_1, y_2
    else:
        return tX_0, tX_1, tX_2

if __name__ == "__main__":
    # 1. Load datasets
    y, tX, _, tX_test, ids_test = load_data()
    # 1.1 Create blank pred array
    pred_arr = np.empty([tX_test.shape[0]])
    # 1.2 Remember feature 22 - on this data we will splitt
    feature22_tX = tX[:, 22]
    feature22_tX_test = tX_test[:, 22]
    
    # 2. Splitt up the data by feature 22
    tX_0, tX_1, tX_2, y_0, y_1, y_2 = split_by_feature(tX, y, feature22_tX)
    tX_test_0, tX_test_1, tX_test_2 = split_by_feature(tX_test, None, feature22_tX_test)
    
    models_trained = []
    predictions = []
    for i, tr_set in enumerate([[tX_0, y_0, tX_test_0], [tX_1, y_1, tX_test_1], [tX_2, y_2, tX_test_2]]):
        print("Data set {}".format(i))
        tX, y_tX, tX_test = tr_set
        # 3. Set NaN
        for t_set in [tX, tX_test]:
            t_set[t_set == -999] = np.nan
        # 4. Log normalize on Skewed data
        tX = log_norm_skewed(tX, log_norm_idx)
        tX_test = log_norm_skewed(tX_test, log_norm_idx)
        # 5. Standardize data
        tX, mean_X_train, std_X_train = standardize(tX)
        tX_test, mean_X_test, std_X_test = standardize(tX_test, mean_X_train, std_X_train)
        # 6. Remove NaN
        for t_set in [tX, tX_test]:
            t_set[np.ma.masked_invalid(t_set).mask] = 0
        # 7. Polynominal expansion       
        if degree != 1:
            tX_train_p = buildpoly(tX, degree)
            tX_test_p = buildpoly(tX_test, degree)
        else:
            tX_train_p, tX_test_p = tX, tX_test
        
        # 8. Train three seperate models of Ridge Regression on the datasets
        #print("Training")
        _, w = ridge_regression(y_tX, tX_train_p, lambda_)
        models_trained.append(w)
        # 9. Predict
        #print("Predicting", tX_test_p.shape[0])
        pred = predict_labels(w, tX_test_p)
        predictions.append(pred)
        
    # place predictions back into test set
    for i in [0, 1, 2]:
        #print(i, len(predictions[i]))
        if i == 2:
            pred_arr[feature22_tX_test >= i] = predictions[i]
        else:
            pred_arr[feature22_tX_test == i] = predictions[i]
    
    # change 0 to -1
    pred_arr[pred_arr == 0] = -1
    create_csv_submission(ids_test, pred_arr, name="pred.csv")
    





