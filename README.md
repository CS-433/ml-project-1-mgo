
# Prediction of the Higgs Boson particle

Given the CERN dataset this code will predict if a Higgs Boson particle occurs

Members of this repository : Oliver Becker, Maxence Bertinger

## Requirements
python 3.8
numpy
matplotlib
tqdm
for the notebook:
seaborn

## Run predictions
Running 'python run.py' will create a pred.csv file with the predictions in it.
Overview of run.py:
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

## Content 
- `run.py` creates submission predictions with best method 
- `proj1_helpers.py` functions to load data, create submission and predict labels
- `implementations.py` implements the ML methods. To fix the numpy overflow for logistic methods there is a stable version for logistic regression and regualized logistic regression
- `plots.py` some visualizations of the data for the notebook
- `helpers_perso.py` functions for standarization, splitting data, ployn. feature expansion, grid search, trainer, cross validation and calculation of the score
