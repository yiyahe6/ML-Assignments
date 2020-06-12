
from random import seed
from random import randrange
from csv import reader
import pandas as pd
import numpy as np
from statistics import mean
from numpy import math
from collections import OrderedDict
from scipy.stats.stats import zscore
from sklearn.preprocessing._data import StandardScaler
import random

def load(file_path='housing.csv'):
    try:
        df = pd.read_csv(file_path, header=None)
        df.rename(columns={ df.columns[-1]: "target" }, inplace=True)
        return df
    except:
        print("no file found")
 
def shuffle(df):
    df = df.set_index(np.random.permutation(df.index)) 
    df = df.reset_index(drop=True) 
    return df

def cross_validation_split(df, folds=10, is_shuffle=True):
    """
    Split a dataset into k folds
    return a tuple of two lists: (list of training data in df, list of testing data in df)
    """
    dataset_split = []
    df_copy = shuffle(df) if is_shuffle else df
    fold_size = int(df_copy.shape[0] / folds)
    training_dataset = []
    testing_dataset = []
    for i in range(folds):
        fold = []
        while len(fold) < fold_size:
            r = randrange(df_copy.shape[0])
            index = df_copy.index[r]
            fold.append(df_copy.loc[[index]])
            df_copy = df_copy.drop(index)
        dataset_split.append(pd.concat(fold))
    for i in range(folds):
        r = list(range(folds))
        r.pop(i)
        for j in r :
            if j == r[0]:
                cv = dataset_split[j]
            else:    
                cv=pd.concat([cv,dataset_split[j]])
        training_data = cv.reset_index(drop=True)
        testing_data = dataset_split[i].reset_index(drop=True)
        training_dataset.append(training_data)
        testing_dataset.append(testing_data)
    return training_dataset, testing_dataset

def get_X_y(df):
    """
    Returns
    -------
    X: features data in array-alike format
    y: target in array-alike format 
    """
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return X, y

def normalize(X):
    """
    normalize continous feature values using z-score.
    z = (x - u) / s

    Parameters
    ----------
    X: array-like features data
    
    Returns
    -------
    dictionary that stores transdata, mean, std  
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    trans_data = (X - mean) / std
    return {"data": trans_data, "mean": mean, "std": std}

class GradientDescent:
    # Our goal is to find the lowest point of the cost function curve 
    # or the value of wi where cost is the lowest as descending gradually

    # root mean squared error
    def cal_cost(self, X, y, w):
        y_predict = X.dot(w)
        # print(y_predict)
        # for i in range(len(y)):
        #     y_predict[i] = round(y_predict[i], 2)
        rmse = np.sqrt(np.mean((y_predict-y)**2))
        # print(rmse)
        return rmse
    def gd(self, X, y, learning_rate = 0.001, tolerence = 0.001, max_iteration = 1000):
        m = X.shape[0] 
        # add a column of ones as constant feature to the left of X
        X = np.c_[np.full(m, 1.0), X]
        n_features = X.shape[1]
        # init  
        w = np.zeros(n_features) 
        cost = self.cal_cost(X, y, w) 
        for i in range(max_iteration):
            y_predict = X.dot(w)
            # for i in range(len(y)):
            #     y_predict[i] = round(y_predict[i], 2)
            gradient = np.mean(X.T.dot(y_predict - y))
            w = w - learning_rate * gradient
            new_cost = self.cal_cost(X, y, w) 
            cost_diff = new_cost - cost
            if abs(cost_diff) < tolerence:
                break
            cost = new_cost
            # print("y_predict", y_predict)
        return w
    def predict(self, X, w):
        m = X.shape[0]
        X = np.c_[np.full(m, 1.0), X]
        y_predict = np.dot(w, X.T)
        return y_predict
def main():
    # test_preprocess()
    # theta = np.random.randn(2) # generate two random numbers [r1, r2]
    # print(theta)
    # gen_line_data()
    df = load("yachtData.csv")
    # df = df[0:5]
    print(df.head())
    X, y = get_X_y(df)
    instance = GradientDescent()
    # w = np.full(X.shape[1], 1.0)
    # rmse = instance.cal_cost(X, y, w)
    # print(rmse)
    w = instance.gd(X, y)
    print(w)
main()
