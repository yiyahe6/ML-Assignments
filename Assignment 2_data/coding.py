
from random import seed
from random import randrange
from csv import reader
import pandas as pd
import numpy as np
from statistics import mean
from numpy import math
from collections import OrderedDict
from scipy.stats.stats import zscore
import random
from matplotlib import pyplot as plt
import warnings

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

def normalize(X, mean=None, std=None):
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

class RidgeRegression:
    def fit(self, X, y, lam=0.2):
        intercept = np.mean(y)
        y = y - np.mean(y)
        I = np.eye(X.shape[1])
        part1 = np.linalg.inv(X.T.dot(X) + lam * I)
        part2 = X.T.dot(y)
        w = part1.dot(part2)
        w = np.insert(w, 0, intercept)
        return w
    def predict(self, X ,w):
        m = X.shape[0]
        X = np.c_[np.full(m, 1.0), X]
        y_predict = np.dot(w, X.T)
        return y_predict

class LeastSquaresRegression:

    def normal_equation(self, X, y):
        m = X.shape[0] 
        X = np.c_[np.full(m, 1.0), X]
        part1 = np.linalg.inv(X.T.dot(X))
        part2 = X.T.dot(y)
        w = part1.dot(part2)
        return w
    def predict(self, X ,w):
        m = X.shape[0]
        X = np.c_[np.full(m, 1.0), X]
        y_predict = np.dot(w, X.T)
        return y_predict

class GradientDescent:
    # Our goal is to find the lowest point of the cost function curve 
    def cal_cost(self, X, y, w):
        y_predict = X.dot(w)
        squared_error = (y_predict-y) ** 2
        mse = np.mean(squared_error)
        rmse = np.sqrt(mse)
        return rmse
    def fit(self, X, y, learning_rate = 0.001, tolerance = 0.001, max_iteration = 1000):
        m = X.shape[0] 
        # add a column of ones as constant feature to the left of X
        X = np.c_[np.full(m, 1.0), X]
        n_features = X.shape[1]
        # init  
        w = np.zeros(n_features) 
        cost_history = []
        cost = self.cal_cost(X, y, w) 
        for i in range(max_iteration):
            y_predict = X.dot(w)
            gradient = X.T.dot(y_predict - y) / m
            w = w - learning_rate * gradient
            new_cost = self.cal_cost(X, y, w) 
            cost_diff = new_cost - cost
            if abs(cost_diff) < tolerance:
                break
            cost = new_cost
            cost_history.append(cost)
        return w, cost_history
    def predict(self, X, w):
        m = X.shape[0]
        X = np.c_[np.full(m, 1.0), X]
        y_predict = np.dot(w, X.T)
        return y_predict

class Evaluate:
    def sse(self, y, y_predict):
        squared_error = (y_predict-y) ** 2
        sse = np.sum(squared_error)
        return sse
    def rmse(self, y, y_predict):
        squared_error = (y_predict-y) ** 2
        mse = np.mean(squared_error)
        rmse = np.sqrt(mse)
        return rmse
    def mse(self, y, y_predict):
        squared_error = (y_predict-y) ** 2
        mse = np.mean(squared_error)
        return mse
    def score(self, y, y_predict, type="rmse"):
        score = None
        if type == "rmse":
            score = self.rmse(y, y_predict)
        elif type == "mse":
            score = self.mse(y, y_predict)
        else:
            score = self.mse(y, y_predict)
        return score
"""
Solutions
=================================================================================================
"""
"""
Q2
"""
def housing_linear_regression():
    print("Result for housing_linear_regression")
    print("------------------------------------")
    learn_rate = 0.0004
    tol = 0.005
    df = load("housing.csv")
    folds = 10
    training_dataset = cross_validation_split(df, folds, is_shuffle=True)[0]
    testing_dataset = cross_validation_split(df, folds, is_shuffle=True)[1]
    rmse_test_list, rmse_train_list = [], []
    sse_list = []
    fold_to_plot = random.randint(0, 9)
    print("Report the RMSE (both training and test) for each fold")
    for i in range(folds):
        train, test = training_dataset[i], testing_dataset[i]
        X_train, y_train = get_X_y(train)
        X_test, y_test = get_X_y(test)
        X_train, mean, std = normalize(X_train)['data'], normalize(X_train)['mean'], normalize(X_train)['std']
        X_test = normalize(X_test, mean, std)['data']
        GD = GradientDescent()
        w_gd, cost_history = GD.fit(X_train, y_train, learning_rate=learn_rate, tolerance=tol)
        y_predict_test = GD.predict(X_test, w_gd)
        y_predict_train = GD.predict(X_train, w_gd)
        evaluate = Evaluate()
        rmse_test = evaluate.score(y_predict_test, y_test)
        rmse_test_list.append(rmse_test)
        rmse_train = evaluate.score(y_predict_train, y_train)
        rmse_train_list.append(rmse_train)
        sse = evaluate.score(y_predict_test, y_test)
        sse_list.append(sse)
        # choose any fold to plot
        if i == fold_to_plot:
            plt.plot(range(1000), cost_history, 'b', label="train")
            plt.legend()
            plt.title('housing: training RMSE over iteration') 
            plt.xlabel('iteration')
            plt.ylabel('RMSE')
            plt.show()  
        print("rmse_test: %5.2f, rmse_train: %5.2f" % (rmse_test, rmse_train))
    print("overall mean RMSE for testing data: %5.2f, overall mean RMSE for training data: %5.2f" % (np.mean(rmse_test_list), np.mean(rmse_train_list)))
    print("average SSE: %5.2f, standard deviation: %5.2f" % (np.mean(sse_list), np.std(sse_list)))
    print("========================================================================================================\n") 
def yatcht_linear_regression():
    print("Result for yatcht_linear_regression")
    print("-----------------------------------")
    folds = 10
    learn_rate = 0.001
    tol = 0.001
    df = load("yachtData.csv")
    training_dataset = cross_validation_split(df, folds, is_shuffle=True)[0]
    testing_dataset = cross_validation_split(df, folds, is_shuffle=True)[1]
    rmse_test_list, rmse_train_list = [], []
    sse_list = []
    fold_to_plot = random.randint(0, 9)
    print("Report the RMSE (both training and test) for each fold")
    for i in range(folds):
        train, test = training_dataset[i], testing_dataset[i]
        X_train, y_train = get_X_y(train)
        X_test, y_test = get_X_y(test)
        X_train, mean, std = normalize(X_train)['data'], normalize(X_train)['mean'], normalize(X_train)['std']
        X_test = normalize(X_test, mean, std)['data']
        GD = GradientDescent()
        w_gd, cost_history = GD.fit(X_train, y_train, learning_rate=learn_rate, tolerance=tol)
        y_predict_test = GD.predict(X_test, w_gd)
        y_predict_train = GD.predict(X_train, w_gd)
        evaluate = Evaluate()
        rmse_test = evaluate.score(y_predict_test, y_test)
        rmse_test_list.append(rmse_test)
        rmse_train = evaluate.score(y_predict_train, y_train)
        rmse_train_list.append(rmse_train)
        sse = evaluate.score(y_predict_test, y_test)
        sse_list.append(sse)
        # choose any fold to plot
        if i == fold_to_plot:
            plt.plot(range(1000), cost_history, 'b', label="train")
            plt.legend()
            plt.title('yachtData: training RMSE over iteration') 
            plt.xlabel('iteration')
            plt.ylabel('RMSE')
            plt.show()  
        print("rmse_test: %5.2f, rmse_train: %5.2f" % (rmse_test, rmse_train))
    print("overall mean RMSE for testing data: %5.2f, overall mean RMSE for training data: %5.2f" % (np.mean(rmse_test_list), np.mean(rmse_train_list)))
    print("average SSE: %5.2f, standard deviation: %5.2f" % (np.mean(sse_list), np.std(sse_list)))
    print("========================================================================================================\n")
def concrete_linear_regression():
    print("Result for concrete_linear_regression")
    print("-------------------------------------")
    folds = 10
    learn_rate = 0.0007
    tol = 0.0001
    df = load("concreteData.csv")
    training_dataset = cross_validation_split(df, folds, is_shuffle=True)[0]
    testing_dataset = cross_validation_split(df, folds, is_shuffle=True)[1]
    rmse_test_list, rmse_train_list = [], []
    sse_list = []
    fold_to_plot = random.randint(0, 9)
    print("Report the RMSE (both training and test) for each fold")
    for i in range(folds):
        train, test = training_dataset[i], testing_dataset[i]
        X_train, y_train = get_X_y(train)
        X_test, y_test = get_X_y(test)
        X_train, mean, std = normalize(X_train)['data'], normalize(X_train)['mean'], normalize(X_train)['std']
        X_test = normalize(X_test, mean, std)['data']
        GD = GradientDescent()
        w_gd, cost_history = GD.fit(X_train, y_train, learning_rate=learn_rate, tolerance=tol)
        y_predict_test = GD.predict(X_test, w_gd)
        y_predict_train = GD.predict(X_train, w_gd)
        evaluate = Evaluate()
        rmse_test = evaluate.score(y_predict_test, y_test)
        rmse_test_list.append(rmse_test)
        rmse_train = evaluate.score(y_predict_train, y_train)
        rmse_train_list.append(rmse_train)
        sse = evaluate.score(y_predict_test, y_test)
        sse_list.append(sse)
        # choose any fold to plot
        if i == fold_to_plot:
            plt.plot(range(1000), cost_history, 'b', label="train")
            plt.legend()
            plt.title('concreteData: training RMSE over iteration') 
            plt.xlabel('iteration')
            plt.ylabel('RMSE')
            plt.show()  
        print("rmse_test: %5.2f, rmse_train: %5.2f" % (rmse_test, rmse_train))
    print("overall mean RMSE for testing data: %5.2f, overall mean RMSE for training data: %5.2f" % (np.mean(rmse_test_list), np.mean(rmse_train_list)))
    print("average SSE: %5.2f, standard deviation: %5.2f" % (np.mean(sse_list), np.std(sse_list)))
    print("========================================================================================================\n")
"""
Q3
"""
def housing_least_squared_regression():
    print("Result for housing_least_squared_regression")
    print("-------------------------------------------")
    folds = 10
    df = load("housing.csv")
    training_dataset = cross_validation_split(df, folds, is_shuffle=True)[0]
    testing_dataset = cross_validation_split(df, folds, is_shuffle=True)[1]
    rmse_test_list, rmse_train_list = [], []
    sse_list = []
    print("Report the RMSE (both training and test) for each fold")
    for i in range(folds):
        # load data
        train, test = training_dataset[i], testing_dataset[i]
        X_train, y_train = get_X_y(train)
        X_test, y_test = get_X_y(test)
        # normalize
        X_train, mean, std = normalize(X_train)['data'], normalize(X_train)['mean'], normalize(X_train)['std']
        X_test = normalize(X_test, mean, std)['data']
        # Least Squares Regression
        LSR = LeastSquaresRegression()
        w_lsr = LSR.normal_equation(X_train, y_train)
        # print('Intercept', w[0], 'Weights', w[1:])
        y_predict_test = LSR.predict(X_test, w_lsr)
        y_predict_train = LSR.predict(X_train, w_lsr)
        evaluate = Evaluate()
        rmse_test = evaluate.score(y_predict_test, y_test)
        rmse_test_list.append(rmse_test)
        rmse_train = evaluate.score(y_predict_train, y_train)
        rmse_train_list.append(rmse_train)
        sse = evaluate.score(y_predict_test, y_test)
        sse_list.append(sse)
        print("rmse_test: %5.2f, rmse_train: %5.2f" % (rmse_test, rmse_train))
    print("overall mean RMSE for testing data: %5.2f, overall mean RMSE for training data: %5.2f" % (np.mean(rmse_test_list), np.mean(rmse_train_list)))
    print("========================================================================================================\n")

def yatcht_least_squared_regression():
    print("Result for yatcht_least_squared_regression")
    print("------------------------------------------")
    folds = 10
    df = load("yachtData.csv")
    training_dataset = cross_validation_split(df, folds, is_shuffle=True)[0]
    testing_dataset = cross_validation_split(df, folds, is_shuffle=True)[1]
    rmse_test_list, rmse_train_list = [], []
    sse_list = []
    print("Report the RMSE (both training and test) for each fold")
    for i in range(folds):
        # load data
        train, test = training_dataset[i], testing_dataset[i]
        X_train, y_train = get_X_y(train)
        X_test, y_test = get_X_y(test)
        # normalize
        X_train, mean, std = normalize(X_train)['data'], normalize(X_train)['mean'], normalize(X_train)['std']
        X_test = normalize(X_test, mean, std)['data']
        # Least Squares Regression
        LSR = LeastSquaresRegression()
        w_lsr = LSR.normal_equation(X_train, y_train)
        # print('Intercept', w[0], 'Weights', w[1:])
        y_predict_test = LSR.predict(X_test, w_lsr)
        y_predict_train = LSR.predict(X_train, w_lsr)
        evaluate = Evaluate()
        rmse_test = evaluate.score(y_predict_test, y_test)
        rmse_test_list.append(rmse_test)
        rmse_train = evaluate.score(y_predict_train, y_train)
        rmse_train_list.append(rmse_train)
        sse = evaluate.score(y_predict_test, y_test)
        sse_list.append(sse)
        print("rmse_test: %5.2f, rmse_train: %5.2f" % (rmse_test, rmse_train))
    print("overall mean RMSE for testing data: %5.2f, overall mean RMSE for training data: %5.2f" % (np.mean(rmse_test_list), np.mean(rmse_train_list)))
    print("========================================================================================================\n")

"""
Q5
"""
def sinData_polynomial_regression():
    print("Result for sinData_polynomial_regression")
    print("----------------------------------------")
    train_file = "sinData_Train.csv"
    val_file = "sinData_Validation.csv"
    df_train = load(train_file)
    df_val = load(val_file)
    X_train, y_train = get_X_y(df_train)
    X_val, y_val = get_X_y(df_val)
    degree = 15
    # create polynomial features
    p = np.arange(1, degree + 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        X_train = np.hstack((X_train ** i for i in p))
        X_val = np.hstack((X_val ** i for i in p))
    # fit model and predict
    lsr = LeastSquaresRegression()
    w = lsr.normal_equation(X_train, y_train)
    y_predict = lsr.predict(X_val, w)
    # get score
    evaluate = Evaluate()
    rmse_score = evaluate.score(y_val, y_predict, type="rmse")
    print("RMSE for whole validation set is: %.2f" % rmse_score)
    # evaluate polynomial fit
    score_list_val, score_list_train = [], []
    for i in p:
        # validation
        y_predict_val = lsr.predict(X_val[:,:i], w[:i+1])
        score_val = evaluate.score(y_val, y_predict_val, type="mse")
        score_list_val.append(score_val)
        # train
        y_predict_train = lsr.predict(X_train[:,:i], w[:i+1])
        score_train = evaluate.score(y_train, y_predict_train, type="mse")
        score_list_train.append(score_train)
    print("mean SSE for validation set is: \n", score_list_val)
    print("mean SSE for train set is: \n", score_list_train)
    # plot
    plt.plot(p, score_list_val, 'g', label="validation")
    plt.plot(p, score_list_train, 'b', label="train")
    plt.legend()
    plt.title('sinData: polynomial fit') 
    plt.xlabel('Power of input feature')
    plt.ylabel('mean SSE')
    plt.show()
    print("========================================================================================================\n")

def yatcht_polynomial_regression():
    print("Result for yatcht_polynomial_regression")
    print("---------------------------------------")
    folds = 10
    df = load("yachtData.csv")
    training_dataset = cross_validation_split(df, folds, is_shuffle=True)[0]
    testing_dataset = cross_validation_split(df, folds, is_shuffle=True)[1]
    rmse_test_list, rmse_train_list = [], []
    degree = 7
    final_score_val, final_score_train = [], []
    for i in range(folds):
        train, val = training_dataset[i], testing_dataset[i]
        X_train, y_train = get_X_y(train)
        X_val, y_val = get_X_y(val)
        X_train, mean, std = normalize(X_train)['data'], normalize(X_train)['mean'], normalize(X_train)['std']
        X_val = normalize(X_val, mean, std)['data']
        # create polynomial features
        p = np.arange(1, degree + 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            X_train = np.hstack((X_train ** i for i in p))
            X_val = np.hstack((X_val ** i for i in p))
        # get weights
        lsr = LeastSquaresRegression()
        w = lsr.normal_equation(X_train, y_train)
        # evaluate polynomial fit
        evaluate = Evaluate()
        score_list_val, score_list_train = [], []
        for j in p:
            # validation
            y_predict_val = lsr.predict(X_val[:,:j], w[:j+1])
            score_val = evaluate.score(y_val, y_predict_val, type="rmse")
            score_list_val.append(score_val)
            # train
            y_predict_train = lsr.predict(X_train[:,:j], w[:j+1])
            score_train = evaluate.score(y_train, y_predict_train, type="rmse")
            score_list_train.append(score_train)
        final_score_val.append(score_list_val)
        final_score_train.append(score_list_train)
    final_score_val, final_score_train = np.array(final_score_val), np.array(final_score_train)
    avg_score_val, avg_score_train = np.mean(final_score_val, axis=0), np.mean(final_score_train, axis=0)
    print("avg_score_train", avg_score_train)
    print("avg_score_val", avg_score_val)
    # plot
    plt.plot(p, avg_score_val, 'g', label="validation")
    plt.plot(p, avg_score_train, 'b', label="train")
    plt.legend()
    plt.title('yachtData: polynomial fit') 
    plt.xlabel('Power of input feature')
    plt.ylabel('average RMSE across 10 folders')
    plt.show()
    print("========================================================================================================\n")
"""
Q7
"""
def sinData_ridge_regression(degree):
    print("Result for sinData_ridge_regression with p of %s" % degree)
    print("------------------------------------------------")
    # get list of lamdas
    lam_options = []
    lam = 0
    while lam <= 10:
        lam_options.append(round(lam, 1))
        lam += 0.2
    # 10 fold cross validation
    folds = 10
    df = load("sinData_Train.csv")
    training_dataset = cross_validation_split(df, folds, is_shuffle=True)[0]
    testing_dataset = cross_validation_split(df, folds, is_shuffle=True)[1]
    final_score_test, final_score_train = [], []
    for i in range(folds):
        train, val = training_dataset[i], testing_dataset[i]
        X_train, y_train = get_X_y(train)
        X_test, y_test = get_X_y(val)
        X_train, mean, std = normalize(X_train)['data'], normalize(X_train)['mean'], normalize(X_train)['std']
        X_test = normalize(X_test, mean, std)['data']
        # create polynomial features
        p = np.arange(1, degree + 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            X_train = np.hstack((X_train ** i for i in p))
            X_test = np.hstack((X_test ** i for i in p))
        # model fit and evaluate
        rr = RidgeRegression()
        evaluate = Evaluate()
        score_test_list, score_train_list = [], []
        for lam in lam_options:
            w = rr.fit(X_train, y_train, lam)
            y_predict_test = rr.predict(X_test, w)
            y_predict_train = rr.predict(X_train, w)
            score_test = evaluate.score(y_test, y_predict_test, type="rmse")
            score_train = evaluate.score(y_train, y_predict_train, type="rmse")
            score_test_list.append(score_test)
            score_train_list.append(score_train)
        final_score_test.append(score_test_list)
        final_score_train.append(score_train_list)
    final_score_test, final_score_train = np.array(final_score_test), np.array(final_score_train)
    avg_score_test, avg_score_train = np.mean(final_score_test, axis=0), np.mean(final_score_train, axis=0)
    print("avg_score_train", avg_score_train)
    print("avg_score_test", avg_score_test)
    # plot
    plt.plot(lam_options, avg_score_test, 'g', label="test")
    plt.plot(lam_options, avg_score_train, 'b', label="train")
    plt.legend()
    plt.title('sinData: RMSE vs lamda') 
    plt.xlabel('lamda')
    plt.ylabel('average RMSE across 10 folders')
    plt.show()
    print("========================================================================================================\n")

def main():
    # housing_linear_regression()
    # yatcht_linear_regression()
    # concrete_linear_regression()
    # housing_least_squared_regression()
    # yatcht_least_squared_regression()
    # sinData_polynomial_regression()
    # yatcht_polynomial_regression()
    # sinData_ridge_regression(5)
    sinData_ridge_regression(9)

main()

