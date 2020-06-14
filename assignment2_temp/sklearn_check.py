import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
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
def linear_reg(X, y):
    #Sklearn Linear Regression
    ols=linear_model.LinearRegression()
    LR=ols.fit(X,y)
    print('Intercept', LR.intercept_, 'Weights', LR.coef_)
def main():
    df = load("housing.csv")
    X, y = get_X_y(df)
    linear_reg(X, y)
main()