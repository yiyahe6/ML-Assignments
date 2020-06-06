# Import required libraries
import pandas as pd
import numpy as np 
from math import ceil, sqrt
import decision_tree as dt

def preprocess(filename, normalize=False, categorial_target=False, cat_to_bin=False, is_shuffle=True, kfolds=10, get_unique_targets=False):
    """
    return training_dataset, testing_dataset in df
    """
    df = dt.load(filename, categorial_target, cat_to_bin)
    n_samples = len(df)
    if normalize:
        df = dt.normalize(df)
    unique_targets = df["class"].unique() if get_unique_targets else None
    training_dataset = dt.cross_validation_split(df, kfolds, is_shuffle)[0]
    testing_dataset = dt.cross_validation_split(df, kfolds, is_shuffle)[1]
    return training_dataset, testing_dataset, n_samples, unique_targets

"""
1.1(a) For the Iris dataset use ηmin ∈ {0.05, 0.10, 0.15, 0.20}, and calculate the accuracy using ten
fold cross-validation for each value of ηmin
"""

def iris():
    folds = 10
    training_dataset, testing_dataset, n_samples, unique_target_vals = preprocess('iris.csv', normalize=True, categorial_target=True, get_unique_targets=True)
    factors = [0.05, 0.10, 0.15, 0.20]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    accuracy, std = [], []
    stats_sensitivity = []
    stats_precision = []
    stats_recall = []
    for min_leaf in min_leafs:
        for i in range(folds):
            train, test = training_dataset[i], testing_dataset[i]
            train = train.to_numpy()
            X_test, y_test = dt.get_X_y(test)
            X_test, y_test = X_test.values.tolist(), y_test.values.tolist()
            tree = dt.DecisionTree().build_tree(train, max_depth=10, min_size=min_leaf, method="entropy")
            predict = tree.predict(X_test)
            matrix, statistics = dt.Evaluate().create_confusion_matrix(y_test, predict)
            for target_values in unique_target_vals:
                stats_sensitivity.append(statistics[target_values]["stats"]["sensitivity"])
                stats_precision.append(statistics[target_values]["stats"]["precision"])
                stats_recall.append(statistics[target_values]["stats"]["recall"])
            accuracy, std = dt.Evaluate().get_accuracy(y_test, predict)
        print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
        print("Average accuracy:  %.2f%%" % np.mean(accuracy))  
        print("Standard deviation:  %.2f%%" % np.mean(std))            
        print("Confusion matrix:\n", matrix) 
        print("Stats sensitivity: %.2f%%" % (np.mean(stats_sensitivity)*100))
        print("Stats precision: %.2f%%" % (np.mean(stats_precision)*100))
        print("Stats recall: %.2f%%" % (np.mean(stats_recall)*100))

# iris()
"""
1.1(b) For the Spambase dataset use ηmin ∈ {0.05, 0.10, 0.15, 0.20, 0.25}, and calculate the accuracy
using ten fold cross-validation for each value of ηmin
"""
def spambase():
    folds = 10
    training_dataset, testing_dataset, n_samples, unique_target_vals = preprocess('spambase.csv', normalize=True, categorial_target=True, get_unique_targets=True)
    factors = [0.05, 0.10, 0.15, 0.20, 0.25]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    accuracy, std = [], []
    stats_sensitivity = []
    stats_precision = []
    stats_recall = []
    for min_leaf in min_leafs:
        for i in range(folds):
            print(min_leaf)
            train, test = training_dataset[i], testing_dataset[i]
            train = train.to_numpy()
            X_test, y_test = dt.get_X_y(test)
            X_test, y_test = X_test.values.tolist(), y_test.values.tolist()
            tree = dt.DecisionTree().build_tree(train, max_depth=5, min_size=min_leaf, method="entropy")
            predict = tree.predict(X_test)
            matrix, statistics = dt.Evaluate().create_confusion_matrix(y_test, predict)
            for target_values in unique_target_vals:
                stats_sensitivity.append(statistics[target_values]["stats"]["sensitivity"])
                stats_precision.append(statistics[target_values]["stats"]["precision"])
                stats_recall.append(statistics[target_values]["stats"]["recall"])
            accuracy, std = dt.Evaluate().get_accuracy(y_test, predict)
        print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
        print("Average accuracy:  %.2f%%" % np.mean(accuracy))  
        print("Standard deviation:  %.2f%%" % np.mean(std))            
        print("Confusion matrix:\n", matrix) 
        print("Stats sensitivity: %.2f%%" % (np.mean(stats_sensitivity)*100))
        print("Stats precision: %.2f%%" % (np.mean(stats_precision)*100))
        print("Stats recall: %.2f%%" % (np.mean(stats_recall)*100))
# spambase()

"""
2.1(a) Grow a binary decision tree using ηmin ∈ {0.05, 0.10, 0.15}, and calculate the accuracy using ten fold cross-validation for each value of ηmin
"""
def mushroom_binary():
    folds = 10
    training_dataset, testing_dataset, n_samples, unique_target_vals = preprocess('mushroom.csv', normalize=False, categorial_target=True, get_unique_targets=True)
    factors = [0.05, 0.10, 0.15]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    accuracy, std = [], []
    stats_sensitivity = []
    stats_precision = []
    stats_recall = []
    for min_leaf in min_leafs:
        for i in range(folds):
            print(min_leaf)
            train, test = training_dataset[i], testing_dataset[i]
            train = train.to_numpy()
            X_test, y_test = dt.get_X_y(test)
            X_test, y_test = X_test.values.tolist(), y_test.values.tolist()
            tree = dt.DecisionTree().build_tree(train, max_depth=5, min_size=min_leaf, method="entropy")
            predict = tree.predict(X_test)
            accuracy, std = dt.Evaluate().get_accuracy(y_test, predict)
        print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
        print("Average accuracy:  %.2f%%" % np.mean(accuracy))  
        print("Standard deviation:  %.2f%%" % np.mean(std))            
# mushroom_binary()
    
"""
2.1(a) Grow a multiway decision tree using ηmin ∈ {0.05, 0.10, 0.15}, and calculate the accuracy using ten fold cross-validation for each value of ηmin
"""
def mushroom_multiway():
    folds = 10
    training_dataset, testing_dataset, n_samples, unique_target_vals = preprocess('mushroom.csv', normalize=False, categorial_target=True, get_unique_targets=True)
    factors = [0.05, 0.10, 0.15]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    accuracy, std = [], []
    for min_leaf in min_leafs:
        for i in range(folds):
            training_data, testing_data = training_dataset[i], testing_dataset[i]
            tree = dt.MultiwayDecisionTree(training_data).build_tree(training_data,training_data.columns[:-1],min_leaf=min_leaf)
            predicted = dt.MultiwayDecisionTree(training_data).predict(testing_data, tree)
            accuracy = dt.Evaluate().get_accuracy(testing_data, predicted)[0]
        print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
        print("Average accuracy:  %.2f%%" % np.mean(accuracy))  
# mushroom_multiway()
"""
2.1(b) Replace each categorical feature F ∈ {f1, f2, . . . , fv} with v binary features, corresponding to
each distinct value of the feature.
"""
def mushroom_binary_replace_bool():
    folds = 10
    training_dataset, testing_dataset, n_samples, unique_target_vals = preprocess('mushroom.csv', normalize=False, categorial_target=True, get_unique_targets=True,cat_to_bin=True)
    factors = [0.05, 0.10, 0.15]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    accuracy, std = [], []
    for min_leaf in min_leafs:
        for i in range(folds):
            train, test = training_dataset[i], testing_dataset[i]
            train = train.to_numpy()
            X_test, y_test = dt.get_X_y(test)
            X_test, y_test = X_test.values.tolist(), y_test.values.tolist()
            tree = dt.DecisionTree().build_tree(train, max_depth=5, min_size=min_leaf, method="entropy")
            predict = tree.predict(X_test)
        print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
        print("Average accuracy:  %.2f%%" % np.mean(accuracy))  
        print("Standard deviation:  %.2f%%" % np.mean(std))            
# mushroom_binary_replace_bool()
    
"""
6. growing decision trees using only binary splits. Use the drop in sum of squared errors (SSE) to define the splits
"""

def housing_regression():
    folds = 10
    training_dataset, testing_dataset, n_samples, unique_target_vals = preprocess('housing.csv', normalize=True, categorial_target=False, get_unique_targets=True)
    factors = [0.05, 0.10, 0.15, 0.20]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    accuracy, std = [], []
    for min_leaf in min_leafs:
        for i in range(folds):
            train, test = training_dataset[i], testing_dataset[i]
            train = train.to_numpy()
            X_test, y_test = dt.get_X_y(test)
            X_test, y_test = X_test.values.tolist(), y_test.values.tolist()
            tree = dt.DecisionTree().build_tree(train, max_depth=5, min_size=min_leaf, method="sse")
            predict = tree.predict(X_test)
            accuracy, std = dt.Evaluate().get_accuracy(y_test, predict,model_type="regressor")
        print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
        print("Average accuracy:  %.2f%%" % np.mean(accuracy))  
        print("Standard deviation:  %.2f%%" % np.mean(std)) 
# housing_regression()

