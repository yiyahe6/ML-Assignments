# Import required libraries
import pandas as pd
import numpy as np 
from math import ceil, sqrt
import decision_tree as dt


"""
1.1(a) For the Iris dataset use ηmin ∈ {0.05, 0.10, 0.15, 0.20}, and calculate the accuracy using ten
fold cross-validation for each value of ηmin
"""
def iris():
    folds = 10
    dataset = pd.read_csv('iris.csv', names=["sepal_length", "sepal_width", "petal_length",  "petal_width", "class"]) 
    dataset = dt.normalize(dataset)
    training_dataset = dt.cross_validation_split(dataset, folds)[0]
    testing_dataset = dt.cross_validation_split(dataset, folds)[1]
    n_samples = len(dataset)
    factors = [0.05, 0.10, 0.15, 0.20]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    stats = []
    for min_leaf in min_leafs:
        for i in range(folds):
            training_data, testing_data = training_dataset[i], testing_dataset[i]
            tree = dt.MultiwayDecisionTree(dataset).build_tree(training_data,training_data.columns[:-1],min_leaf=min_leaf)
            predicted = dt.MultiwayDecisionTree(dataset).index_predict(testing_data, tree)
            matrix, statistics = dt.Evaluate().create_confusion_matrix(testing_data, predicted)
            for target_values in dataset["class"].unique():
                stats.append(statistics[target_values]["stats"]["accuracy"])
        print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
        print("Stats accuracy: %.2f%%" % (np.mean(stats)*100))
        print("Confusion matrix:\n", matrix) 
"""
1.2 Interpreting the results
(a) Select the best value of ηmin for the Iris dataset, and create a class confusion matrix using
ten-fold cross validation(use only the test set for populating the confusion matrix ). How do
you interpret the confusion matrix, and why?

(b) Select the best value of ηmin for the Spambase dataset, and create a class confusion matrix
using ten-fold cross validation(use only the test set for populating the confusion matrix ). How
do you interpret the confusion matrix, and why?

(c) How does different values of ηmin impact classifier performance for both datasets and why?
Support your claims/insights through your results.

"""
# iris()
# def mushroom_binary():
"""
2.1(a) Grow a multiway decision tree using ηmin ∈ {0.05, 0.10, 0.15}, and calculate the accuracy using ten fold cross-validation for each value of ηmin
"""
def mushroom_multiway():
    folds = 10
    dataset = pd.read_csv('mushroom.csv', names=list(range(21))+ ["class"]) 
    training_dataset = dt.cross_validation_split(dataset, folds)[0]
    testing_dataset = dt.cross_validation_split(dataset, folds)[1]
    n_samples = len(dataset)
    factors = [0.05, 0.10, 0.15]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    stats = []
    for min_leaf in min_leafs:
        for i in range(folds):
            training_data, testing_data = training_dataset[i], testing_dataset[i]
            tree = dt.MultiwayDecisionTree(dataset).build_tree(training_data,training_data.columns[:-1],min_leaf=min_leaf)
            predicted = dt.MultiwayDecisionTree(dataset).index_predict(testing_data, tree)
            matrix, statistics = dt.Evaluate().create_confusion_matrix(testing_data, predicted)
            for target_values in dataset["class"].unique():
                stats.append(statistics[target_values]["stats"]["accuracy"])
        print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
        print("Stats accuracy: %.2f%%" % (np.mean(stats)*100))
# mushroom_multiway()
"""
2.2 Interpreting the results
(a) Select the best value of ηmin for the the above two cases i.e., multiway and binary splits,
and create a class confusion matrix using ten-fold cross validation(use only the test set for
populating the confusion matrix ). How do you interpret the confusion matrix, and why?
(b) Is there a difference in the optimal value of ηmin for mulitway vs binary splits? Please explain
your finding using your results i.e., if there is a difference what are the probable causes?
on the other hand if the optimal values are similar what does this tell you about binary vs
multiway splitting in decision trees?
"""
"""
3 Entropy (20 points)
Consider training a binary decision tree using entropy splits.
(a) Prove that the decrease in entropy by a split on a binary yes/no feature can never be greater
than 1 bit.
Entropy is a meaure of average amount of information from an event. 
(b) Generalize this result to the case of arbitrary multiway branching.
"""
"""
6. growing decision trees using only binary splits. Use the drop in sum of squared errors (SSE) to define the splits
"""
def housing_regression():
    folds = 10
    dataset = pd.read_csv('housing.csv', names=["CRIM", "ZN", "INDUS",  "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "class"]) 
    dataset = dataset[:50]
    dataset = dt.normalize(dataset)
    training_dataset = dt.cross_validation_split(dataset, folds)[0]
    testing_dataset = dt.cross_validation_split(dataset, folds)[1]
    n_samples = len(dataset)
    factors = [0.05, 0.10, 0.15, 0.20]
    min_leafs = [ceil(factor*n_samples) for factor in factors]
    stats = []
    training_data, testing_data = training_dataset[0], testing_dataset[0]
    X = training_data.drop('class', axis=1)
    y = training_data['class']
    X_test = testing_data.drop('class', axis=1)
    regressor = dt.DecisionTreeRegressor().fit(X, y)
    preds = regressor.predict(X_test)
    print("real\n", preds)
    # for min_leaf in min_leafs:
    #     for i in range(folds):
    #         training_data, testing_data = training_dataset[i], testing_dataset[i]
    #         X = training_data.drop('class', axis=1)
    #         y = training_data['class']
    #         X_test = testing_data.drop('class', axis=1)
    #         rtree = r_instance.fit(X, y)
    #         # print(rtree)
    #         pred = rtree.predict(X_test)
    #         # matrix, statistics = c_instance.create_confusion_matrix(testing_data, tree)

    #     print("min_leaf is {min_leaf}".format(min_leaf=min_leaf))
    #     print(pred)


