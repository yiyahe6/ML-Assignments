
from random import seed
from random import randrange
from csv import reader
import pandas as pd
import numpy as np
from statistics import mean
from numpy import math
from collections import OrderedDict

def load(file_path='iris.csv', categorial_target=False, cat_to_bin=False):
    try:
        df = pd.read_csv(file_path, header=None)
        n_samples, n_features = df.shape[0], df.shape[1]
        df.rename(columns={ df.columns[-1]: "class" }, inplace=True)
        target_name = df.columns[-1] 
        target_names = np.unique(df[target_name])
        if categorial_target:
            dic = {target_name: {}}
            for i, c in enumerate(target_names):
                dic[target_name][c] = i
            df.replace(dic, inplace=True)
        if cat_to_bin:
            df = pd.get_dummies(df)
        return df
    except:
        print("no file found")
 
def shuffle(df):
    df = df.copy()
    df = df.set_index(np.random.permutation(df.index)) 
    df = df.reset_index(drop=True) 
    return df

# Split a dataset into k folds
def cross_validation_split(df, folds=10, is_shuffle=True):
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
    X: features in dataFrame format
    y: target in dataFrame format
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def normalize(df):
    """
    normalize continous feature values
    """
    result = df.copy()
    for feature_name in df.columns[:-1]:
        df[feature_name] = df[feature_name].astype(float)
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

class DecisonTreeNode:
    def __init__(self, data, depth, max_depth, min_size, method="entropy"):
        self.data = data
        self.split_col = None
        self.split_val = None
        self.left = None
        self.right = None
        self.max_depth = max_depth
        self.min_size = min_size
        self.depth = depth
        self.method = method
        self.is_leaf = False
    # Split a dataset based on an attribute and an attribute value
    def divide(self, data, split_col, split_val):
        left, right = list(), list()
        for row in data:
            if row[split_col] < split_val:
                left.append(row)
            else:
                right.append(row)
        return left, right
    def entropy(self, target_col):
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy
    def cal_sse(self, target_col):
        y_mean = mean(target_col)
        sse = np.square(target_col - y_mean).sum()
        return sse
    def info_gain(self, groups, class_values):
        gain = 0.0
        parent_gain = self.entropy([row[-1] for row in self.data])
        for class_value in class_values:
            for group in groups:
                size = len(group)
                if size == 0:
                    continue
                weight = [row[-1] for row in group].count(class_value) / float(size)
                gain += (weight * self.entropy([row[-1] for row in group]))
        return gain
    
    # Select the best split point for a dataset
    def find_best_split(self):
        class_values = list(set(row[-1] for row in self.data))
        score, node = 0.0, None
        for col in range(len(self.data[0])-1):
            for row in self.data:
                groups = self.divide(self.data, col, row[col])
                if self.method == "entropy":
                    curr_score = self.info_gain(groups, class_values)
                else: # sse
                    # decrease in impurity
                    left, right = 0, 0
                    curr_score = self.cal_sse([row[-1] for row in self.data])
                    if groups[0]:
                        left = self.cal_sse([row[-1] for row in groups[0]]) 
                    if groups[1]:
                        right = self.cal_sse([row[-1] for row in groups[1]]) 
                    curr_score -= left - right
                if curr_score > score:
                    self.split_col, self.split_val, score, node = col, row[col], curr_score, groups
        return {'index':self.split_col, 'value':self.split_val, "score": curr_score, 'groups':node} # value is split_val, index is best_feature, group is left, right list of row index
    
    # Create a terminal node value
    def to_terminal(self):
        outcomes = [row[-1] for row in self.data]
        return max(set(outcomes), key=outcomes.count)
    
    # Recursively split
    def split(self):
        split_info = self.find_best_split()
        # stopping criterior
        if not split_info['groups']:
            self.is_leaf = True
            return 
        left, right = split_info['groups']
        if not left or not right:
            self.is_leaf = True
            return 
        if len(self.data) <= self.min_size:
            self.is_leaf = True
            return 
        if self.depth >= self.max_depth:
            self.is_leaf = True

            return 
        if split_info["score"] <= 0:
            self.is_leaf = True
            return 
        # create left, right node
        self.left = DecisonTreeNode(left, self.depth+1, self.max_depth, self.min_size, self.method)
        self.right = DecisonTreeNode(right, self.depth+1, self.max_depth, self.min_size, self.method)
        
        self.split_col, self.split_val = split_info['index'], split_info['value']

        if not self.left or not self.right:
            self.is_leaf = True
            return 
        if self.left:
            self.left.split()
        if self.right:
            self.right.split() 
    def predict_row(self, x):

        if self.is_leaf: return self.to_terminal()
        if x[self.split_col] < self.split_val:
            return self.left.predict_row(x)
        else:
            return self.right.predict_row(x)
    def predict(self,X_test):
        return np.array([self.predict_row(x) for x in X_test])

class DecisionTree:
    # Build a decision tree
    def build_tree(self, train, max_depth, min_size, method):
        self.tree = DecisonTreeNode(train, 1, max_depth, min_size, method)
        self.tree.split()
        return self.tree
    def predict(self, X_test):
        return self.tree.predict(X_test)

class MultiwayDecisionTree:
    def __init__(self, dataset):
        
        self.dataset = dataset

    def entropy(self,target_col):
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy

    def info_gain(self,data,split_attribute_name,target_name="class"):  
        parent_entropy = self.entropy(data[target_name])
        vals,counts= np.unique(data[split_attribute_name],return_counts=True)
        weighted_entropy = np.sum([(counts[i]/np.sum(counts))*self.entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
        info_gain = parent_entropy - weighted_entropy
        return info_gain

        
    def build_tree(self,data,features,target_attribute_name="class",default_class = None,min_leaf=10):
        # stopping criterier
        # all target_values have the same value, return this value
        if len(np.unique(data[target_attribute_name])) <= 1:
            return np.unique(data[target_attribute_name])[0]
        elif len(features) == 0:
            return default_class
        elif len(data[target_attribute_name]) <= min_leaf:
            return default_class
        else:
            # most frequent class of the current node
            default_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
            item_values = [self.info_gain(data,feature,target_attribute_name) for feature in features] 
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]
            
            multiway_tree = {best_feature:{}}
            
            # remove selected feature
            features = [i for i in features if i != best_feature]
            
            # grow branches for selected feature
            for value in np.unique(data[best_feature]):
                value = value
                sub_data = data.where(data[best_feature] == value).dropna()   
                subtree = self.build_tree(sub_data,features,target_attribute_name,default_class,min_leaf)                
                multiway_tree[best_feature][value] = subtree
            return multiway_tree    
    
    def _predict(self,query,tree,default=1):
        for key in list(query.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][query[key]] 
                except:
                    return default
                result = tree[key][query[key]]
                if isinstance(result,dict):
                    return self._predict(query,result)
                else:
                    return result

    def predict(self,testing_data,tree):
        queries = testing_data.iloc[:,:-1].to_dict(orient = "records") 
        predicted = [] 
        for i in range(len(testing_data)):
            predicted = predicted.append(self._predict(queries[i],tree)) 
        return predicted

class Evaluate:
    def get_accuracy(self,y_actual,y_predicted,model_type="classification"): 
        std = None
        if model_type == "classification": 
            accuracy = np.sum(y_predicted == y_actual)/len(y_actual)*100
            std = np.std(y_predicted == y_actual)*100
        else: # regressor, use RMSE to evaluate how far 
            s, n = 0, len(y_actual)
            for i in range(n):
                diff = y_actual[i] - y_predicted[i]
                s += diff ** 2
            accuracy = np.sqrt(s/n)
        return accuracy, std
    
    def create_confusion_matrix(self,y_actual,y_predicted):
        actual = np.array(y_actual)
        pred = np.array(y_predicted)
        n_classes = np.unique(actual)
        K = len(n_classes) # Number of classes 
        matrix = np.zeros((K, K))
        imap = {key: i for i, key in enumerate(n_classes)}
        for p, a in zip(pred, actual):
            matrix[imap[p]][imap[a]] += 1
        sigma = sum([sum(matrix[imap[i]]) for i in n_classes])
        # Scaffold Statistics Data Structure
        statistics = OrderedDict(((i, {"counts" : OrderedDict(), "stats" : OrderedDict()}) for i in n_classes))
        for i in n_classes:
            loc = matrix[imap[i]][imap[i]]
            row = sum(matrix[imap[i]][:])
            col = sum([row[imap[i]] for row in matrix])
            # Get TP/TN/FP/FN
            tp  = loc
            fp  = row - loc
            fn  = col - loc
            tn  = sigma - row - col + loc
            # Populate Counts Dictionary
            statistics[i]["counts"]["tp"]   = tp
            statistics[i]["counts"]["fp"]   = fp
            statistics[i]["counts"]["tn"]   = tn
            statistics[i]["counts"]["fn"]   = fn
            statistics[i]["counts"]["pos"]  = tp + fn
            statistics[i]["counts"]["neg"]  = tn + fp
            statistics[i]["counts"]["n"]    = tp + tn + fp + fn
            # Populate Statistics Dictionary
            statistics[i]["stats"]["sensitivity"]   = tp / (tp + fn) if tp > 0 else 0.0
            statistics[i]["stats"]["precision"]     = tp / (tp + fp) if tp > 0 else 0.0
            statistics[i]["stats"]["recall"]        = tp / (tp + fn) if tp > 0 else 0.0
            statistics[i]["stats"]["accuracy"]      = (tp + tn) / (tp + tn + fp + fn) if (tp + tn) > 0 else 0.0

        return matrix, statistics

            