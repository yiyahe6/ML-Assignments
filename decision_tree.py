import numpy as np
import pandas as pd
from typing import OrderedDict
from random import randrange
import math 

def preprocess_data(df):
    new_columns = df.columns.values
    for n, column in enumerate(new_columns):
        new_columns[n] = df.columns.values[n] + " {" + df.iloc[0][column] + "}"
    new_df = df.drop(df.index[0])
    new_df.columns = new_columns
    return new_df.apply(pd.to_numeric)  # convert strings to numbers

def normalize(df):
    """
    features_name = df.columns[:-1]
    """
    result = df.copy()
    for feature_name in df.columns[:-1]:
        df[feature_name] = df[feature_name].astype(float)
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def cross_validation_split(dataset, folds=5):
    dataset_split = []
    df_copy = dataset
    fold_size = int(df_copy.shape[0] / folds)
    training_dataset = []
    testing_dataset = []
    # for loop to save each fold
    for i in range(folds):
        fold = []
        # while loop to add elements to the folds
        while len(fold) < fold_size:
            # select a random element
            r = randrange(df_copy.shape[0])
            # determine the index of this element 
            index = df_copy.index[r]
            # save the randomly selected line 
            fold.append(df_copy.loc[[index]])
            # delete the randomly selected line from
            # dataframe not to select agai
            df_copy = df_copy.drop(index)
        # save the fold     
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


class BinaryTreeNode:
    
    def __init__(self, x, y, idxs, min_leaf=5):
        self.x = x 
        self.y = y
        self.idxs = idxs 
        self.min_leaf = min_leaf
        self.decision_method = decision_method
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.ig = float('-inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        for c in range(self.col_count): self.find_better_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = BinaryTreeNode(self.x, self.y, self.idxs[lhs], self.min_leaf)
        self.rhs = BinaryTreeNode(self.x, self.y, self.idxs[rhs], self.min_leaf)
        
    def find_better_split(self, var_idx, decision_method="sse"):
        x = self.x.values[self.idxs, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue
            curr_score = self.sse(lhs, rhs)
            if curr_score < self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def sse(self, lhs, rhs):
        y = self.y[self.idxs]
        lhs_mean = y[lhs].mean()
        rhs_mean = y[rhs].mean()
        lhs_sse = np.square(y[lhs] - lhs_mean).sum()
        rhs_sse = np.square(y[rhs] - rhs_mean).sum()
        return lhs_sse + lhs_sse
                
    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]
                
    @property
    def is_leaf(self): 
        if self.decision_method == "ig":
            return self.ig == float('-inf') 
        else:
            return self.score == float('inf')                

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)

class DecisionTreeRegressor:
  def fit(self, X, y, min_leaf = 5, decision_method="sse"):
    self.dtree = BinaryTreeNode(X, y, np.array(np.arange(len(y))), min_leaf, decision_method)
    return self
  
  def predict(self, X):
    return self.dtree.predict(X.values)

dataset = pd.read_csv('housing.csv', names=["CRIM", "ZN", "INDUS",  "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "class"]) 
dataset = pd.read_csv('mushroom.csv', names=list(range(21))+ ["class"]) 
dataset = dataset[:50]
# dataset = normalize(dataset)
# training_dataset = cross_validation_split(dataset, 10)[0]
# testing_dataset = cross_validation_split(dataset, 10)[1]
# n_samples = len(dataset)
# instance = BinaryDecisionTree()
# training_data, testing_data = training_dataset[0], testing_dataset[0]
# X = training_data.drop('class', axis=1)
# y = training_data['class']
# X_test = testing_data.drop('class', axis=1)
# regressor = instance.fit(X, y)
# preds_sse = regressor.predict(X_test)
# regressor2 = instance.fit(X, y, decision_method="ig")
# preds_ig = regressor.predict(X_test)
# # print("preds_sse\n", preds_sse)
# print("preds_ig\n", preds_ig)

class MultiwayDecisionTree:
    def __init__(self, dataset):
        
        self.dataset = dataset

    def entropy(self,target_col):
        """
        Calculate the entropy of a dataset.
        The only parameter of this function is the target_col parameter which specifies the target column
        """
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy


    def InfoGain(self,data,split_attribute_name,target_name="class"):
        """
        Calculate the information gain of a dataset. This function takes three parameters:
        1. data = The dataset for whose feature the IG should be calculated
        2. split_attribute_name = the name of the feature for which the information gain should be calculated
        3. target_name = the name of the target feature. The default for this example is "class"
        """    
        #Calculate the entropy of the total dataset
        total_entropy = self.entropy(data[target_name])
        
        ##Calculate the entropy of the dataset
        
        #Calculate the values and the corresponding counts for the split attribute 
        vals,counts= np.unique(data[split_attribute_name],return_counts=True)
        
        #Calculate the weighted entropy
        Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*self.entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
        
        #Calculate the information gain
        Information_Gain = total_entropy - Weighted_Entropy
        return Information_Gain

    def build_tree(self,data,features,target_attribute_name="class",parent_node_class = None,min_leaf=1):

        #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
        
        #If all target_values have the same value, return this value
        if len(np.unique(data[target_attribute_name])) <= 1:
            return np.unique(data[target_attribute_name])[0]
        
        elif len(features) == 0:
            return parent_node_class
        elif len(data[target_attribute_name]) <= min_leaf:
            return parent_node_class
        #If none of the above holds true, grow the tree!
        else:
            #Set the default value for this node --> The mode target feature value of the current node
            parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
            #Select the feature which best splits the dataset
            item_values = [self.InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]
            
            #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
            #gain in the first run
            multiway_tree = {best_feature:{}}
            
            
            #Remove the feature with the best inforamtion gain from the feature space
            features = [i for i in features if i != best_feature]
            
            #Grow a branch under the root node for each possible value of the root node feature
            
            for value in np.unique(data[best_feature]):
                value = value
                #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
                sub_data = data.where(data[best_feature] == value).dropna()
                
                #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
                subtree = self.build_tree(sub_data,features,target_attribute_name,parent_node_class,min_leaf)
                
                #Add the sub tree, grown from the sub_dataset to the tree under the root node
                multiway_tree[best_feature][value] = subtree
            return multiway_tree    

    def predict(self,query,tree, default=1):
        """
        Prediction of a new/unseen query instance. This takes two parameters:
        1. The query instance as a dictionary of the shape {"feature_name":feature_value,...}

        2. The tree 
        """
        for key in list(query.keys()):
            if key in list(tree.keys()):
                #2.
                try:
                    result = tree[key][query[key]] 
                except:
                    return default
    
                #3.
                result = tree[key][query[key]]
                #4.
                if isinstance(result,dict):
                    return self.predict(query,result)

                else:
                    return result

    def index_predict(self,data,tree):
            #Create new query instances by simply removing the target feature column from the original dataset and 
        #convert it to a dictionary
        queries = data.iloc[:,:-1].to_dict(orient = "records") # drop target col
        #Create a empty DataFrame in whose columns the prediction of the tree are stored
        predicted = pd.DataFrame(columns=["predicted"]) 
        
        #Calculate the prediction accuracy
        for i in range(len(data)):
            predicted.loc[i,"predicted"] = self.predict(queries[i],tree) 
        return predicted
class Evaluate:
    def get_accuracy(self,data,predicted):  
        accuracy = np.sum(predicted["predicted"].reset_index(drop=True) == data["class"].reset_index(drop=True))/len(data)*100
        return accuracy
        
    def create_confusion_matrix(self,data,predicted):
        y_predicted = predicted["predicted"].reset_index(drop=True)
        y_actual = data["class"].reset_index(drop=True)
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
        # Iterate Through Classes & Compute Statistics
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

            


