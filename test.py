from random import shuffle
import pandas as pd
import decision_tree as dt
class DecisionTreeNode():
    
    def __init__(self, x, y, min_node):
        self.left = None
        self.right = None
        self.split_value = None
        
        # store all data points of current nodes, feature
        self.x = x
        # store all labels of original training set, label
        self.y = y
        self.min_node = min_node
        self.stop = False

    
    # given label, return pk
    def proportion(self, y):
        result = []
        labels = set(y)
        for label in labels:
            result.append(len([yi for yi in self.y if yi == label]) / len(self.y))
        return result

    
    # entropy value of current node  
    def entropy(self):
        
        pk = self.proportion(self.y)
        entropy_sum = 0
        for p in pk:
            if p != 0:
                entropy_sum += p * math.log2(p)
        return -entropy_sum
 
    
    def split_data(self, x, y, index):
        x_split = [x[i] for i in index]
        y_split = [y[i] for i in index]
        
        return x_split, y_split
    
    def get_thresholds(self, x):
        data = sorted(set(x))
        thresholds = []
        for i in range(0, len(data) - 1):
            thresholds.append((data[i] + data[i + 1])/ 2)
        
        thresholds = shuffle(thresholds)
        if len(thresholds) < 10:
            return thresholds
        return thresholds[0 : max(len(thresholds) // 40, 1)]
    
    def split_helper(self, x, y, idx, t):
        left_x, left_y = [], []
        right_x, right_y = [], []
        
        for i in range(0, len(x)):
            if x[i][idx] < t:
                left_x.append(x[i])
                left_y.append(y[i])
            else:
                right_x.append(x[i])
                right_y.append(y[i])
        return left_x, left_y,right_x,right_y
    
    ''' 
        split current node based on max info gain
    '''
    def split(self):
        if len(set(self.y)) == 1:
            self.stop = True
            return
            
        #only do split if eta value of current node is greater than eta_threashold
        if len(self.x) > self.min_node:
            current_entropy = self.entropy()
            
            node_pairs = []
            for i in range(0, len(self.x[0])):
                thresholds = self.get_thresholds([k[i] for k in self.x])
                for t in thresholds:
                    left_x, left_y, right_x, right_y = self.split_helper(self.x, self.y, i, t)
             
                    left_node = DecisionTreeNode(left_x, left_y, self.min_node)
                    right_node = DecisionTreeNode(right_x, right_y, self.min_node)
                    
                    node_pairs.append((left_node, right_node, t, i))
                    
            # find max info gain split
            max_gain = 0
            split_pair = None

            for pair in node_pairs:
                weighted_entropy = 0
                for node in pair[0:2]:
                    weighted_entropy = weighted_entropy + (1.0 * len(node.x)) / len(self.x) * node.entropy()
                gain = self.entropy() - weighted_entropy

                if gain > max_gain:
                    max_gain = gain
                    split_pair = pair

            if split_pair == None:
                self.stop = True
                return
            
            self.left = split_pair[0]
            self.right = split_pair[1]
            self.value = split_pair[2]
            self.idx = split_pair[3]

            '''
                split child node recursively
            '''
            if self.left == None:
                assert(False)
            if self.right == None:
                assert(False)
            if self.left != None:
                self.left.split()
            if self.right != None:
                self.right.split()
            
        else:
            self.stop = True
            
    '''
        given input feature vector x, output prediction by go down current node till reach a leaf
    '''
    def predict(self, x):
        if self.stop:
            return max(self.y, key = self.y.count)
        else:
            if (x[self.idx] < self.value):
                return self.left.predict(x)
            else:
                return self.right.predict(x)
        
        
class DecisionTree():
    
    def __init__(self, eta_threshold):
        self.eta_threshold = eta_threshold

    
    '''
        training function
        
        X: features
        y: corresponding labels
    '''
    def fit(self, x, y):
        self.root = None
        self.root = DecisionTreeNode(x, y, self.eta_threshold * len(x))
        self.root.split()

    
    '''
        given input feature, return prediction output.
    '''
    def predict(self, x):
        return self.root.predict(x)
folds = 10
dataset = pd.read_csv('iris.csv', names=["sepal_length", "sepal_width", "petal_length",  "petal_width", "class"]) 
training_dataset = dt.cross_validation_split(dataset, folds)[0]
testing_dataset = dt.cross_validation_split(dataset, folds)[1]
training_data, testing_data = training_dataset[0], testing_dataset[0]
X = training_data.drop('class', axis=1)
y = training_data['class']
X_test = testing_data.drop('class', axis=1)
result = DecisionTree(2).fit(X,y).predict(X_test)
print(result)