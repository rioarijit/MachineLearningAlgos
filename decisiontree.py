### Small Python code snippet to implement Decision Tree###
### Created by Arijit Chakraborty ###

###kept the downloaded dataset in folder callied 'Iris' under the parent folder structure###
###Dataset download link: https://archive.ics.uci.edu/dataset/53/iris ###

###Section to import required librarries### 
import numpy as num
import pandas as pan
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap 
import random

###IRIS Data Path###
DatasetPath = 'iris/iris.data' ### -> Data Set path###
###Function to Split Data###
def customTrainTest_split(X, y, testsize=0.2, randomstate=None, stratify=None):
    if randomstate is not None:
        random.seed(randomstate)
    numofsamples =len(X)
    numtest =int(numofsamples * testsize)
    indices =list(range(numofsamples))
    random.shuffle(indices)
    testindices=indices[:numtest]
    trainindices=indices[numtest:]
    X_train=X[trainindices]
    X_test=X[testindices]
    y_train=y[trainindices]
    y_test=y[testindices]
    return X_train,X_test,y_train,y_test

###Load and prepare Dataset###
try:
    column_names = [
        'sepal length (cm)','sepal width (cm)','petal length (cm)',
        'petal width (cm)','species'
    ]
    irisdatafile = pan.read_csv(DatasetPath, header=None, names=column_names)
    ###Convert species names (strings) to numeric labels (0, 1, 2)###
    target_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    irisdatafile['target'] = irisdatafile['species'].map(target_mapping)
    X =irisdatafile.iloc[:, :4].values
    y =irisdatafile['target'].values
    featurenames= column_names[:4]
    classnames= num.array(['setosa', 'versicolor', 'virginica'])
except FileNotFoundError:
    print(f"ERROR: Dataset not found at the expected path")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()
###Apply the custom 80/20 data split###
X_train,X_test,y_train,y_test = customTrainTest_split(X, y,testsize=0.2,randomstate=42)
#Implementation of Decision Tree #
###Implementing the class for Decision tree model###
###Defining Class###
class Node:
    ###Initialising Self###
    def __init__(self,featureindex=None,threshold=None,left=None,right=None,impurity=None,value=None):
        self.featureindex = featureindex 
        self.threshold =threshold
        self.left= left
        self.right =right
        self.impurity =impurity
        self.value = value

###DecisionTreeClassifier Class###
###Defining Class##
class DecisionTreeClassifier:
    ###Initialising Self###
    def __init__(self,max_depth=None,minumofsamples_split=2):
        self.root =None
        self.max_depth = max_depth
        self.minumofsamples_split =minumofsamples_split
        self.featurenames = None
        self.classnames= None

    ####Impurity and Information Gain###
    def _calculate_gini(self, y):
        if len(y) == 0:
            return 0.0
        classcounts = {}
        for label in y:
            classcounts[label] =classcounts.get(label, 0) + 1
        impurity = 1.0
        totalsamples= len(y)
        for count in classcounts.values():
            p_i = count / totalsamples
            impurity -= p_i**2
        return impurity

    ###Calculate information gain from split###
    def gaininformethod(self, y_parent, y_left, y_right):
        impparent= self._calculate_gini(y_parent)
        N = len(y_parent)
        N_left =len(y_left)
        N_right =len(y_right)
        if N_left == 0 and N_right == 0:
            return 0.0
        impleft = self._calculate_gini(y_left)
        impright = self._calculate_gini(y_right)
        weightedimpurity = (N_left / N) * impleft + (N_right / N) * impright
        gaininformation =impparent- weightedimpurity
        return gaininformation

    ###Core Tree Building Helper Functions###
    ###Finds the best feature and threshld to split the dataset, maximizing Information Gain###
    def getbestsplit(self, X, y):
        best_gain = -1
        bestsplit = {}
        n_features = X.shape[1]
        for featureindex  in range(n_features):
            thresholds =num.unique(X[:, featureindex ])
            for threshold in thresholds:
                leftindices =num.where(X[:, featureindex ] <= threshold)[0]
                rightindices= num.where(X[:, featureindex ] > threshold)[0]
                if len(leftindices) == 0 or len(rightindices) == 0:
                    continue
                y_left, y_right = y[leftindices], y[rightindices]
                gain = self.gaininformethod(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    bestsplit['featureindex '] = featureindex 
                    bestsplit['threshold'] = threshold
                    bestsplit['leftindices'] = leftindices
                    bestsplit['rightindices'] = rightindices
                    bestsplit['gain'] = gain
        return bestsplit
    ###Return most frequesnt class in the lebels y###
    def majority_vote(self, y):
        if len(y) == 0:
            return None
        classcounts = {}
        for label in y:
            classcounts[label] = classcounts.get(label, 0) + 1
        majority_class =max(classcounts, key=classcounts.get)
        return majority_class
    ###Builds Desicion Tree###
    def buildtree(self,X,y,current_depth=0):
        numofsamples = len(X)
        if self.max_depth is not None and current_depth >= self.max_depth:
            return Node(value=self.majority_vote(y), impurity=self._calculate_gini(y))
        if numofsamples < self.minumofsamples_split:
            return Node(value=self.majority_vote(y), impurity=self._calculate_gini(y))
        if len(num.unique(y)) == 1:
            return Node(value=y[0], impurity=0.0)
        bestsplit = self.getbestsplit(X, y)
        if 'gain' not in bestsplit or bestsplit['gain'] <= 0:
            return Node(value=self.majority_vote(y), impurity=self._calculate_gini(y))
        ###Recursive Split###
        X_left, y_left = X[bestsplit['leftindices']], y[bestsplit['leftindices']]
        X_right, y_right = X[bestsplit['rightindices']], y[bestsplit['rightindices']]
        lchild= self.buildtree(X_left,y_left,current_depth + 1)
        rchild= self.buildtree(X_right,y_right,current_depth + 1)
        ###Create the decision node###
        return Node(
            featureindex =bestsplit['featureindex '],
            threshold=bestsplit['threshold'],
            left=lchild,
            right=rchild,
            impurity=self._calculate_gini(y)
        )
    ###fit and predict methods###
    ###method that initiates the recursive tree-bulding process###
    def fit(self,X,y,featurenames,classnames):
        self.featurenames= featurenames
        self.classnames= classnames
        self.root= self.buildtree(X, y)
    ###helper method to traverse the trained tree for a single test sample###
    def traversetree(self, x, node):
        if node.value is not None:
            return node.value
        feature_value = x[node.featureindex ]
        if feature_value <= node.threshold:
            return self.traversetree(x, node.left)
        else:
            return self.traversetree(x, node.right)
    ###Returns predicted class###
    def predict(self, X):
        predictions = [self.traversetree(x, self.root) for x in X]
        return num.array(predictions)

#Evaluation and Visualization###
###Calcuate accuracy###
def custom_accuracyscore(y_true,y_pred):
    correctpredict = 0
    for true, pred in zip(y_true,y_pred):
        if true == pred:
            correctpredict += 1
    return correctpredict / len(y_true)

###Model Evaluation###
print("Model Evaluation: Accuracy v Maximum Depth")
max_depths = [1,2,3,5,10,None]
results = []
for depth in max_depths:
    depth_label = f'{depth}' if depth is not None else 'No limit'
    print(f"Training model with maximum depth: {depth_label}")
    dtc = DecisionTreeClassifier(max_depth=depth,minumofsamples_split=2)
    dtc.fit(X_train,y_train,featurenames,classnames)
    y_train_pred =dtc.predict(X_train)
    y_test_pred =dtc.predict(X_test)
    train_acc = custom_accuracyscore(y_train, y_train_pred)
    test_acc = custom_accuracyscore(y_test, y_test_pred)
    results.append([depth_label,f'{train_acc:.4f}',f'{test_acc:.4f}'])
    if depth == 3:
        initial_test_accuracy = test_acc
        initial_dtc = dtc 
###Print the results table###
print("\nAccuracy Results Table for Training v Testing Accuracy:")
print(f"{'Max Depth':<10} | {'Training Accuracy':<20} | {'Testing Accuracy':<20}")
for row in results:
    print(f"{row[0]:<10} | {row[1]:<20} | {row[2]:<20}")
print(f"\nAccuracy on the test set for max_depth=3 model: {initial_test_accuracy:.4f}")

#Visualization of Tree Structure###
def print_tree_structure(node,featurenames,classnames,depth=0):
    indent = "    " * depth
    if node.value is not None:
        predicted_class_name =classnames[int(node.value)]
        print(f"{indent}Predict: {predicted_class_name} (Impurity: {node.impurity:.4f})")
    else:
        feature_name = featurenames[node.featureindex ]
        print(f"{indent}IF {feature_name} <= {node.threshold:.3f} (Impurity: {node.impurity:.4f})")
        print_tree_structure(node.left, featurenames, classnames, depth + 1)
        print(f"{indent}ELSE (i.e., {feature_name} > {node.threshold:.3f})")
        print_tree_structure(node.right, featurenames, classnames, depth + 1)

print("Visualization: Tree Structure")
print_tree_structure(initial_dtc.root, initial_dtc.featurenames, initial_dtc.classnames)
#Visualization of Decision Boundary Plot###
X_2D = X[:, [2, 3]]
featurenames_2D =[featurenames[2], featurenames[3]]
X_train_2D,X_test_2D,y_train_2D,y_test_2D =customTrainTest_split(X_2D, y, testsize=0.2, randomstate=42)
dtc_2D = DecisionTreeClassifier(max_depth=3, minumofsamples_split=2)
dtc_2D.fit(X_train_2D, y_train_2D, featurenames_2D, classnames)
###methods to plot the 2d graph###
def plot_decision_boundary(X_data,y_data, classifier,featurenames,classnames,title):
    h = 0.02   
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    x_min, x_max = num.min(X_data[:, 0]) - 0.1, num.max(X_data[:, 0]) + 0.1
    y_min, y_max = num.min(X_data[:, 1]) - 0.1, num.max(X_data[:, 1]) + 0.1
    xx, yy = num.meshgrid(num.arange(x_min, x_max, h),
                         num.arange(y_min, y_max, h))    
    X_mesh = num.column_stack([xx.ravel(), yy.ravel()])
    Z = classifier.predict(X_mesh)
    Z = Z.reshape(xx.shape)
    ###Plotting###
    plot.figure(figsize=(10, 7))
    plot.contourf(xx, yy, Z,cmap=cmap_light, alpha=0.8)
    ###lot the training points###
    plot.scatter(X_data[:, 0],X_data[:, 1], c=y_data,cmap=cmap_bold, edgecolor='k', s=40)
    plot.xlabel(featurenames[0])
    plot.ylabel(featurenames[1])
    plot.title(title)
    legend_handles = [plot.Line2D([0], [0], marker='o', color='w', label=classnames[i],
                               markerfacecolor=cmap_bold(i), markersize=8) for i in range(len(classnames))]
    plot.legend(handles=legend_handles, title="Classes")
    plot.show()

print("\nVisualization of Decision Bondary Plot")
print("\nPlot generated successfully in separate window")
###Generate the plot###
plot_decision_boundary(X_train_2D, y_train_2D, dtc_2D, featurenames_2D, classnames,
                       'Decision Boundary (Petal Length vs Petal Width, Max Depth=3)')
#################################################
