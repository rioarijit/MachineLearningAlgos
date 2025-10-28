### Small Python code snippet to implement kNN###
### Created by Arijit Chakraborty ###

### Dataset download link: https://archive.ics.uci.edu/dataset/53/iris ###

###Section to import required libraries### 
import numpy as num
import matplotlib.pyplot as plot
from collections import Counter
import random
##import os

###Function to read the IRIS data###
def loadirisdata(file_path='iris/iris.data'):
    X = []
    y = []
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    with open(file_path, 'r') as file:
        for line in file:
            line =line.strip()
            if not line:
                continue
            partsOfLine =line.split(',')
            features =[float(p) for p in partsOfLine[:4]]
            className= partsOfLine[4]
            X.append(features)
            y.append(label_map[className])            
    return num.array(X), num.array(y)

###Function to split the data to test and training featues and lebels##
def traintestdatasplit(X, y, test_size=0.2, seed=42):
    random.seed(seed)
    num.random.seed(seed)
    unique_classes =num.unique(y)
    train_indices = []
    test_indices = []
    for c in unique_classes:
        classIndices= num.where(y==c)[0]
        n_test =int(len(classIndices) * test_size)      
        num.random.shuffle(classIndices)      
        test_indices.extend(classIndices[:n_test])
        train_indices.extend(classIndices[n_test:])
    num.random.shuffle(train_indices)
    num.random.shuffle(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

###Method to load the data to perform split and standatdise the same###
def preparedata(file_path='iris/iris.data'):
    X, y = loadirisdata(file_path)
    X_train, X_test, y_train, y_test =traintestdatasplit(X, y, test_size=0.2, seed=42)
    mean = num.mean(X_train, axis=0)
    std = num.std(X_train, axis=0)
    X_train_scaled =(X_train - mean)/(std + 1e-8)
    X_test_scaled =(X_test - mean)/(std + 1e-8)
    return X_train_scaled, X_test_scaled, y_train, y_test, X, y

###Defining the CLASS for KNEARESTNEIGHBORS###
class KNearestNeighbors:
    ###initialising self - model with k value###
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
 
    ###Training method###
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
 
    ###Calculate Euclidian distance###
    def euclideandistance(self, p, q):
        return num.sqrt(num.sum((q - p)**2))

    ###Classify a single test sample###
    def predictsample(self, x_test_sample):
        ## calculates the distance from the test sample to all training sample ##
        distances = [
            self.euclideandistance(x_train, x_test_sample)
            for x_train in self.X_train
        ]
        k_indices = num.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    ###Prediction Method### 
    def predict(self, X_test):
        predictions = [self.predictsample(x) for x in X_test]
        return num.array(predictions)

###Method for calcutating mmodel accuracy
def calculateaccuracy(y_true, y_pred):
    return num.sum(y_pred == y_true) / len(y_true)

###This method is for tuning the hyperparameter###
def tuneevaluate(X_train, X_test, y_train, y_test, k_values):
    accuracy_results = []
    for k in k_values:
        knn = KNearestNeighbors(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = calculateaccuracy(y_test, y_pred)
        accuracy_results.append(accuracy)
    return k_values, accuracy_results

###method to display the decision boundry for 2D plot##
def plotboundary(ax, k_value, X, y, features, title_suffix):
    knn = KNearestNeighbors(k=k_value)
    knn.fit(X, y) 
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = num.meshgrid(num.arange(x_min, x_max, 0.02), num.arange(y_min, y_max, 0.02))
    X_grid = num.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(X_grid)
    Z = Z.reshape(xx.shape) 
    cmap_light = plot.cm.get_cmap('Pastel1', 3)
    cmap_bold = plot.cm.get_cmap('Set1', 3)
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)   
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    ax.set_title(f'Decision Boundary \n {title_suffix}')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])  
    classes = ['Setosa', 'Versicolor', 'Virginica']
    ax.legend(handles=scatter.legend_elements()[0], labels=classes, title="Classes")

if __name__ == '__main__':
    X_train_s, X_test_s, y_train, y_test, X_all, y_all = preparedata()
    print(f"Data Prepared: {len(X_train_s)} training samples, {len(X_test_s)} testing samples.")
    ## Part (a) and (b): Hyperparameter Tuning and /Evaluation ##
    ###Assigning multiple k values###
    k_values = [1, 3, 5, 7, 11, 15]
    k_results, accuracy_results = tuneevaluate(X_train_s, X_test_s, y_train, y_test, k_values)
    ## Display tabular results###
    print("Results - Accuracy vs k):")
    print("{:<5} {:<10}".format("k", "Accuracy"))
    print("-" * 15)
    for k, acc in zip(k_results, accuracy_results):
        print(f"{k:<5} {acc:<10.4f}")   
    ###Display optimal k###
    optimal_k = k_results[num.argmax(accuracy_results)]
    print(f"\nOptimal k: {optimal_k}\n")
    ###Part (b): Visualisation###
    # 1. Prepare 2D data for decision boundaries
    X_2d = X_all[:, 2:4]
    y_2d = y_all
    features_2d_names = ['Length (cm)', 'Width (cm)']
    ###Create 3 subplots###
    fig, axes = plot.subplots(1, 3, figsize=(18, 5))
    plot.suptitle('k-NN Evaluation and Decision Boundaries', fontsize=16)
    ###Plot for accuracy vs k-value###
    ax = axes[0]
    ax.plot(k_results, accuracy_results, marker='o', linestyle='-', color='red')
    ax.set_title('Accuracy vs k-Value')
    ax.set_xlabel('k-Value')
    ax.set_ylabel('Test Set Accuracy')
    ax.set_xticks(k_values)
    ax.grid(True, linestyle='--', alpha=0.6)
    ###Plot for decision boundary for k=1###
    plotboundary(
        ax=axes[1], k_value=1, X=X_2d, y=y_2d, features=features_2d_names, 
        title_suffix='High Variance (k=1)'
    )
    ###Plot for decision boundary for k=15###
    plotboundary(
        ax=axes[2], k_value=15, X=X_2d, y=y_2d, features=features_2d_names, 
        title_suffix='High Bias (k=15)'
    )
    plot.tight_layout(rect=[0, 0, 1, 0.96])
    plot.show() ###Display the final plots###
