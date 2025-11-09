# 1. The 'Mall_Customers.csv' dataset is located in the same directory as this script #####
# 2. Only the two features 'Annual Income (k$)' and 'Spending Score (1-100)' are used #####
# 3. Feature standardization is performed manually (no sklearn utilities used) #####
# 4. K-Means and Silhouette Score are implemented entirely from scratch #####

### Section to Import Required Libraries ###
import numpy as num
import pandas as pan
import matplotlib.pyplot as plot

### Helper Function for Manual Standardization ###
### Manually standardize features to zero mean and unit variance ###
### This ensures both dimensions contribute equally to the Euclidean distance ###
def standardize(X):
    mean = num.mean(X, axis=0)
    std = num.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

### CLASS FOR K-MEANS IMPLEMENTATION ###
class KMeansScratch:
    ### Initializing Self ###
    def __init__(self, k=5, max_iters=100, tolerance=1e-4, init='random'):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.init = init
        self.centroids = None
        self.labels_ = None
    ### Method to Initialize Centroids using Random or K-Means++ Method ###
    def initializecentroids(self, X):
        if self.init == 'random':
            indices = num.random.choice(X.shape[0], self.k, replace=False)
            return X[indices]
        elif self.init == 'kmeans++':
            centroids = [X[num.random.randint(0, X.shape[0])]]
            for _ in range(1, self.k):
                dist_sq = num.min([num.linalg.norm(X - c, axis=1) ** 2 for c in centroids], axis=0)
                probs = dist_sq / num.sum(dist_sq)
                cumulative_probs = num.cumsum(probs)
                r = num.random.rand()
                for i, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids.append(X[i])
                        break
            return num.array(centroids)
        else:
            raise ValueError("Invalid initialization method. Use 'random' or 'kmeans++'.")
    ### Method to Train the K-Means Model using Iterative Optimization ###
    def fit(self, X):
        self.centroids = self.initializecentroids(X)
        for iteration in range(self.max_iters):
            ### Assign each point to the nearest centroid ###
            distances = num.array([[num.linalg.norm(x - c) for c in self.centroids] for x in X])
            labels = num.argmin(distances, axis=1)
            ### Recompute Centroids ###
            new_centroids = num.array([
                X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else self.centroids[i]
                for i in range(self.k)
            ])
            ### Check for Convergence (based on tolerance) ###
            if num.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                break
            self.centroids = new_centroids
        self.labels_ = labels
        return self
    ### Method to Predict Cluster for New Data Points ###
    def predict(self, X):
        distances = num.array([[num.linalg.norm(x - c) for c in self.centroids] for x in X])
        return num.argmin(distances, axis=1)
    ### Method to Calculate Within-Cluster Sum of Squares (WCSS) ###
    def inertia(self, X):
        return num.sum([num.linalg.norm(X[i] - self.centroids[self.labels_[i]]) ** 2 for i in range(len(X))])


### METHOD FOR SILHOUETTE SCORE CALCULATION ###
### Manually compute the average silhouette score for cluster evaluation ###
### Higher silhouette scores indicate better-defined clusters ###
def silhouettescore(X, labels):
    unique_labels = num.unique(labels)
    n = len(X)
    silhouette_vals = []
    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == l] for l in unique_labels if l != labels[i]]
        ### Intra-Cluster Distance (a) ###
        a = num.mean([num.linalg.norm(X[i] - x) for x in same_cluster]) if len(same_cluster) > 1 else 0
        ### Nearest Other-Cluster Distance (b) ###
        b = num.min([num.mean([num.linalg.norm(X[i] - x) for x in cluster]) for cluster in other_clusters])
        silhouette_vals.append((b - a) / max(a, b))
    return num.mean(silhouette_vals)


### MAIN EXECUTION BLOCK ###
if __name__ == "__main__":
    ### --- Data Loading and Preprocessing for Mall Customers Dataset ###
    ### The dataset must be in the same directory as this file ###
    try:
        df = pan.read_csv("Mall_Customers.csv")
    except FileNotFoundError:
        print("ERROR: 'Mall_Customers.csv' not found in the current directory.")
        exit()
    ### Selecting Only Two Relevant Features ###
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    ### Manual Standardization ###
    X_scaled, mean, std = standardize(X)
    ### Part (a): Training and Evaluation for Different K Elbow Method ###
    wcss = []
    K_values = range(1, 11)
    for k in K_values:
        model = KMeansScratch(k=k)
        model.fit(X_scaled)
        wcss.append(model.inertia(X_scaled))
    ### Part (b): Silhouette Score Calculation ###
    silhouettescores = []
    for k in range(2, 11):
        model = KMeansScratch(k=k)
        model.fit(X_scaled)
        score = silhouettescore(X_scaled, model.labels_)
        silhouettescores.append(score)
    ### Selecting Best K (Based on Elbow Method Observation) ###
    best_k = 5
    model = KMeansScratch(k=best_k)
    model.fit(X_scaled)
    labels = model.labels_
    ### Printing Summary of Results ###
    print("\nSUMMARY OF RESULTS")
    print(f"WCSS values for K = 1 to 10:\n{wcss}\n")
    print(f"Silhouette Scores for K = 2 to 10:\n{silhouettescores}\n")
    print(f"Best number of clusters (by Elbow): {best_k}")
    ### Part (c): Combined Visualization Page ###
    fig, axs = plot.subplots(1, 3, figsize=(18, 5))
    plot.suptitle('K-Means Clustering Analysis on Mall Customers Dataset', fontsize=14)
    ### Subplot 1: Elbow Method Plot ###
    axs[0].plot(K_values, wcss, marker='o')
    axs[0].set_title('Elbow Method - WCSS vs K')
    axs[0].set_xlabel('Number of Clusters (k)')
    axs[0].set_ylabel('WCSS')
    axs[0].grid(True)
    ### Subplot 2: Cluster Visualization ###
    for i in range(best_k):
        axs[1].scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], label=f'Cluster {i+1}')
    axs[1].scatter(model.centroids[:, 0], model.centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    axs[1].set_title(f'Customer Clusters (k={best_k})')
    axs[1].set_xlabel('Annual Income (standardized)')
    axs[1].set_ylabel('Spending Score (standardized)')
    axs[1].legend()
    ### Subplot 3: Silhouette Scores Plot ###
    axs[2].plot(range(2, 11), silhouettescores, marker='o', color='green')
    axs[2].set_title('Average Silhouette Score vs K')
    axs[2].set_xlabel('Number of Clusters (k)')
    axs[2].set_ylabel('Average Silhouette Score')
    axs[2].grid(True)
    plot.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot.show()
