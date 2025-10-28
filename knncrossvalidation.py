### Small Python code snippet to implement KNN cross validation###
### Created by Arijit Chakraborty ###

### Kept the downloaded dataset in folder callied 'Iris' under the parent folder structure###
### Dataset download link: https://archive.ics.uci.edu/dataset/53/iris ###

###Section to import required libraries### 
import numpy as num
import pandas as pand
import matplotlib.pyplot as plot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # Used ONLY for the Part (b) comparison

### --- A. Data Loading and Preprocessing (Part (a) - Step 2 & 3) --- ###
###Define the relative path to the dataset###
DatasetPath= 'iris/iris.data' ## -> Dataset is kept in this location##
try:
    dfiris =pand.read_csv(DatasetPath, header=None)
except FileNotFoundError:
    print(f"Error: Dataset not found at '{DatasetPath}'. Please ensure the 'iris.data' file is in the 'iris' subfolder.")
    exit()
###Features and Target###
X_features =dfiris.iloc[:, :-1].values
y_target_names =dfiris.iloc[:, -1].values
###Convert target names to integers###
unique_classes=num.unique(y_target_names)
class_to_int_map={name: i for i, name in enumerate(unique_classes)}
y_target_int=num.array([class_to_int_map[label] for label in y_target_names])
# Data Preprocessing to standardize the features using built-in StandardScaler, as allowed###
scaler=StandardScaler()
X_standardized=scaler.fit_transform(X_features)
print("\nDataset loaded and preprocessed successfully.")

### --- B. K-Fold Cross-Validation Function --- ###
###Implements K-Fold Cross-Validation logic strictly from scratch###
def k_fold_cross_validation(model,X_data,y_data,k_folds):
    ###Combine X and y into one array for synchronous shuffling###
    combinedData= num.hstack((X_data, y_data[:, num.newaxis]))   
    ###Shuffle the dataset randomly###
    num.random.shuffle(combinedData)
    ###Separate data back into features and integer labels###
    X_shuffled =combinedData[:, :-1]
    y_shuffled= combinedData[:, -1].astype(int)
    totalsamples = len(X_shuffled)   
    ####Calculate fold size###
    foldsize =totalsamples // k_folds    
    foldaccuracyscores = []
    ###Determine indices for each fold###
    FoldIndices_List = []
    for i in range(k_folds):
        start_index=i * foldsize
        end_index=(i + 1) * foldsize         
        if i == k_folds - 1:
            end_index=totalsamples        
        FoldIndices_List.append(num.arange(start_index, end_index))

    ###Loop through each fold for training and validation###
    for currentfoldindex in range(k_folds):        
        ###Current fold is the validtion set###
        validation_Indices =FoldIndices_List[currentfoldindex]
        ###Remaining k-1 folds are the training set###
        trainingIndices = num.concatenate(
            [FoldIndices_List[j] for j in range(k_folds) if j != currentfoldindex]
        )
        X_train, y_train = X_shuffled[trainingIndices],y_shuffled[trainingIndices]
        X_val, y_val = X_shuffled[validation_Indices],y_shuffled[validation_Indices]

        ###Train the model (must be cloned for a fresh start each fold)###
        model_clone =KNeighborsClassifier(n_neighbors=model.n_neighbors)
        model_clone.fit(X_train, y_train)
        ###Predict and calculate accuracy######
        y_pred= model_clone.predict(X_val)
        current_accuracy= accuracy_score(y_val, y_pred)
        foldaccuracyscores.append(current_accuracy)
    ####Print results only once for the primary run, but return all scores###
    if model.n_neighbors == 5 and k_folds == 5:
        for idx, acc in enumerate(foldaccuracyscores):
            print(f"  Fold {idx + 1}/{k_folds} - Accuracy: {acc:.4f}")
    return foldaccuracyscores 

###KNeighborsClassifier from Scikit-learn###
kNN_Model = KNeighborsClassifier(n_neighbors=5) 
###Run the Cross-Validation with k=5###
print(f"\nStarting primary 5-Fold Cross-Validation for n_neghbors=5...")
k_fold_CV_scores =k_fold_cross_validation(kNN_Model, X_standardized, y_target_int, k_folds=5)
# --- C. Analysis and Comparison ---

###Model Performance Evaluation (K-Fold Results)###
###Calculate mean accuracy###
mean_accuracy_kFold =num.mean(k_fold_CV_scores)
###Calculate standard deviation###
stdDev_accuracy_kFold =num.std(k_fold_CV_scores)
print("K-Fold Cross-Validation (k=5) Results:")
print(f"  Individual Accuracies: {k_fold_CV_scores}")
print(f"  Mean Accuracy (Robust Estimate): {mean_accuracy_kFold:.4f}")
print(f"  Standard Deviation (Performance Spread): {stdDev_accuracy_kFold:.4f}")
###Comparison with Simple Train-Test Split###
simplesplitresults = []
numberrepeats = 5
splitratio = 0.20 # 80/20 split
print(f"\nComparison with {numberrepeats} repeated 80/20 Train-Test Splits:")

for i in range(numberrepeats):
    ###Use built-in train_test_split###
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_standardized, y_target_int, test_size=splitratio, random_state=i, shuffle=True
    )
    ###Train and evaluate the model for this split###
    modelcloneSplit = KNeighborsClassifier(n_neighbors=kNN_Model.n_neighbors)
    modelcloneSplit.fit(X_train_split, y_train_split)
    y_pred_split = modelcloneSplit.predict(X_test_split)
    # Calculate accuracy
    currentsplitaccuracy= accuracy_score(y_test_split, y_pred_split)
    simplesplitresults.append(currentsplitaccuracy)
    print(f"  Split {i+1}/{numberrepeats} (random_state={i}): Accuracy: {currentsplitaccuracy:.4f}")
print(f"Simple Split Accuracies: {simplesplitresults}")


### --- D. Visualization  --- ###
###Created a single figure with three subplots###
fig, axes = plot.subplots(1, 3, figsize=(20, 7)) 
fig.suptitle('\nModel Performance and Hyperparameter Tuning Analysis', fontsize=16)
###Determine a consistent Y-axis range for visual comparison (Plots 1 and 2)###
y_min_data =min(num.min(k_fold_CV_scores), num.min(simplesplitresults))
y_max_data =max(num.max(k_fold_CV_scores), num.max(simplesplitresults))
y_min = num.floor(y_min_data * 10) / 10 - 0.05
y_max = min(1.0, num.ceil(y_max_data * 10) / 10 + 0.05)
###Box Plot for K-Fold Scores (First Plot)###
ax1 = axes[0]
ax1.boxplot(k_fold_CV_scores, patch_artist=True, boxprops=dict(facecolor='skyblue'))
ax1.set_title('5-Fold CV: Distribution of Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xticks([1], ['5-Fold CV Scores'])
ax1.grid(axis='y', linestyle='--')
ax1.set_ylim(y_min, y_max)
###Strip Plot for Repeated Train-Test Splits (Second Plot)###
ax2 =axes[1]
splitlabels =[f'Split {i+1}' for i in range(numberrepeats)]
ax2.scatter(splitlabels, simplesplitresults, color='red', s=100, zorder=3, label='Single Split Score')
ax2.plot(splitlabels, simplesplitresults, linestyle='--', color='gray', alpha=0.5, label='Score Trend') 
ax2.set_title('Repeated 80/20 Splits: Variability')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Random Split Instance')
ax2.grid(axis='y', linestyle='--')
ax2.legend()
ax2.set_ylim(y_min, y_max)
### --- E. Bonus Challenge Hyperparameter Tuning --- ###
### Use Cross-Validation for Hyperparameter Tuning: Loop through n_neighbors=1 to 25###
krange =range(1, 26)
ktunmeanAccuracies = []
print("\nHyperparameter Tuning (n_neighbors=1 to 25)")
###Loop through each k value (n_neighbors)###
for k_value in krange:
    ###Instantiate the model with the current k_value###
    current_model = KNeighborsClassifier(n_neighbors=k_value)   
    ###Run 5-fold cross-validation using the custom function###
    cv_scores =k_fold_cross_validation(current_model, X_standardized, y_target_int, k_folds=5)    
    mean_accuracy =num.mean(cv_scores)
    ktunmeanAccuracies.append(mean_accuracy)
    print(f"k = {k_value:2}: Mean CV Accuracy = {mean_accuracy:.4f}")
###Find the optimal k###
optimalkindex = num.argmax(ktunmeanAccuracies)
optimalkvalue = krange[optimalkindex]
optimalaccuracy = ktunmeanAccuracies[optimalkindex]
print(f"\nOptimal n_neighbors: k = {optimalkvalue} (Mean Accuracy: {optimalaccuracy:.4f})")
###Tuning Plot (Third Plot)###
ax3 = axes[2]
ax3.plot(krange, ktunmeanAccuracies, marker='o', linestyle='-', color='indigo')
ax3.set_title('Hyperparameter Tuning')
ax3.set_xlabel('Number of Neighbors (k)')
ax3.set_ylabel('Mean 5-Fold CV Accuracy')
ax3.set_xticks(krange[::2]) # Show every other k value on the axis
ax3.grid(axis='both', linestyle='--')
ax3.legend()
####Final Display --- ###
plot.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=2.0)
plot.show()
#################################################
