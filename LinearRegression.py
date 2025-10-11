### Small Python code snippet to implement Linear Regression Algorithm###
### Created by Arijit Chakraborty ###

###Section to import required libraries### 
import numpy as num
import matplotlib.pyplot as plot
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

###Initialising Dataset###
housing_dataset = fetch_california_housing()
x =housing_dataset.data
y =housing_dataset.target
feature_names = housing_dataset.feature_names
###Splitting to testing and training data###
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- Part(a): Implementation of Linear Regression Model #
###Assumption: For siumplified example the dataset is 2D###
###Section to define CLASS###
###Creating CLASS definition for LINEAR REGRESSION###
class LinearRegression:
    ###initialising self###
    def __init__(self, learning_rate, n_iterations):
        self.alpha = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    ###Training method###
    def fit(self, x, y):
        m, n =x.shape
        self.weights = num.zeros(n)
        self.bias = 0
        for i in range(self.n_iterations):
            y_pre = x.dot(self.weights) + self.bias
            e =y_pre - y
            meanCost = num.mean(e**2)
            self.cost_history.append(meanCost)
            deri_w = (1/m)*x.T.dot(e)
            deri_b = (1/m)*num.sum(e)
            self.weights = self.weights - self.alpha * deri_w
            self.bias = self.bias - self.alpha * deri_b
        return self

    ###Prediction Method###    
    def predict(self, x):
        return x.dot(self.weights)+self.bias

# --- Part(b): Evaluation and Visualisation #
###Section to train the model and perform evaluation###
learningRate = 0.01
numIterations = 10000 
print("Starting Gradient Descent training...")
learningModel = LinearRegression(learning_rate=learningRate, n_iterations=numIterations)
learningModel.fit(x_train_scaled, y_train)
print("Training complete.")
y_pre_test = learningModel.predict(x_test_scaled)
###calculate MSE###
mse = num.mean((y_pre_test -y_test)**2) 
###calculate R2###
ss_total = num.sum((y_test - num.mean(y_test))**2)
ss_residual = num.sum((y_test - y_pre_test)**2)
r2 = 1 - (ss_residual / ss_total) 
###displaying results###
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print(f"Learned Bias: {learningModel.bias}")

###Plotting the graphs###
###Single image with both graphs###
plot.figure(figsize=(12,5))
plot.subplot(1,2,1)
plot.plot(range(learningModel.n_iterations), learningModel.cost_history, color='blue')
plot.title(f"Learning Curve (Cost vs. Iterations)\nAlpha={learningRate}")
plot.xlabel("Iterations")
plot.ylabel("Cost (MSE)")
plot.grid(True)
plot.subplot(1, 2, 2)
plot.scatter(y_test, y_pre_test, alpha=0.3)
min_val = min(y_test.min(), y_pre_test.min())
max_val = max(y_test.max(), y_pre_test.max())
plot.plot([min_val, max_val], [min_val, max_val], 'r--',lw=2,label='Ideal 45-degree line') 
plot.title("Actual vs. Predicted Values (Test Set)")
plot.xlabel("Actual Prices (y_test)")
plot.ylabel("Predicted Prices (y_predicted)")
plot.legend()
plot.grid(True)
plot.tight_layout()
plot.show()
