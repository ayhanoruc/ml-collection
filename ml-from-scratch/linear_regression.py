import numpy as np 
from sklearn.datasets import make_regression 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 



class LinearRegressor:

    def __init__(self, lr=0.01, num_iterations=1000):
        self.lr = lr
        self.num_iterations = num_iterations
        self.weights = None 
        self.bias = None 
        self.train_error_hist = []
        self.val_error_hist = []

    def _plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.num_iterations), self.train_error_hist, label='Train Error')
        plt.plot(range(self.num_iterations), self.val_error_hist, label='Validation Error')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Train vs Validation Error')
        plt.legend()
        plt.show()

    
    def _mse(self,y_pred,y_true):
        return np.mean((y_pred-y_true)**2)

    def fit(self,X,y):
        """
        X*w + b : prediction , we want to optimize this to get best w & b possible.
        cost function: mse , based on this cost/loss function we will have dw, db gradients as update rules
        1- initialize weights and biases as zero
        for step in num_iterations:
            2- forward pass
            3- calculate gradients 
            4- update weights
            5- eval on validation
            6- plot training step
        """
    
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12)
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for step in range(self.num_iterations):
            y_train_pred = np.dot(X_train, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X_train.T, (y_train_pred - y_train))
            db = (1 / n_samples) * np.sum(y_train_pred - y_train)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            train_error = self._mse(y_train_pred, y_train)

            y_val_pred = np.dot(X_val, self.weights) + self.bias
            val_error = self._mse(y_val_pred, y_val)
            print(f"Step #{step} Train Error: {train_error} Validation Error: {val_error}")

            self.train_error_hist.append(train_error)
            self.val_error_hist.append(val_error)

        self._plot()

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
    


if __name__ == "__main__":

    # prepare the dataset
    X,y = make_regression(n_samples=1000, n_features=3, noise=10,random_state=12)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=12)

    regressor = LinearRegressor()
    regressor.fit(X_train,y_train)

    predictions = regressor.predict(X_test)

    error  = regressor._mse(predictions,y_test)
    print("test error", error)