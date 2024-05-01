import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn import datasets 
import matplotlib.pyplot as plt 


def sigmoid(num):
    return 1/(1+ np.exp(-1*num))

def accuracy(y_preds,y):
    return np.sum(y_preds == y)/len(y_preds)


class LogisticRegressor:

    def __init__(self, lr=0.001, num_iters = 10000 ):
        self.lr = lr 
        self.num_iters = num_iters 
        self.weights = None 
        self.bias = None 


    def fit(self,X,y):
        """
        - initialize weights and biases as zero 
        - iterate #num_iters times :
            - linear predict 
            - sigmoid prob. result
            - calculate partial derivates dw and db 
            - update weights with decaying loss 
        """
        n_samples , n_features = X.shape 
        self.weights = np.zeros(n_features)
        self.bias = 0 

        for idx in range(self.num_iters):
            linear_preds = np.dot(X,self.weights) + self.bias 
            probs = sigmoid(linear_preds)
            
            dw = (1/n_samples) * np.dot(X.T,(probs-y))
            db = (1/n_samples) * np.sum(probs-y)

            # Binary Cross-Entropy Loss
            loss = -np.mean(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9))



            self.weights -= self.lr*dw 
            self.bias -= self.lr *db
            print(f"iter# {idx} train-error: {loss} acc_metric: {accuracy([1 if prob >= 0.5 else 0 for prob in probs],y)}")

 

    def predict(self,X):
        """
        - linear predict
        - sigmoid
        - determine class based on prob. and threshold
        """
        linear_preds = np.dot(X,self.weights) + self.bias  # learned weights and bias
        probs = sigmoid(linear_preds)
        return [1 if prob >= 0.5 else 0 for prob in probs]


if __name__ == "__main__":
    
    breast_cancer_df = datasets.load_breast_cancer()
    X,y = breast_cancer_df.data, breast_cancer_df.target
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=23)

    logistic_regressor = LogisticRegressor()
    logistic_regressor.fit(X_train,y_train)
    test_preds = logistic_regressor.predict(X_test)
    print(test_preds)
    print("test accuracy : ",accuracy(test_preds,y_test))