import numpy as np

class logisticregression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr= learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self,X,y):
        n_samples, n_features =X.shape

        self.weights = np.zeros(n_features)
        self.bias=0

        # now gradient descent to learn the weights and bias
        for _ in range(self.n_iters):
            linear_model = np.dot(X,self.weights)+self.bias
            y_predicted = self.sigmoid(linear_model)

            # gradients i.e. the derivatives are
            dw= (1/n_samples)*np.dot(X.T, (y_predicted - y))
            db= (1/n_samples)*np.sum(y_predicted - y)

            self.weights-=self.lr*dw
            self.bias-=self.lr*db
# here decision boundary is 0.5, if the predicted value is greater than or equal to 0.5 then we will classify it as 1 otherwise 0
    def prediction(self, X):
        linear_model = np.dot(X, self.weights)+self.bias
        y_predicted = self.sigmoid(linear_model)
        return np.where(y_predicted >= 0.5,1,0)  # this means if the predicted y is greater than 0.5 then it is class 1 otherwise class 0
    
