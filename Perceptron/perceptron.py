import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self,learning_rate,epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                y_desired = y[i]

                y_pred = self.predict(x)

                error = y_desired - y_pred

                self.weights = self.weights + (self.learning_rate * error * x)
                self.bias = self.bias + (self.learning_rate * error)
    
    def predict(self, x):
        net = np.dot(self.weights, x) + self.bias
        return np.where(net<0, 0, 1)



    