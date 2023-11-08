import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            for i in range(X.shape[0]):
                xi = X[i]
                yi = y[i]
                update = self.learning_rate * (yi - self.predict(xi))
                self.weights += update * xi
                self.bias += update

    def predict(self, X):
        return 1 if (np.dot(X, self.weights) + self.bias) > 0 else 0

def plot_decision_boundary(X, y, classifier, title):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.array([classifier.predict(np.array([xxi, yyi])) for xxi, yyi in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Accent, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Dark2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

# Load the Iris dataset
iris = load_iris()
X = iris.data[:100, :2]  # We use only the first two features and first 100 samples for visualization
y = iris.target[:100]  # Only two classes (0 and 1) for binary classification

# Create a Perceptron instance
perceptron = Perceptron(learning_rate=0.1, n_iterations=100)

# Train the Perceptron on the data
perceptron.fit(X, y)

# Visualize the decision boundary of the Perceptron
plot_decision_boundary(X, y, perceptron, "Perceptron Decision Boundary")
