import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")

class Perceptron():
    def __init__(self, alpha=(10**-4),epoch=1000):
        self.feature = np.array([[1,2,1,2,3,3,4,2,5,3], [1,1,3,2,1,3,2,4,1,4]])
        self.y = np.array([0,0,0,0,0,1,1,1,1,1])
        self.m = len(self.y) # no of examples
        self.X = np.vstack((np.ones(self.m), self.feature)).T
        self.n = self.X.shape[1] # no of features
        self.parameters = np.zeros(self.n)
        self.alpha = alpha
        self.epoch = epoch
    
    def hypothesis(self, x):
        return 1 if (self.parameters @ x) >= 0 else 0

    def error_sum(self):
        error_sums = np.zeros_like(self.parameters)
        for i in range(self.m):
            error_sums+=(self.y[i] - self.hypothesis(self.X[i]))*self.X[i]
        return self.alpha * error_sums
    
    def update(self):
        #self.plot()
        for i in range(self.epoch):
            self.parameters +=  self.error_sum()
            #self.plot()

    def plot(self):
    # separate classes
        X0 = self.X[self.y == 0] # circles
        X1 = self.X[self.y == 1] # crosses

    # plot points
        plt.scatter(X0[:,1], X0[:,2], label="Class 0", marker='X')
        plt.scatter(X1[:,1], X1[:,2], label="Class 1", marker='o')

    # decision boundary: θ0 + θ1*x1 + θ2*x2 = 0
        x_vals = np.linspace(min(self.X[:,1]), max(self.X[:,1]), 100)

        if self.parameters[2] != 0:
            y_vals = -(self.parameters[0] + self.parameters[1]*x_vals) / self.parameters[2]
            plt.plot(x_vals, y_vals, label="Decision Boundary")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.title("Perceptron Visualization")
        plt.show()


p = Perceptron(epoch=60)
p.update()
p.plot()