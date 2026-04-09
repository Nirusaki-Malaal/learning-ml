import numpy as np
import matplotlib.pyplot as plt

class SoftmaxRegression():
    def __init__(self, alpha=0.1, epoch=10000):
        self.feature = np.array([
                                # f1  f2   f3   f4
                                [1.2, 3.4, 5.6, 2.1], # exampless 
                                [2.3, 4.5, 1.2, 3.3],
                                [3.1, 2.2, 4.4, 1.1],
                                [5.5, 2.3, 3.3, 4.4],
                                [6.7, 3.2, 2.1, 5.5],
                                [7.8, 4.1, 3.2, 6.6],
                                [8.9, 5.5, 4.4, 7.7],
                                [2.2, 1.1, 0.5, 2.2],
                                [3.3, 2.2, 1.1, 3.3],
                                [4.4, 3.3, 2.2, 4.4],
                                [6.6, 5.5, 3.3, 5.5],
                                [7.7, 6.6, 4.4, 6.6],
                                [8.8, 7.7, 5.5, 7.7],
                                [9.9, 8.8, 6.6, 8.8],
                                [1.1, 0.9, 0.8, 1.0]])
        
        self.y = np.array([
                            0, 0, 1, 1, 2, # 0 light user , 1 moderate user , 2 heavy user 
                            2, 2, 0, 0, 1,
                            1, 2, 2, 2, 0
                            ])
        self.m = len(self.y) # no of examples
        self.num_classes = len(np.unique(self.y))
        self.X = np.c_[np.ones(self.m), self.feature]
        self.n = self.X.shape[1] # no of features
        self.parameters = np.zeros((self.n, self.num_classes))
        self.alpha = alpha
        self.epoch = epoch

    def one_hot(self):
        y_one_hot = np.zeros((self.m, self.num_classes))
        y_one_hot[np.arange(self.m), self.y] = 1
        return y_one_hot
    
    def hypothesis(self):
        z = self.X @ self.parameters   # (m × k)
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cost(self):
        y_hat = self.hypothesis()
        y_one_hot = self.one_hot()

        loss = -np.sum(y_one_hot * np.log(y_hat + 1e-9)) / self.m
        return loss


    def update(self):
        y_hat = self.hypothesis()# (m × k)
        y_one_hot = self.one_hot()             # (m × k)
    
        gradient = (self.X.T @ (y_hat - y_one_hot)) / self.m   # (n × k)
        
        self.parameters = self.parameters - self.alpha * gradient

    def train(self):
        costs = []

        for i in range(self.epoch):
            self.update()
            cost = self.cost()
            costs.append(cost)

            if i % 100 == 0:
                print(f"Epoch {i}, Cost: {cost}")

        self.plot(costs)

    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        z = X @ self.parameters
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return np.argmax(probs, axis=1)
    
    def accuracy(self):
        preds = self.predict(self.feature)
        return np.mean(preds == self.y)

    def plot(self, costs=None):
        plt.figure(figsize=(12,5))

        # -------- Cost Plot --------
        if costs is not None:
            plt.subplot(1,2,1)
            plt.plot(costs)
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.title("Cost vs Epoch")

        # -------- Decision Boundary --------
        plt.subplot(1,2,2)

        x_min, x_max = self.feature[:, 0].min()-1, self.feature[:, 0].max()+1
        y_min, y_max = self.feature[:, 1].min()-1, self.feature[:, 1].max()+1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # baaki features fix
        f3_mean = np.mean(self.feature[:, 2])
        f4_mean = np.mean(self.feature[:, 3])

        grid = np.c_[xx.ravel(), yy.ravel(),
                     np.full(xx.size, f3_mean),
                     np.full(xx.size, f4_mean)]

        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)
        region_levels = np.arange(self.num_classes + 1) - 0.5
        boundary_levels = region_levels[1:-1]

        plt.contourf(xx, yy, Z, levels=region_levels, alpha=0.3)
        if len(boundary_levels) > 0:
            plt.contour(xx, yy, Z, levels=boundary_levels, colors='k', linewidths=1.5)

        colors = ['red', 'green', 'blue']
        for i in range(self.num_classes):
            idx = np.where(self.y == i)
            plt.scatter(self.feature[idx, 0], self.feature[idx, 1],
                        color=colors[i], label=f"Class {i}")

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary")
        plt.legend()

        plt.tight_layout()
        plt.show()

    

if __name__ == "__main__":
    model = SoftmaxRegression()
    model.train()
    print(f"Accuracy: {model.accuracy():.2%}")
