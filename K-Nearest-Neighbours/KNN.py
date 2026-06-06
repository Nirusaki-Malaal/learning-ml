import numpy as np
import matplotlib.pyplot as plt

class KNN():
    def __init__(self,X,K,Y):
        self.X =  X
        self.K = K 
        self.y = Y # labels
    def distance(self, x,p=2): ## Minkowski Distance p = 1 => Manhattan , p = 2 => Euclidean Distance. P--> infinity Cherbyshev Distance
        if (p == np.inf):
            return np.max(np.abs(x - self.X), axis=1)
        return (np.sum(np.abs(x - self.X)**p, axis=1))**(1/p)

    def predict(self, x, p=2, method="inverse"):
        distances = self.distance(x,p)
        n_i = np.argsort(distances)[:self.K] # nearest neighbours index # [2,3,4,7,3,1,0]
        distances_k =  distances[n_i] # top 5 distances
        weights = self.weights(distances_k, method=method) 
        classes = self.y[n_i] # [0,0,1,1,2,2,1]
        unique, inverse = np.unique(classes, return_inverse=True)
        votes = np.bincount(inverse,weights=weights)
        return unique[np.argmax(votes)]
    
    def weights(self, distances ,method=None):
        if method=="inverse":
            return 1 / (distances + 1e-10)
            pass
        elif method=="inverse_square":
            return 1 / (distances + 1e-10)**2
        else:
            return np.ones(distances.shape)

    def plot(self):
        plt.figure(figsize=(8, 6))
        classes = np.unique(self.y)
        for cls in classes:
            mask = self.y == cls
            plt.scatter(self.X[mask, 0],self.X[mask, 1],label=f"Class {cls}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"KNN Dataset (K={self.K})")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    X = np.array([[1,2],[2,3],[3,3],[6,7],[7,8],[8,8]])
    y = np.array([0,0,0,1,1,1])
    knn = KNN(X, K=3, Y=y)
    x = np.array([4,4])
    print("Point:", x)
    print("Prediction:", knn.predict(x, p=2, method="inverse")) 
    knn.plot()




           