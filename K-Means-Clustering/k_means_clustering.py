import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
class Kmeans():
    def __init__(self, cluster, epoch):
        self.X = np.array([
                [1.0, 2.0], [1.5, 1.8], [2.0, 2.2], [8.0, 8.0], [8.5, 7.5], 
                [9.0, 8.5], [4.5, 5.0], [5.0, 4.5], [5.5, 5.2], [0.5, 1.0],
                [9.5, 9.0], [4.8, 5.5], [1.2, 1.9], [0.8, 2.1], [1.8, 1.6],
                [2.2, 2.5], [1.4, 2.3], [8.2, 8.4], [8.8, 7.8], [7.5, 8.1],
                [9.2, 8.8], [8.6, 9.2], [4.0, 4.8], [4.2, 5.6], [5.8, 4.2],
                [5.2, 5.8], [4.7, 4.9], [1.1, 1.7], [1.6, 2.1], [0.9, 1.2],
                [1.7, 1.9], [2.1, 2.4], [1.3, 1.5], [0.7, 1.8], [2.3, 2.1],
                [1.0, 1.3], [1.9, 2.0], [1.2, 2.4], [1.4, 1.1], [2.4, 2.3],
                [1.5, 2.2], [0.6, 1.5], [1.8, 2.3], [2.0, 1.7], [0.8, 1.4],
                [8.1, 8.5], [8.7, 8.0], [7.8, 8.2], [9.1, 8.7], [8.4, 9.0],
                [8.3, 7.9], [8.9, 8.6], [7.6, 8.4], [9.3, 8.2], [8.5, 8.8],
                [8.0, 7.7], [9.4, 9.1], [7.9, 8.9], [8.2, 8.1], [9.0, 7.6],
                [8.6, 8.5], [7.7, 7.8], [8.8, 9.3], [9.5, 8.4], [8.1, 8.9],
                [8.9, 8.1], [7.5, 8.6], [9.2, 8.0], [8.4, 8.3], [9.1, 9.2],
                [4.1, 5.1], [4.9, 4.6], [5.3, 4.8], [4.4, 5.3], [5.1, 5.4],
                [4.6, 4.2], [5.4, 4.9], [4.3, 4.5], [5.6, 5.0], [4.8, 5.1],
                [5.0, 4.3], [5.5, 5.7], [4.2, 4.9], [5.7, 4.5], [4.7, 5.6],
                [5.2, 4.7], [4.5, 4.4], [5.8, 5.3], [4.1, 4.6], [5.9, 4.1],
                [4.6, 5.5], [5.1, 4.2], [5.4, 5.6], [4.4, 4.1], [5.3, 5.5],
                [1.1, 2.2], [2.2, 1.8], [8.3, 8.7], [7.9, 7.9], [4.9, 5.4],
                [5.5, 4.4], [1.4, 1.7], [8.7, 8.9], [4.3, 5.2], [9.3, 8.6]
            ])
        self.K = cluster
        indices = np.random.choice(self.X.shape[0], size=self.K,replace=False) # replace=false means no duplicate allowed
        self.centroids = self.X[indices]
        self.epoch = epoch
        self.C = np.zeros(self.X.shape[0], dtype=int)
    def cost(self,c, u):
        loss = 0
        for i in range(self.X.shape[0]):
            example = self.X[i]
            loss += np.linalg.norm(u[c[i]]- example)**2
        return loss 


    def train(self, verbose=True):
        for epoch in range(self.epoch):
            old_centroids = self.centroids.copy()
            C = np.zeros(self.X.shape[0], dtype=int)  ## np.shape[0]
            ##l2_norm = np.linalg.norm(v)**2 ## l2 norm
            for i in range(self.X.shape[0]):
                example = self.X[i]
                C[i] = np.argmin((np.linalg.norm(self.centroids - example, axis=1))**2)
            self.C = C
            ## centroid update
            for j in range(self.K):
                numerator = np.zeros(self.X.shape[1])
                denominator = 0
                for i in range(self.X.shape[0]):
                    numerator+= int(C[i] == j) * self.X[i]
                    denominator+= int(C[i] == j)
                if denominator > 0:
                    self.centroids[j] = numerator / denominator
                else:
                    idx = np.random.choice(self.X.shape[0])
                    self.centroids[j] = self.X[idx]
            if np.allclose(old_centroids, self.centroids):
                print("Converged")
                break
            if verbose:
                loss = self.cost(C,self.centroids)
                print(f"Running Iteration:{epoch+1} Loss:{loss:.4f}")
    def visualize(self):
        plt.scatter(self.X[:,0], self.X[:,1], c=self.C, s=100) ## points plot s = marker size bigger number bigger dots
        plt.scatter(self.centroids[:,0],self.centroids[:,1],marker='X',s=300)
        plt.show()

    def predict(self, x):
        x = np.array(x)
        cluster_label = np.argmin((np.linalg.norm(self.centroids - x, axis=1))**2)
        print(cluster_label)
        plt.scatter(self.X[:,0], self.X[:,1], c=self.C, s=100) ## points plot s = marker size bigger number bigger dots
        plt.scatter(self.centroids[:,0],self.centroids[:,1],marker='X',s=300)
        plt.scatter(x[0], x[1],c=[cluster_label], marker="*", s=400)
        plt.show()

if __name__ == "__main__":
    kmeans = Kmeans(3, 1000)
    kmeans.train()
    kmeans.visualize()
