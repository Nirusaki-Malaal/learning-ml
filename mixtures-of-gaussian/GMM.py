import numpy as np
import sys,os
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../K-Means-Clustering"))
from k_means_clustering import Kmeans
class GMM:
    def __init__(self, cluster, epoch=1000):
        self.X = np.array([
            # Cluster 1 (around [2, 2])
            [2.1, 1.9], [1.8, 2.2], [2.0, 2.0], [2.3, 1.7], [1.9, 2.1],
            [2.2, 2.3], [1.7, 1.8], [2.4, 2.1], [1.8, 1.9], [2.1, 2.2],
            [1.9, 1.7], [2.0, 2.4], [2.3, 2.0], [1.8, 1.8], [2.2, 1.9],
            [1.7, 2.1], [1.9, 2.3], [2.1, 1.8], [2.4, 2.2], [1.8, 2.0],
            [2.0, 1.9], [2.2, 2.1], [2.3, 2.3], [1.9, 1.8], [1.7, 2.0],
            [2.1, 2.4], [1.8, 1.7], [2.0, 2.1], [2.2, 1.8], [2.3, 1.9],
            [1.9, 2.2], [2.1, 2.0], [1.8, 2.3], [2.0, 2.0], [2.2, 2.2],
            [1.7, 1.9], [2.3, 2.1], [1.9, 2.0], [2.1, 1.7], [1.8, 2.4],
            [2.0, 1.8], [2.2, 2.0], [2.4, 2.3], [1.9, 1.9], [1.7, 2.2],
            [2.1, 2.3], [1.8, 2.1], [2.0, 1.7], [2.2, 2.4], [2.3, 1.8],

            # Cluster 2 (around [8, 8])
            [8.1, 7.9], [7.8, 8.2], [8.0, 8.0], [8.3, 7.7], [7.9, 8.1],
            [8.2, 8.3], [7.7, 7.8], [8.4, 8.1], [7.8, 7.9], [8.1, 8.2],
            [7.9, 7.7], [8.0, 8.4], [8.3, 8.0], [7.8, 7.8], [8.2, 7.9],
            [7.7, 8.1], [7.9, 8.3], [8.1, 7.8], [8.4, 8.2], [7.8, 8.0],
            [8.0, 7.9], [8.2, 8.1], [8.3, 8.3], [7.9, 7.8], [7.7, 8.0],
            [8.1, 8.4], [7.8, 7.7], [8.0, 8.1], [8.2, 7.8], [8.3, 7.9],
            [7.9, 8.2], [8.1, 8.0], [7.8, 8.3], [8.0, 8.0], [8.2, 8.2],
            [7.7, 7.9], [8.3, 8.1], [7.9, 8.0], [8.1, 7.7], [7.8, 8.4],
            [8.0, 7.8], [8.2, 8.0], [8.4, 8.3], [7.9, 7.9], [7.7, 8.2],
            [8.1, 8.3], [7.8, 8.1], [8.0, 7.7], [8.2, 8.4], [8.3, 7.8]
        ])
        self.j = cluster # total of cluster
        self.kmeans = Kmeans(self.X, self.j)
        self.kmeans.train(verbose=False)
        self.epoch = epoch
        ## parameter initalization for best results using kmeans algorithm
        self.mu = self.kmeans.centroids.copy() 
        self.phi = np.zeros(self.j)
        # # len(C[ C == j]) / X.shape[0]
        # C = kmeans.C
        # for i in range(self.j):
        #     self.phi[i] = len(C[C==i]) / kmeans.X.shape[0]

        self.phi = np.bincount(self.kmeans.C) / len(self.kmeans.C)
        # j=1 
        self.sigma = np.zeros((self.j , self.X.shape[1], self.X.shape[1]))
        for j in range(self.j):
            self.sigma[j] = ((self.X[self.kmeans.C ==j]-self.mu[j]).T @(self.X[self.kmeans.C ==j]-self.mu[j])) /np.count_nonzero(self.kmeans.C == j)
        self.sigma += 1e-8 * np.eye(self.X.shape[1]) # epsilon stablization

    def multivariate_gaussian(self,x, sigma , mu): # P(xi | zi = j) ~ N(u,E)
        return  (1/(((2*np.pi)**(self.X.shape[1]*0.5))*(np.linalg.det(sigma)**0.5))) * np.exp(-0.5 * np.inner((x-mu), np.linalg.inv(sigma) @ (x-mu)))
        # just implement # p(x) = (1 / ((2π)^(n/2) * |Σ|^(1/2))) * exp(-1/2 * (x-μ)^T Σ^(-1) (x-μ))
    
    def train(self, verbose=True):
        W = np.zeros((self.X.shape[0], self.j))
        old_loss = -np.inf
        for _ in range(self.epoch):
            #  E-Step
            for j in range(self.j):
                for i in range(self.X.shape[0]):
                    numerator = self.multivariate_gaussian(self.X[i], self.sigma[j], self.mu[j]) * self.phi[j]
                    denominator = 0
                    for l in range(self.j):
                        denominator+=self.multivariate_gaussian(self.X[i], self.sigma[l], self.mu[l]) * self.phi[l]
                    W[i,j] = numerator/denominator
            # M-Step
            
            for j in range(self.j): # updating phi
                sum_phi = 0
                sum_mu = np.zeros(self.mu.shape[1])
                sum_sigma = np.zeros((self.X.shape[1],self.X.shape[1]))
                for i in range(self.X.shape[0]):
                    sum_phi+=W[i,j]
                    sum_mu += W[i,j] * self.X[i]
                    sum_sigma += W[i,j] * np.outer((self.X[i] - self.mu[j]) ,(self.X[i] - self.mu[j]))
                self.phi[j] = sum_phi/self.X.shape[0]
                self.mu[j] = sum_mu/sum_phi
                self.sigma[j] = sum_sigma/sum_phi
                self.sigma[j] += 1e-6 * np.eye(self.X.shape[1])
        
            new_loss = self.log_likelyhood()
            if np.abs(new_loss -old_loss) < 1e-4:
                print("converged")
                break
            old_loss = new_loss
            if(verbose):
                print(f"Loss : {self.log_likelyhood():.4f}")
    
    def log_likelyhood(self):
        likelyhood = 0
        for i in range(self.X.shape[0]):
            temp =0
            for j in range(self.j):
               temp+= self.phi[j] * self.multivariate_gaussian(self.X[i], self.sigma[j], self.mu[j]) 
            likelyhood += np.log(temp)
        return likelyhood
    
    def predict(self, x):
        W = np.zeros((self.j)) # [1,2,3]
        for j in range(self.j):
            numerator = self.multivariate_gaussian(x, self.sigma[j], self.mu[j]) * self.phi[j]
            denominator = 1e-8
            for l in range(self.j):
                denominator+=self.multivariate_gaussian(x, self.sigma[l], self.mu[l]) * self.phi[l]
            W[j] = numerator/denominator
        return W

    def visualize(self, inputGiven=False):
        plt.scatter(self.X[:,0], self.X[:,1], c=self.kmeans.C, s=30)
        plt.scatter(self.kmeans.centroids[:,0],self.kmeans.centroids[:,1],marker='X',s=30)
        plt.scatter(self.mu[:,0], self.mu[:,1],marker='*', s=300 )
        x = np.linspace(np.min(self.X[:,0]-1), np.max(self.X[:,0]+1), 200) ## start end no. of points
        y = np.linspace(np.min(self.X[:,1]-1), np.max(self.X[:,1]+1), 200)
        x, y = np.meshgrid(x,y)
        grid = np.c_[x.ravel(), y.ravel()] ## add this column wise
        Z = np.zeros(x.shape)
        for j in range(self.j):
            temp_sigma = self.sigma[j] * 1.5
            for i in range(grid.shape[0]):
                Z.ravel()[i] += (
                    self.phi[j] *
                    self.multivariate_gaussian(
                        grid[i],
                        temp_sigma,
                        self.mu[j]
                    )
        )
        Z = Z.reshape(x.shape)
        plt.contour(x, y, Z)
        #if(inputGiven):
          #  plt.scatter()
        #plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    gmm = GMM(cluster=2)
    gmm.train()
    gmm.visualize()
    example = eval(input("enter your example pair (x1,x2)"))
    probs = gmm.predict(example)
    for i, p in enumerate(probs):
        print(f"Cluster {i}: {p:.3f}")