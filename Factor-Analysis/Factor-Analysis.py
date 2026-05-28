import numpy as np

class FactorAnalysis:
    def __init__(self, X, k, epoch=100):
        self.X = X
        self.K = k # no of latent factors
        self.psi =  np.diag(np.var(X, axis=0))
        self.mu = np.mean(X, axis=0)
        self.lamda = np.random.randn(self.X.shape[1],k) * 0.01
        self.epoch = epoch
    def train(self):
        for _ in range(self.epoch):
            # E-STEP
            # (Z | X) ~ (0,u) , sigma)
            mu = np.zeros((self.X.shape[0], self.K))
            sigma = np.zeros((self.X.shape[0],self.K, self.K))
            some_var = np.linalg.inv(((self.lamda @ self.lamda.T) + self.psi))
            for i in range(self.X.shape[0]):
                mu[i] = self.lamda.T @ some_var  @ (self.X[i] - self.mu)
                sigma[i] = np.eye(self.K) - self.lamda.T @ some_var @ self.lamda
            # M STEP

            self.mu = np.sum(self.X, axis=0) / self.X.shape[0]
            self.lamda = (self.X.T @ mu) @ np.linalg.inv(np.sum((sigma + mu[:, :, None] * mu[:, None , :]),axis=0))
            A = np.diag((self.X.T @ self.X - self.lamda @ mu.T @ self.X )/self.X.shape[0])
            self.psi = np.diag(A)
    
    def multivariate_gaussian(self,x, sigma , mu): # P(xi | zi = j) ~ N(u,E)
        return  (1/(((2*np.pi)**(self.X.shape[1]*0.5))*(np.linalg.det(sigma)**0.5))) * np.exp(-0.5 * np.inner((x-mu), np.linalg.inv(sigma) @ (x-mu)))
        # just implement # p(x) = (1 / ((2π)^(n/2) * |Σ|^(1/2))) * exp(-1/2 * (x-μ)^T Σ^(-1) (x-μ))
            