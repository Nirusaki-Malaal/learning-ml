import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot  as plt
import matplotlib
matplotlib.use("QtAgg")

class FactorAnalysis:
    def __init__(self, X, d, epoch=100):
        self.X = X # (m,n)
        self.d = d # no of latent factors # 
        self.psi =  np.diag(np.var(X, axis=0)) ## psi is a diagonal matrix here Ai!=j (n,n)
        self.mu = np.mean(X, axis=0) #(1,n)
        self.lamda = np.random.randn(self.X.shape[1],d) * 0.01 #(n,d)
        self.epoch = epoch 

    def train(self):
        for _ in range(self.epoch):
            # E-STEP
            # (Z | X) ~ (0,u) , sigma)
            mu = np.zeros((self.X.shape[0], self.d)) # (m,d)
            some_var = np.linalg.inv(((self.lamda @ self.lamda.T) + self.psi)) #(n,n)
            mu =  (self.X - self.mu) @ some_var @ self.lamda  # (m,n) @  (d,n)(n,n) (d,n) => (m,n)(n,d) => (m,d)
            sigma = np.eye(self.d) - self.lamda.T @ some_var @ self.lamda

            # M STEP
            self.mu = np.sum(self.X, axis=0) / self.X.shape[0]
            self.lamda = (self.X.T @ mu) @ np.linalg.inv(np.sum((sigma + mu[:, :, None] * mu[:, None , :]),axis=0))
            psi_diag = np.diag((self.X.T @ self.X - self.lamda @ mu.T @ self.X )/self.X.shape[0])
            self.psi = np.diag(psi_diag)
        
    def multivariate_gaussian(self,x, mu, sigma): # P(X) ~ N(xi; u,A.AT + psi)
        n = x.shape[1]
        diff = x - mu                                                          # (m,n)
        mahal = np.sum(diff @ np.linalg.inv(sigma) * diff, axis=1)            # (m,)
        coef = 1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(sigma) ** 0.5)
        return coef * np.exp(-0.5 * mahal)
        
        # just implement # p(x) = (1 / ((2π)^(n/2) * |Σ|^(1/2))) * exp(-1/2 * (x-μ)^T Σ^(-1) (x-μ))
            
    
    def log_likelyhood(self):
        return np.sum(np.log(self.multivariate_gaussian(self.X, self.mu , (self.lamda @ self.lamda.T) + self.psi)))
    
    
    def transform(self, x):
        return (x - self.mu) @ np.linalg.inv((self.lamda @ self.lamda.T) + self.psi) @ self.lamda

    def reconstruction_error(self, x):
        x_hat = (self.transform(x) @ self.lamda.T) + self.mu 
        return np.linalg.norm(x - x_hat) ** 2 / self.X.shape[0]
    
    @staticmethod
    def visualize(model, X, labels=None):
        Z = model.transform(X)                                  # (m, d)
        X_hat = (model.transform(X) @ model.lamda.T) + model.mu # (m, n)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax = axes[0]
        if labels is not None:
            scatter = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='tab10', s=15)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(Z[:, 0], Z[:, 1], s=15)
        ax.set_title("Latent Space (d=2)")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax = axes[1]
        ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5, label="original")
        ax.scatter(X_hat[:, 0], X_hat[:, 1], s=15, alpha=0.5, label="reconstructed")
        ax.set_title("Original vs Reconstructed")
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        ax.legend()

        #lambda heatmap
        ax = axes[2]
        im = ax.imshow(model.lamda, aspect='auto', cmap='coolwarm')
        plt.colorbar(im, ax=ax)
        ax.set_title("Factor Loadings (Lambda)")
        ax.set_xlabel("latent factors")
        ax.set_ylabel("features")

        figuu = plt.figure()
        ax = figuu.add_subplot(111, projection='3d')
        ax.scatter(Z[:,0], Z[:,1], Z[:,2], c=labels, cmap='tab10', s=15)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    m, n, d = 300, 10, 3
    true_Z = np.random.randn(m, d)
    true_L = np.random.randn(n, d)
    noise  = np.random.randn(m, n) * 0.5
    X      = true_Z @ true_L.T + noise
    labels = np.array([0]*100 + [1]*100 + [2]*100)
    X[:100]  += 3
    X[100:200] -= 3
    model = FactorAnalysis(X, d=d, epoch=200)
    model.train()
    print(f"log likelihood:       {model.log_likelyhood():.4f}")
    print(f"reconstruction error: {model.reconstruction_error(X):.4f}")
    model.visualize(model, X, labels=labels)