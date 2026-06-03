import numpy as np
import matplotlib.pyplot as plt
class PCA():
    def __init__(self, X, K=10):
        X = np.array(X)
        self.X = np.array(X)
        self.mu = np.sum(self.X , axis=0) / X.shape[0]
        self.X = self.X - self.mu
        self.std = np.std(self.X, axis=0)
        self.X = self.X / self.std
        self.K = K

    def train(self):
        sigma  = (self.X.T @ self.X) / self.X.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(sigma) ## ascending
        eigenvalues = eigenvalues[::-1] ## descending 
        eigenvectors = eigenvectors [:, ::-1] # (n,n) # pick top K [u1 , u2 ..... un]
        self.U = eigenvectors[: , :self.K] # (n,k)
    
    def transform(self, x): # x is an example
        return (np.array(x) - self.mu) / self.std @ self.U[:, :self.K]
    
    def reconstruction(self,y):
        return ((y @ self.U.T) * self.std) + self.mu

    @staticmethod
    def simulate(model, X, labels=None):
        Y = model.transform(X)           # (m,k)
        X_hat = model.reconstruction(Y)  # (m,n)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax = axes[0]
        if labels is not None:
            scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=15)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(Y[:, 0], Y[:, 1], s=15)
        ax.set_title("Latent Space (K=2)")
        ax.set_xlabel("u1")
        ax.set_ylabel("u2")
        ax = axes[1]
        ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5, label="original")
        ax.scatter(X_hat[:, 0], X_hat[:, 1], s=15, alpha=0.5, label="reconstructed")
        ax.set_title("Original vs Reconstructed")
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        ax.legend()
        ax = axes[2]
        im = ax.imshow(model.U, aspect='auto', cmap='coolwarm')
        plt.colorbar(im, ax=ax)
        ax.set_title("Eigenvectors (U)")
        ax.set_xlabel("top K components")
        ax.set_ylabel("features")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    m, n, k = 300, 10, 2
    true_Z = np.random.randn(m, k)
    true_U = np.random.randn(n, k)
    noise  = np.random.randn(m, n) * 0.5
    X      = true_Z @ true_U.T + noise
    labels = np.array([0]*100 + [1]*100 + [2]*100)
    X[:100]   += 3
    X[100:200] -= 3
    model = PCA(X, K=k)
    model.train()
    print(f"reconstruction error: {np.linalg.norm(X - model.reconstruction(model.transform(X)))**2 / X.shape[0]:.4f}")
    model.simulate(model, X, labels=labels)