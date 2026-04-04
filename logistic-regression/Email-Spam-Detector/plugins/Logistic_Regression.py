import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self,feature_input , y_input,keywords, epoch=10000, alpha=(10 ** (-3.5))):
        self.feature = feature_input
        self.Y = self.convert(keywords , y_input)
        self.m = len(self.Y)
        self.features = np.column_stack((np.ones(self.m), self.feature))
        self.n = self.features.shape[1]
        self.parameters = np.zeros(self.n)
        self.epoch = epoch
        self.alpha = 10 ** (-3.5)
        self.weights = None
        self.deciding_line = 0.5
    
    def convert(self, keywords, y):
        y = list(y)   # ← ye add karo
        for i in range(len(y)):
            if y[i] == keywords[0]:
                y[i] = 0
            else:
                y[i] = 1
        return np.array(y, dtype=int)
    
    def hypothesis_all(self):
        z = self.features @ self.parameters   # (m, n) @ (n,) = (m,)
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    # for both
    def hypothesis(self, feature):
        z = self.parameters @ feature
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    ## for newtons method
    def weight_matrix(self):
        self.weights = []
        for i in range(self.m):
            weight = self.hypothesis(self.features[i]) * (1 - self.hypothesis(self.features[i]))
            self.weights.append(weight)
        self.weights = np.diag(self.weights)
        return self.weights

    # for both
    def cost(self):
        predictions = self.hypothesis_all()
        return np.sum(self.Y * np.log(predictions) + (1 - self.Y) * np.log(1 - predictions))
    


    def error_sum(self):
        error_sums = 0
        for i in range(self.m):
            error_sums += self.Y[i] - self.hypothesis(self.features[i])
        return error_sums

    def gradient(self):
        predictions = self.hypothesis_all()
        errors = self.Y - predictions
        return self.features.T @ errors

    def gradient_descent(self):
        for _ in range(self.epoch):
            self.parameters = self.parameters + (self.alpha * self.gradient())

    def newtons_method(self, epoch=10, tol=1e-4, lambda_reg=1e-4):
        # We can use a much smaller number of epochs for Newton's method
        self.epoch = epoch
        
        # Identity matrix for L2 Regularization (speeds up inversion infinitely)
        I = np.eye(self.n)
        I[0, 0] = 0 # Don't regularize the bias term
        
        for _ in range(self.epoch):
            predictions = self.hypothesis_all()
            errors = self.Y - predictions
            
            # Gradient with L2 Regularization
            gradient = self.features.T @ errors - lambda_reg * self.parameters
            
            # Early stopping if gradient is close to zero (Huge speedup)
            if np.max(np.abs(gradient)) < tol:
                break
            
            # W = diag(predictions * (1 - predictions))
            W_diag = predictions * (1 - predictions)
            W_diag = np.clip(W_diag, 1e-15, 1 - 1e-15)
            
            # Hessian = X^T W X + L2 Regularization
            X_W = self.features * W_diag[:, np.newaxis]
            Hessian = self.features.T @ X_W + lambda_reg * I
            
            # L2 Reg guarantees invertibility, meaning solve() is almost always used
            # solve() is multiple orders of magnitude faster than pinv()
            try:
                update = np.linalg.solve(Hessian, gradient)
            except np.linalg.LinAlgError:
                update = np.linalg.pinv(Hessian) @ gradient
                
            self.parameters = self.parameters + update

    def predict(self, x):
        feature = np.concatenate(([1] , x))
        h = self.hypothesis(feature)
        return 1 if h >= self.deciding_line else 0

    def predict_probablity(self, x):
        feature = np.concatenate(([1] , x))
        h = self.hypothesis(feature)
        return h

    def plot(self):
        costs = []
        self.parameters = np.zeros(self.n)  # reset
        for _ in range(self.epoch):
            costs.append(self.cost())
            self.parameters = self.parameters + (self.alpha * self.gradient())

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(costs)
        plt.title("Cost vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")

        plt.subplot(1, 2, 2)
        x_vals = np.linspace(0, 7, 100)
        y_vals = [self.hypothesis(np.array([1, x])) for x in x_vals]
        plt.plot(x_vals, y_vals, label="Sigmoid")
        plt.scatter(self.feature, self.Y, color="red", label="Data")
        plt.axhline(0.5, color="gray", linestyle="--", label="Decision Boundary")
        plt.title("Decision Boundary")
        plt.xlabel("Feature")
        plt.ylabel("Probability")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def summary(self):
        print("=" * 35)
        print("      LOGISTIC REGRESSION SUMMARY")
        print("=" * 35)
        print(f"  Epochs       : {self.epoch}")
        print(f"  Alpha        : {self.alpha:.6f}")
        print(f"  Parameters   : {self.parameters}")
        print(f"  Final Cost   : {self.cost():.4f}")
        print("-" * 35)
        print("  Predictions:")
        for i in range(self.m):
            pred = self.predict(self.feature[i])
            actual = self.Y[i]
            status = "✅" if pred == actual else "❌"
            print(f"  x={self.feature[i]} | Actual={actual} | Predicted={pred} {status}")
        print("=" * 35)

    def interactive_predict(self):
        print("\n--- Predict Karo ---")
        while True:
            x = input("Feature value daalo (ya 'q' press karo exit ke liye): ")
            if x.lower() == 'q':
                break
            try:
                x = float(x)
                pred = self.predict(x)
                feature = np.array([1, x])
                prob = self.hypothesis(feature)
                print(f"  x={x} | Probability={prob:.4f} | Predicted Class={pred}")
            except ValueError:
                print("  Sirf number daalo!")
    



