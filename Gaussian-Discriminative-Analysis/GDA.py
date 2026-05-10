import numpy as np
import matplotlib.pyplot as plt
class GDA():
    def __init__(self):
        self.X = np.array([
            [1.0, 1.3, 2.0, 2.2, 1.8, 2.5, 3.0, 3.2,
            6.0, 6.5, 7.0, 7.2, 6.8, 7.5, 8.0, 8.2],

            [2.1, 1.8, 2.2, 1.9, 2.7, 2.3, 3.1, 2.9,
            5.8, 6.2, 6.8, 7.1, 6.5, 7.3, 7.8, 8.1]
        ]) # features | [x1,x2]

        self.y = np.array([
            0,0,0,0,0,0,0,0,
            1,1,1,1,1,1,1,1
        ])
        self.mean0 = np.mean(self.X[: ,self.y==0], axis=1) # false responses [x,y]
        self.mean1 = np.mean(self.X[: ,self.y==1], axis=1) # positive responses
        self.phi = np.sum(self.y)/self.y.shape[0] # phi k
        self.m = self.X.shape[1]
        self.n = self.X.shape[0]
        self.M = np.column_stack([
            self.mean0 if label == 0 else self.mean1
            for label in self.y
            ])
        self.covariance = ((self.X - self.M) @ (self.X - self.M).T)/self.m# sigma
        self.covariance_inv = np.linalg.inv(self.covariance)
        self.covariance_det = np.linalg.det(self.covariance)

    def show(self):

        plt.scatter(
            self.X[0][self.y == 0],
            self.X[1][self.y == 0],
            color='red',
            label='Class 0'
        )

        plt.scatter(
            self.X[0][self.y == 1],
            self.X[1][self.y == 1],
            color='blue',
            label='Class 1'
        )

        # mean points
        plt.scatter(
            self.mean0[0],
            self.mean0[1],
            color='darkred',
            marker='X',
            s=200,
            label='Mean 0'
        )

        plt.scatter(
            self.mean1[0],
            self.mean1[1],
            color='darkblue',
            marker='X',
            s=200,
            label='Mean 1'
        )

        # grid creation
        x = np.linspace(
            np.min(self.X[0]) - 2,
            np.max(self.X[0]) + 2,
            200
        )

        y = np.linspace(
            np.min(self.X[1]) - 2,
            np.max(self.X[1]) + 2,
            200
        )

        Xg, Yg = np.meshgrid(x, y)

        # density matrices
        Z0 = np.zeros(Xg.shape)
        Z1 = np.zeros(Xg.shape)

        # gaussian densities
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):

                point = [Xg[i, j], Yg[i, j]]

                Z0[i, j] = self.multivariate_gaussian(point, 0)
                Z1[i, j] = self.multivariate_gaussian(point, 1)

        # contour of negative classes
        plt.contour(
            Xg,
            Yg,
            Z0,
            levels=6,
            colors='red',
            alpha=0.7
        )

        # contour of positive classes
        plt.contour(
            Xg,
            Yg,
            Z1,
            levels=6,
            colors='blue',
            alpha=0.7
        )
        Z_boundary = np.zeros(Xg.shape)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):

                point = [Xg[i,j], Yg[i,j]]

                Z_boundary[i,j] = self.predict(point)
        plt.contour(
            Xg,
            Yg,
            Z_boundary,
            levels=[0.5],
            colors='black',
            linewidths=2
        )
        plt.contourf(
            Xg,
            Yg,
            Z_boundary,
            alpha=0.2,
            cmap='bwr'
        )

        plt.xlabel('X1')
        plt.ylabel('X2')

        plt.title('Gaussian Discriminant Analysis')

        plt.grid(True)

        plt.legend()

        plt.show()

    def multivariate_gaussian(self, feature, y): #y= 1 or 0
        feature = np.array(feature)
        if y:
            return (1/(((2*np.pi)**(self.n*0.5)) * (self.covariance_det**0.5))) * np.exp(-0.5 * (feature-self.mean1)@ self.covariance_inv @ (feature-self.mean1).T)
        else:
            return (1/(((2*np.pi)**(self.n*0.5)) * (self.covariance_det**0.5))) * np.exp(-0.5 * (feature-self.mean0)@ self.covariance_inv @ (feature-self.mean0).T)
    
    def predict(self ,x):
        x = np.array(x)
        score0 = self.multivariate_gaussian(x, 0) * (1-self.phi)
        score1 = self.multivariate_gaussian(x,1) * self.phi
        if score1 > score0:
            return 1
        else:
            return 0

if __name__ == "__main__":
    gda = GDA()
    gda.show()