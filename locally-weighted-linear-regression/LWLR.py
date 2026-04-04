import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


class LWLR:
    def __init__(self):
        self.feature = np.array(
            [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
             0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float
        )

        self.Y = np.array(
            [-0.14, -0.60, -0.91, -0.99, -0.84, -0.48, 0.02,
             0.51, 0.84, 0.99, 0.91, 0.60, 0.14], dtype=float
        )

        self.m = len(self.Y)
        self.features = np.array([np.ones(self.m), self.feature]).T
        self.n = self.features.shape[1]

        self.parameters = np.zeros(self.n)
        self.epoch = 10000
        self.bandwidth = 0.5
        self.alpha = 10 ** (-3.5)

        self.query = None
        self.weights = None

    def hypothesis(self, feature):
        return np.dot(feature, self.parameters)

    def weight_matrix(self):
        difference = self.features - self.query
        squared_difference = difference * difference
        squared_distance = np.sum(squared_difference, axis=1)
        denominator = 2 * (self.bandwidth ** 2)
        weights = np.exp((-1 * squared_distance) / denominator)
        return weights

    def cost(self):
        predictions = self.hypothesis(self.features)
        prediction_errors = self.Y - predictions
        squared_errors = prediction_errors * prediction_errors
        weighted_squared_errors = self.weights * squared_errors
        total_cost = np.sum(weighted_squared_errors)
        return total_cost

    def gradient_descent(self):
        predictions = self.hypothesis(self.features)
        prediction_errors = predictions - self.Y
        weighted_errors = self.weights * prediction_errors
        gradients = np.dot(self.features.T, weighted_errors)
        self.parameters = self.parameters - (self.alpha * gradients)

    def fit_query(self, x, method="gd", show_cost=False):
        self.query = np.array([1, x])

        if method == "gd":
            self.parameters = np.zeros(self.n)
            self.weights = self.weight_matrix()

            for i in range(self.epoch):
                if show_cost and i % 100 == 0:
                    print(f"cost {self.cost():.3f}")
                self.gradient_descent()

            return self.hypothesis(self.query)

        if method == "closed":
            self.predict_closed_form(x)
            return self.hypothesis(self.query)

        raise ValueError("method must be 'gd' or 'closed'")

    def predict_closed_form(self, x):
        self.query = np.array([1, x])
        self.weights = self.weight_matrix()

        weighted_features = self.features * self.weights[:, np.newaxis]
        x_transpose = self.features.T
        left_side = np.dot(x_transpose, weighted_features)
        right_side = np.dot(x_transpose, self.weights * self.Y)

        inverse_part = np.linalg.pinv(left_side)
        self.parameters = np.dot(inverse_part, right_side)

    def show_lwlr(self):
        xs = np.linspace(min(self.feature), max(self.feature), 100)
        ys = []

        for x in xs:
            self.predict_closed_form(x)
            ys.append(self.hypothesis(self.query))

        plt.scatter(self.feature, self.Y)
        plt.plot(xs, ys)
        plt.title("LWLR (Closed Form)")
        plt.show()


if __name__ == "__main__":
    model = LWLR()

    x = float(input("enter your query: "))

    prediction = model.fit_query(x, method="gd", show_cost=True)
    print(prediction)

    model.show_lwlr()
