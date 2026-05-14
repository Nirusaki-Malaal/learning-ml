import numpy as np

# Output / Target
# 0 = No (Don't Play)
# 1 = Yes (Play)
class NaiveBayes:
    def __init__(self):
        self.X = np.array([
    # Outlook (0=Sunny, 1=Overcast, 2=Rain)
    [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2],
    # Temperature (0=Hot, 1=Mild, 2=Cool)
    [0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1],
    # Humidity (0=High, 1=Normal)
    [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    # Wind (0=Weak, 1=Strong)
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        ])
        self.y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
        self.n , self.m = self.X.shape
        self.num_values = np.array([3,3,2,2])
        self.phi = (np.sum(self.y)+1)/(self.m+2)
        self.phi_1 = [np.zeros(v) for v in self.num_values]
        self.phi_0 = [np.zeros(v) for v in self.num_values]

    def compute_phi(self):
        total_y1 = np.sum(self.y == 1) # finding denominators
        total_y0 = np.sum(self.y == 0)
        for j in range(self.n):
            values = self.num_values[j] # example teen values
            counts_y1 = np.zeros(values)
            counts_y0 = np.zeros(values)
            for i in range(self.m):
                val = self.X[j][i]
                if self.y[i] == 1:
                    counts_y1[val] +=1
                else:
                    counts_y0[val] +=1
            self.phi_1[j] = (counts_y1 + 1) / (total_y1 + values)
            self.phi_0[j] = (counts_y0 + 1) / (total_y0 + values) # laplace smoothing value


    def prediction(self,x):
        p1 = self.phi
        p0 = 1-self.phi
        for i in range(self.n):
            val = x[i]
            p1 *= self.phi_1[i][val]
            p0 *= self.phi_0[i][val]
        # if p1>p0:
        #     return 1
        # else:
        #     return 0
        return p1

    def show(self):
        import matplotlib.pyplot as plt

        feature_names = ["Outlook", "Temperature", "Humidity", "Wind"]

        for j in range(self.n):
            x = np.arange(len(self.phi_1[j]))
            plt.figure()
            plt.bar(x - 0.2, self.phi_1[j], width=0.4, label="y=1")
            plt.bar(x + 0.2, self.phi_0[j], width=0.4, label="y=0")
            plt.title(f"{feature_names[j]}")
            plt.xlabel("Value")
            plt.ylabel("Probability")
            plt.xticks(x)
            plt.legend()
            plt.grid(alpha=0.3)

            plt.show()



if __name__ == "__main__":
    model = NaiveBayes()
    model.compute_phi()
    x = eval(input("Enter a list of features"))
    model.show()