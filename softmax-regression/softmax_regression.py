import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


class SoftmaxRegression():
    def __init__(self, alpha=(10**-4),epoch=1000):
        self.feature = np.array([
                                # f1  f2   f3   f4
                                [1.2, 3.4, 5.6, 2.1], # exampless 
                                [2.3, 4.5, 1.2, 3.3],
                                [3.1, 2.2, 4.4, 1.1],
                                [5.5, 2.3, 3.3, 4.4],
                                [6.7, 3.2, 2.1, 5.5],
                                [7.8, 4.1, 3.2, 6.6],
                                [8.9, 5.5, 4.4, 7.7],
                                [2.2, 1.1, 0.5, 2.2],
                                [3.3, 2.2, 1.1, 3.3],
                                [4.4, 3.3, 2.2, 4.4],
                                [6.6, 5.5, 3.3, 5.5],
                                [7.7, 6.6, 4.4, 6.6],
                                [8.8, 7.7, 5.5, 7.7],
                                [9.9, 8.8, 6.6, 8.8],
                                [1.1, 0.9, 0.8, 1.0]])
        
        self.y = np.array([
                            0, 0, 1, 1, 2, # ) light user , 1 moderate user , 2 heavy user 
                            2, 2, 0, 0, 1,
                            1, 2, 2, 2, 0
                            ])
        self.m = len(self.y) # no of examples
        self.num_classes = len(np.unique(self.y))
        self.X = np.vstack((np.ones(self.m), self.feature)).T
        self.n = self.X.shape[1] # no of features
        self.parameters = np.zeros((self.num_classes, self.n))
        self.alpha = alpha
        self.epoch = epoch

    def cost(self):
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                pass

    def hypothesis(self, x, class_number):
        sum_of_all = 0
        for i in range(self.num_classes):
            sum_of_all += np.exp(self.parameters[i] @ x)

        return np.exp(self.parameters[class_number] @ x) / sum_of_all

    def update(self):
        pass

    def error_sum(self):
        pass    