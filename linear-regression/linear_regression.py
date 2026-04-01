import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
price = np.array([
    400, 232, 318, 450, 280, 375, 500, 195, 420, 360,
    275, 415, 330, 250, 490, 210, 385, 445, 305, 260,
    520, 175, 395, 465, 315, 240, 410, 355, 285, 430,
    180, 370, 495, 225, 340, 460, 295, 405, 265, 480,
    345, 215, 390, 435, 270, 510, 300, 365, 245, 420
], dtype=float)

size = np.array([
    2104, 1416, 1534, 2400, 1300, 1800, 2650, 1100, 2200, 1750,
    1250, 2180, 1600, 1180, 2580, 1050, 1900, 2350, 1480, 1220,
    2750, 980,  2000, 2480, 1560, 1120, 2150, 1720, 1350, 2280,
    1000, 1820, 2600, 1080, 1650, 2420, 1420, 2100, 1260, 2520,
    1680, 1060, 1960, 2300, 1290, 2700, 1460, 1780, 1140, 2200
], dtype=float)

m = len(price)
features = np.array([np.ones(m), size])
n = len(features)
parameters = np.zeros(n)
epoch = 1000
alpha = 1e-10

def hypothesis(x):
    return np.dot(parameters, x)

def cost():
    total = 0
    for i in range(m):
        total += (hypothesis(features[:, i]) - price[i]) ** 2
    return total / 2

def gradient_descent():
    for i in range(n):
        error_sum = 0
        for j in range(m):
            error_sum += (hypothesis(features[:, j]) - price[j]) * features[i][j]
        parameters[i] = parameters[i] - alpha * error_sum

def train():
    for e in range(epoch):
        gradient_descent()
        if e % 100 == 0:
            print(f"Epoch {e:4d} | Cost: {cost():.4f}")

def visualize():
    plt.scatter(size, price, color='blue', label='Data Points')
    x_line = np.linspace(min(size), max(size), 100)
    y_line = parameters[0] + parameters[1] * x_line
    plt.plot(x_line, y_line, color='red', label='Regression Line')
    plt.xlabel('Size (sq ft)')
    plt.ylabel('Price (thousands)')
    plt.title('Linear Regression - Size vs Price')
    plt.legend()
    plt.show()

train()
visualize()

x = float(input("enter size in (feet)^2\n"))
print("Prediction =", hypothesis(np.array([1, x])), "Grands")
