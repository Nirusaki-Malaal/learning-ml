import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

feature_cols = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]
X = df[feature_cols]
y = df['medv']

def visualize(model):
    global X,y
    plt.scatter(X["rm"],y, color='red',alpha=0.7, edgecolors='darkred') # to scatter points given in  a data frame with some paramteres
    y_pred = model.predict(X)
    plt.scatter(X["rm"], y_pred, color='blue', alpha=0.3, label='Predicted') # to plot predictions for rm
    plt.legend()
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # random_state=42 data split ka random pattern fix karna
# test_size = 0.2 matlab kitna data testing ke liye hai and kitna data training ke liye

"""
| Variable | Meaning         |
| -------- | --------------- |
| X_train  | training inputs |
| X_test   | testing inputs  |
| y_train  | training output |
| y_test   | testing output  |
"""
model = LinearRegression() # creating model from the blueprint
model.fit(X_train, y_train)

if __name__ == '__main__':
    example_input = pd.DataFrame([X.iloc[0].to_dict()])
    y_pred = model.predict(example_input) ## example prediction
    print(round(y_pred[0], 2), "K Dollars", sep="")
    
    visualize(model)
