import numpy as np
import pandas as pd
from plugins.Logistic_Regression import LogisticRegression
from plugins.TF_IDFVectorizer import TFIDFVectorizer

# Reading
df = pd.read_csv("spam.csv", encoding='utf-8-sig', usecols=[0, 1], names=['label', 'text'], skiprows=1, on_bad_lines='skip')
emails = df["text"].tolist()
y = df["label"].values

# TF-IDF
vectorizer = TFIDFVectorizer(max_features=5000)
X = vectorizer.fit_transform(emails)

# Train/Test Split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Logistic Regression
model = LogisticRegression(
    feature_input=X_train,
    y_input=y_train,
    keywords=["ham", "spam"],  # ✅ LIST, not dict
    epoch=15,  # ✅ Reduced epochs for Newton's method
    alpha=1.0  # Learning rate 1.0 for strict Newton's method
)

print("Training ho raha hai...")
model.newtons_method()
print("Training complete!")

# Accuracy
correct = 0
for i in range(len(X_test)):
    pred = model.predict(X_test[i])
    actual = 1 if y_test[i] == "spam" else 0
    if pred == actual:
        correct += 1
accuracy = correct / len(X_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Predict
email = input("Email daalo: ")
x_new = vectorizer.transform([email])[0]
prob = model.predict_probablity(x_new)
pred = model.predict(x_new)
print(f"Prediction : {'spam' if pred == 1 else 'ham'}")
print(f"Spam probability: {prob*100:.2f}%")
print(f"Ham  probability: {(1-prob)*100:.2f}%")