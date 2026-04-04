from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
from plugins.Logistic_Regression import LogisticRegression
from plugins.TF_IDFVectorizer import TFIDFVectorizer
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/statics", StaticFiles(directory="statics"), name="statics")
templates = Jinja2Templates(directory="templates")

# Initialize and train model on startup
print("Loading data...")
df = pd.read_csv("spam.csv", encoding='utf-8-sig', usecols=[0, 1], names=['label', 'text'], skiprows=1, on_bad_lines='skip')
emails = df["text"].tolist()
y = df["label"].values

print("Vectorizing...")
vectorizer = TFIDFVectorizer(max_features=1500)  # slightly reduced for speed with newton's method inversion
X = vectorizer.fit_transform(emails)

# Split data for accuracy calculation
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training model using Newton's Method...")
model = LogisticRegression(
    feature_input=X_train,
    y_input=y_train,
    keywords=["ham", "spam"],
    epoch=15,  # Newton's method doesn't need many epochs
    alpha=1.0  # learning rate usually 1 for strict Newton step
)
model.newtons_method(epoch=15)
print("Training complete!")

# Calculate accuracy on test set
correct = 0
for i in range(len(X_test)):
    pred = model.predict(X_test[i])
    actual = 1 if y_test[i] == "spam" else 0
    if pred == actual:
        correct += 1
test_accuracy = round(correct / len(X_test) * 100, 2)
print(f"Test Accuracy: {test_accuracy}%")

class EmailInput(BaseModel):
    email_text: str

@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.get("/analytics")
def serve_analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html", {"request": request})

@app.post("/predict")
def predict(sms_text: str = Form(...)):
    # Vectorize input SMS
    x_new = vectorizer.transform([sms_text])[0]
    
    # Predict probabilities and class
    prob = model.predict_probablity(x_new)
    pred_class = model.predict(x_new)
    
    result = "Spam" if pred_class == 1 else "Ham (Not Spam)"
    spam_prob = round(float(prob) * 100, 2)
    ham_prob = round((1 - float(prob)) * 100, 2)
    
    return {
        "prediction": result,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob
    }

@app.get("/api/analytics")
def get_analytics():
    """Get analytics data for the dashboard"""
    # Class counts
    spam_count = int(np.sum(y == "spam"))
    ham_count = int(np.sum(y == "ham"))
    total_messages = len(y)
    
    # Message length distribution
    lengths = [len(str(email)) for email in emails]
    spam_lengths = [len(str(emails[i])) for i in range(len(emails)) if y[i] == "spam"]
    ham_lengths = [len(str(emails[i])) for i in range(len(emails)) if y[i] == "ham"]
    
    # Create length bins
    bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
    
    spam_hist = np.histogram(spam_lengths, bins=bins)[0].tolist()
    ham_hist = np.histogram(ham_lengths, bins=bins)[0].tolist()
    
    # Get top spam and ham indicator words from model coefficients (parameters)
    # Skip index 0 which is the bias term
    params = model.parameters[1:]  # Exclude bias
    vocab = vectorizer.vocabulary
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Top spam words (highest positive coefficients)
    top_spam_indices = np.argsort(params)[-15:][::-1]
    top_spam_words = [
        {"word": inv_vocab.get(i, "unknown"), "score": float(params[i])}
        for i in top_spam_indices if i in inv_vocab
    ][:10]
    
    # Top ham words (highest negative coefficients - most negative)
    top_ham_indices = np.argsort(params)[:15]
    top_ham_words = [
        {"word": inv_vocab.get(i, "unknown"), "score": float(abs(params[i]))}
        for i in top_ham_indices if i in inv_vocab
    ][:10]
    
    # Confusion matrix calculation
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(X_test)):
        pred = model.predict(X_test[i])
        actual = 1 if y_test[i] == "spam" else 0
        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 0:
            tn += 1
        elif actual == 0 and pred == 1:
            fp += 1
        else:
            fn += 1
    
    return {
        "total_messages": total_messages,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "accuracy": test_accuracy,
        "feature_count": len(vocab),
        "length_distribution": {
            "labels": bin_labels,
            "spam": spam_hist,
            "ham": ham_hist
        },
        "top_spam_words": top_spam_words,
        "top_ham_words": top_ham_words,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
    }
