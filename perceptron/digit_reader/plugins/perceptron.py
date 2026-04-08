from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
import io
import time

_model = None
_accuracy = None
_train_history = None
_confusion = None
_weights = None
_prediction_history = []
_prediction_counter = 0

def get_model():
    global _model, _accuracy, _train_history, _confusion, _weights
    if _model is not None:
        return _model
    
    print("Fetching MNIST dataset and training the Perceptron model...")
    try:
        mnist = fetch_openml('mnist_784', parser='auto')
    except Exception:
        mnist = fetch_openml('mnist_784')
    
    X = mnist.data
    y = mnist.target.astype(int)
    
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    
    X = X / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with partial_fit to track loss curve
    model = Perceptron(eta0=0.001, max_iter=1, warm_start=True)
    _train_history = []
    
    epochs = 50
    for epoch in range(epochs):
        model.partial_fit(X_train, y_train, classes=[0, 1])
        score = model.score(X_test, y_test)
        _train_history.append({
            "epoch": epoch + 1,
            "accuracy": round(float(score), 4),
            "loss": round(1.0 - float(score), 4)
        })
    
    _model = model
    _accuracy = model.score(X_test, y_test)
    
    # Confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    _confusion = {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1])
    }
    
    # Weights for distribution
    weights = model.coef_.flatten()
    _weights = weights.tolist()
    
    print("Accuracy:", _accuracy)
    return _model

def predict_image(image_bytes):
    global _prediction_counter
    model = get_model()
    
    start_time = time.time()
    
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((28, 28))
    
    img_array = np.array(img)
    img_array = 255 - img_array       # invert colors
    img_array = img_array / 255.0     # normalize
    
    img_flat = img_array.flatten().reshape(1, -1)  # shape (1, 784)
    
    pred = model.predict(img_flat)
    decision = model.decision_function(img_flat)
    
    elapsed = round((time.time() - start_time) * 1000, 1)  # ms
    
    _prediction_counter += 1
    confidence = round(min(abs(float(decision[0])) / 5.0 * 100, 99.9), 1)
    
    record = {
        "id": f"x{1000 - _prediction_counter}",
        "predicted": int(pred[0]),
        "confidence": confidence,
        "time_ms": elapsed
    }
    _prediction_history.insert(0, record)
    if len(_prediction_history) > 50:
        _prediction_history.pop()
    
    return pred[0]

def get_analytics_data():
    model = get_model()
    
    # Decision boundary points (project to 2D using PCA-like approach)
    weights = model.coef_.flatten()
    
    # Get top 2 important features for scatter visualization
    top_indices = np.argsort(np.abs(weights))[-2:]
    
    try:
        mnist = fetch_openml('mnist_784', parser='auto')
    except Exception:
        mnist = fetch_openml('mnist_784')
    
    X = mnist.data
    y = mnist.target.astype(int)
    mask = (y == 0) | (y == 1)
    X = X[mask] / 255.0
    y = y[mask]
    
    # Sample 100 points for scatter plot
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), min(100, len(X)), replace=False)
    X_sample = np.array(X.iloc[sample_idx] if hasattr(X, 'iloc') else X[sample_idx])
    y_sample = np.array(y.iloc[sample_idx] if hasattr(y, 'iloc') else y[sample_idx])
    
    # Use decision function as x-axis, random noise as y
    decisions = model.decision_function(X_sample)
    
    scatter_data = {
        "class0": [],
        "class1": []
    }
    for i in range(len(y_sample)):
        point = {"x": round(float(decisions[i]), 3), "y": round(float(np.random.randn() * 2), 3)}
        if y_sample[i] == 0:
            scatter_data["class0"].append(point)
        else:
            scatter_data["class1"].append(point)
    
    # Weight histogram 
    weight_bins = np.histogram(weights, bins=30)
    weight_hist = {
        "counts": weight_bins[0].tolist(),
        "edges": [round(float(e), 4) for e in weight_bins[1].tolist()]
    }
    
    # Count predictions
    total_predictions = len(_prediction_history)
    pred_0 = sum(1 for p in _prediction_history if p["predicted"] == 0)
    pred_1 = total_predictions - pred_0
    
    return {
        "accuracy": round(float(_accuracy) * 100, 1),
        "total_predictions": total_predictions,
        "avg_response_ms": round(np.mean([p["time_ms"] for p in _prediction_history]) if _prediction_history else 23.0, 1),
        "training_epochs": len(_train_history),
        "train_history": _train_history,
        "confusion": _confusion,
        "scatter_data": scatter_data,
        "weight_histogram": weight_hist,
        "prediction_distribution": {"digit_0": pred_0, "digit_1": pred_1},
        "prediction_history": _prediction_history[:10],
        "weight_stats": {
            "mean": round(float(np.mean(weights)), 5),
            "std": round(float(np.std(weights)), 5),
            "min": round(float(np.min(weights)), 5),
            "max": round(float(np.max(weights)), 5)
        }
    }