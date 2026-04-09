from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import io
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

_model = None
_accuracy = None
_train_history = None
_confusion = None
_weights = None
_pca = None
_pca_data = None
_prediction_history = []
_prediction_counter = 0

def get_model():
    global _model, _accuracy, _train_history, _confusion, _weights, _pca, _pca_data
    if _model is not None:
        return _model

    print("Fetching MNIST dataset and training the Softmax Regression model...")
    try:
        mnist = fetch_openml('mnist_784', parser='auto')
    except Exception:
        mnist = fetch_openml('mnist_784')

    X = np.array(mnist.data, dtype=np.float64)
    y = np.array(mnist.target, dtype=int)

    # Use ALL digits 0-9
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SGDClassifier with loss='log_loss' is equivalent to Softmax Regression
    # (multinomial logistic regression) trained via SGD.
    # Using partial_fit for epoch-by-epoch training to track learning curve.
    classes = np.arange(10)
    model = SGDClassifier(
        loss='log_loss',          # Softmax / cross-entropy loss
        learning_rate='optimal',
        eta0=0.001,
        alpha=1e-4,              # L2 regularization (1/C equivalent)
        random_state=42,
        max_iter=1,
        warm_start=True,
    )

    _train_history = []
    epochs = 50

    for epoch in range(1, epochs + 1):
        # Shuffle training data each epoch
        perm = np.random.RandomState(epoch).permutation(len(X_train))
        model.partial_fit(X_train[perm], y_train[perm], classes=classes)

        score = model.score(X_test, y_test)
        y_proba = _predict_proba_sgd(model, X_test)
        loss = log_loss(y_test, y_proba)

        _train_history.append({
            "epoch": epoch,
            "accuracy": round(float(score), 4),
            "loss": round(float(loss), 4)
        })

    _model = model
    _accuracy = _model.score(X_test, y_test)

    # 10x10 Confusion matrix
    y_pred = _model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    _confusion = cm.tolist()

    # Weight matrix shape: (10, 784) — one weight vector per class
    _weights = _model.coef_  # shape (10, 784)

    # PCA for 2D scatter visualization (precompute for analytics)
    _pca = PCA(n_components=2, random_state=42)
    _pca.fit(X_train)

    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), min(300, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx]

    X_2d = _pca.transform(X_sample)
    _pca_data = {
        "points": X_2d.tolist(),
        "labels": y_sample.tolist()
    }

    print(f"Softmax Regression Accuracy: {_accuracy:.4f}")
    return _model


def _predict_proba_sgd(model, X):
    """Get softmax probabilities from SGDClassifier decision function."""
    decision = model.decision_function(X)
    # Apply softmax
    exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
    return exp_d / exp_d.sum(axis=1, keepdims=True)


def predict_image(image_bytes):
    """Predict digit and return dict with prediction, confidence per class, and timing."""
    global _prediction_counter
    model = get_model()

    start_time = time.time()

    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((28, 28))

    img_array = np.array(img, dtype=np.float64)
    img_array = 255 - img_array       # invert colors (white bg -> black bg)
    img_array = img_array / 255.0     # normalize

    img_flat = img_array.flatten().reshape(1, -1)  # shape (1, 784)

    pred = model.predict(img_flat)
    proba = _predict_proba_sgd(model, img_flat)[0]  # shape (10,)

    elapsed = round((time.time() - start_time) * 1000, 1)  # ms

    _prediction_counter += 1
    predicted_digit = int(pred[0])
    confidence = round(float(proba[predicted_digit]) * 100, 1)

    record = {
        "id": f"x{1000 - _prediction_counter}",
        "predicted": predicted_digit,
        "confidence": confidence,
        "time_ms": elapsed
    }
    _prediction_history.insert(0, record)
    if len(_prediction_history) > 50:
        _prediction_history.pop()

    return {
        "prediction": predicted_digit,
        "confidence": confidence,
        "probabilities": {str(i): round(float(proba[i]) * 100, 2) for i in range(10)},
        "time_ms": elapsed
    }


def get_analytics_data():
    model = get_model()

    # Per-class accuracy from confusion matrix
    cm = np.array(_confusion)
    per_class_accuracy = {}
    for i in range(10):
        total = cm[i].sum()
        correct = cm[i][i]
        per_class_accuracy[str(i)] = round(float(correct / total * 100), 1) if total > 0 else 0.0

    # Weight statistics across all classes
    all_weights = _weights.flatten()
    weight_bins = np.histogram(all_weights, bins=40)
    weight_hist = {
        "counts": weight_bins[0].tolist(),
        "edges": [round(float(e), 4) for e in weight_bins[1].tolist()]
    }

    # Per-class weight norms (L2 norm of each class weight vector)
    class_weight_norms = [round(float(np.linalg.norm(_weights[i])), 4) for i in range(10)]

    # Count predictions per digit
    total_predictions = len(_prediction_history)
    pred_dist = {}
    for d in range(10):
        pred_dist[f"digit_{d}"] = sum(1 for p in _prediction_history if p["predicted"] == d)

    # PCA scatter data formatted per class
    scatter_data = {}
    for i in range(10):
        scatter_data[f"class{i}"] = []

    for idx, label in enumerate(_pca_data["labels"]):
        pt = _pca_data["points"][idx]
        scatter_data[f"class{label}"].append({
            "x": round(pt[0], 3),
            "y": round(pt[1], 3)
        })

    return {
        "accuracy": round(float(_accuracy) * 100, 1),
        "total_predictions": total_predictions,
        "avg_response_ms": round(
            np.mean([p["time_ms"] for p in _prediction_history]) if _prediction_history else 23.0, 1
        ),
        "training_epochs": len(_train_history),
        "train_history": _train_history,
        "confusion_matrix": _confusion,
        "scatter_data": scatter_data,
        "weight_histogram": weight_hist,
        "prediction_distribution": pred_dist,
        "prediction_history": _prediction_history[:10],
        "per_class_accuracy": per_class_accuracy,
        "class_weight_norms": class_weight_norms,
        "weight_stats": {
            "mean": round(float(np.mean(all_weights)), 5),
            "std": round(float(np.std(all_weights)), 5),
            "min": round(float(np.min(all_weights)), 5),
            "max": round(float(np.max(all_weights)), 5)
        }
    }
