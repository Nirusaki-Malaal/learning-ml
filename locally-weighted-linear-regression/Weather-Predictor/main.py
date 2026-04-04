from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np

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

# Load Delhi weather data
df_weather = pd.read_csv("delhi_2013_2024.csv")

class WeatherLWLR:
    """Locally Weighted Linear Regression for weather prediction"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.column_stack([np.ones(len(X)), X])  # Add bias term
        self.y = y
        self.m = len(y)
    
    def _weight_matrix(self, query: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute Gaussian weights for each training point"""
        diff = self.X[:, 1:] - query[1:]  # Exclude bias from distance
        sq_dist = np.sum(diff * diff, axis=1)
        weights = np.exp(-sq_dist / (2 * bandwidth ** 2))
        return weights
    
    def predict(self, query: np.ndarray, bandwidth: float) -> float:
        """Predict using locally weighted closed-form solution"""
        query_with_bias = np.array([1.0] + list(query))
        weights = self._weight_matrix(query_with_bias, bandwidth)
        
        W = np.diag(weights)
        XtW = self.X.T @ W
        XtWX = XtW @ self.X
        XtWy = XtW @ self.y
        
        # Use pseudo-inverse for numerical stability
        theta = np.linalg.pinv(XtWX) @ XtWy
        return float(np.dot(query_with_bias, theta))
    
    def predict_curve(self, x_range: np.ndarray, feature_idx: int, 
                      other_features: np.ndarray, bandwidth: float) -> np.ndarray:
        """Generate LWLR curve for visualization (single feature varying)"""
        predictions = []
        for x_val in x_range:
            query = other_features.copy()
            query[feature_idx] = x_val
            pred = self.predict(query, bandwidth)
            predictions.append(pred)
        return np.array(predictions)


# Feature columns for prediction (inputs)
FEATURE_COLS = ["month", "dayofyear", "humidity", "windspeed", "sealevelpressure"]
TARGET_COL = "temp"

# Prepare training data
X_train = df_weather[FEATURE_COLS].values.astype(float)
y_train = df_weather[TARGET_COL].values.astype(float)

# Normalize features for better LWLR performance
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std == 0] = 1  # Prevent division by zero
X_train_norm = (X_train - X_mean) / X_std

# Create LWLR model
lwlr_model = WeatherLWLR(X_train_norm, y_train)


class WeatherFeatures(BaseModel):
    month: int
    dayofyear: int
    humidity: float
    windspeed: float
    sealevelpressure: float
    bandwidth: Optional[float] = 0.5


@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.post("/predict")
def predict(features: WeatherFeatures):
    query = np.array([
        features.month,
        features.dayofyear,
        features.humidity,
        features.windspeed,
        features.sealevelpressure
    ], dtype=float)
    
    # Normalize query
    query_norm = (query - X_mean) / X_std
    
    prediction = lwlr_model.predict(query_norm, features.bandwidth)
    
    # Clamp to reasonable temperature range
    prediction = max(0, min(50, prediction))
    
    return {"prediction": round(prediction, 2)}


@app.get("/random")
def get_random_sample():
    sample = df_weather.sample(1).iloc[0]
    return {
        "month": int(sample["month"]),
        "dayofyear": int(sample["dayofyear"]),
        "humidity": float(sample["humidity"]),
        "windspeed": float(sample["windspeed"]),
        "sealevelpressure": float(sample["sealevelpressure"]),
        "actual_temp": float(sample["temp"])
    }


@app.get("/analytics")
def serve_analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html", {"request": request})


class FastLWLR:
    """Optimized LWLR using subsampled data for visualization"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sample_size: int = 500):
        # Subsample for faster computation
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]
            y = y[indices]
        
        self.X = np.column_stack([np.ones(len(X)), X])
        self.y = y
        self.m = len(y)
    
    def predict_batch(self, queries: np.ndarray, bandwidth: float) -> np.ndarray:
        """Predict multiple queries efficiently"""
        predictions = []
        for query in queries:
            query_with_bias = np.array([1.0] + list(query))
            
            # Gaussian weights
            diff = self.X[:, 1:] - query
            sq_dist = np.sum(diff * diff, axis=1)
            weights = np.exp(-sq_dist / (2 * bandwidth ** 2))
            
            # Weighted least squares
            W = np.diag(weights)
            XtW = self.X.T @ W
            XtWX = XtW @ self.X
            XtWy = XtW @ self.y
            
            theta = np.linalg.pinv(XtWX) @ XtWy
            predictions.append(float(np.dot(query_with_bias, theta)))
        
        return np.array(predictions)


# Create fast model for analytics (subsampled)
fast_lwlr = FastLWLR(X_train_norm, y_train, sample_size=400)

# Subsample scatter data for visualization
SCATTER_SAMPLE_SIZE = 500
if len(X_train) > SCATTER_SAMPLE_SIZE:
    scatter_indices = np.random.choice(len(X_train), SCATTER_SAMPLE_SIZE, replace=False)
    X_scatter = X_train[scatter_indices]
    y_scatter = y_train[scatter_indices]
else:
    X_scatter = X_train
    y_scatter = y_train


@app.get("/api/analytics")
def get_analytics_data(bandwidth: float = Query(default=0.5, ge=0.01, le=5.0)):
    """Get analytics data with LWLR curves at specified bandwidth"""
    
    result = {
        "features": {},
        "temperatures": y_scatter.tolist(),  # Subsampled for faster rendering
        "bandwidth": bandwidth
    }
    
    feature_info = {
        "month": {"name": "Month", "desc": "Month of year (1-12)"},
        "dayofyear": {"name": "Day of Year", "desc": "Day number (1-365)"},
        "humidity": {"name": "Humidity", "desc": "Relative humidity (%)"},
        "windspeed": {"name": "Wind Speed", "desc": "Wind speed (km/h)"},
        "sealevelpressure": {"name": "Pressure", "desc": "Sea level pressure (hPa)"}
    }
    
    # Mean values for holding other features constant (use full data for means)
    mean_query_norm = np.zeros(len(FEATURE_COLS))  # Normalized mean is 0
    
    for i, feat in enumerate(FEATURE_COLS):
        x_vals = X_scatter[:, i]  # Subsampled scatter values
        x_min, x_max = float(X_train[:, i].min()), float(X_train[:, i].max())
        
        # Fewer curve points for speed
        n_points = 25
        x_range = np.linspace(x_min, x_max, n_points)
        
        # Build normalized queries
        queries_norm = []
        for x_val in x_range:
            query = mean_query_norm.copy()
            query[i] = (x_val - X_mean[i]) / X_std[i]
            queries_norm.append(query)
        
        # Batch prediction
        lwlr_predictions = fast_lwlr.predict_batch(np.array(queries_norm), bandwidth)
        
        result["features"][feat] = {
            "values": x_vals.tolist(),
            "curve": {
                "x": x_range.tolist(),
                "y": lwlr_predictions.tolist()
            },
            "info": feature_info[feat]
        }
    
    return result


@app.get("/api/dataset-stats")
def get_dataset_stats():
    """Get dataset statistics for display"""
    return {
        "total_records": len(df_weather),
        "date_range": {
            "start": df_weather["DATE"].iloc[0],
            "end": df_weather["DATE"].iloc[-1]
        },
        "temp_stats": {
            "min": float(df_weather["temp"].min()),
            "max": float(df_weather["temp"].max()),
            "mean": float(df_weather["temp"].mean())
        },
        "features": FEATURE_COLS
    }
