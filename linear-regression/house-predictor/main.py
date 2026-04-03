from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from plugins.house_predictor import model

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

class HouseFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    b: float
    lstat: float

@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/predict")
def predict(features: HouseFeatures):
    data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(data)
    return {"prediction": round(prediction[0], 2)}

@app.get("/random")
def get_random_sample():
    df = pd.read_csv("data.csv")
    sample = df.sample(1).iloc[0].to_dict()
    return sample

@app.get("/analytics")
def serve_analytics(request: Request):
    return templates.TemplateResponse(request, "analytics.html", {"request": request})

@app.get("/api/analytics")
def get_analytics_data():
    from sklearn.linear_model import LinearRegression as SimpleLinearRegression
    
    df = pd.read_csv("data.csv")
    feature_cols = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]
    y = df['medv'].values
    
    result = {
        "features": {},
        "prices": y.tolist()
    }
    
    # For each feature, compute simple linear regression line
    for feat in feature_cols:
        x_vals = df[feat].values
        
        # Fit simple linear regression for this single feature
        simple_model = SimpleLinearRegression()
        simple_model.fit(x_vals.reshape(-1, 1), y)
        
        # Get line endpoints (min to max of feature)
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        y_min = float(simple_model.predict([[x_min]])[0])
        y_max = float(simple_model.predict([[x_max]])[0])
        
        result["features"][feat] = {
            "values": x_vals.tolist(),
            "line": {
                "x": [x_min, x_max],
                "y": [y_min, y_max]
            },
            "coefficient": float(simple_model.coef_[0]),
            "intercept": float(simple_model.intercept_)
        }
    
    return result
