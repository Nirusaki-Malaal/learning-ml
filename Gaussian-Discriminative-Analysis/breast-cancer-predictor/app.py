from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from plugins.data_service import BreastCancerData
from plugins.gda_model import GDABreastCancerModel
from plugins.schemas import PredictionRequest, VisualizationRequest
from plugins.viz_service import VisualizationService


app = FastAPI(title="Breast Cancer GDA Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/statics", StaticFiles(directory="statics"), name="statics")
templates = Jinja2Templates(directory="templates")

data_service = BreastCancerData()
model = GDABreastCancerModel(data_service.features_frame(), data_service.target_series())
viz_service = VisualizationService(data_service, model)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/api/metadata")
def metadata():
    payload = data_service.metadata()
    payload["model"] = {
        "type": "Gaussian Discriminant Analysis",
        "training_accuracy": round(model.accuracy_on_training_data(), 4),
        "features_used": len(model.feature_names),
        "regularization": model.regularization,
    }
    return payload


@app.get("/api/random")
def random_sample():
    return data_service.random_sample()


@app.post("/api/predict")
def predict(request: PredictionRequest):
    return model.predict(request.features)


@app.post("/api/visualization")
def visualization(request: VisualizationRequest):
    return viz_service.build(
        features=request.features,
        x_feature=request.x_feature,
        y_feature=request.y_feature,
        grid_size=request.grid_size,
    )


@app.get("/api/confusion")
def confusion():
    return viz_service.confusion_matrix()


@app.get("/api/correlation")
def correlation():
    return viz_service.correlation_matrix()


@app.get("/api/class-stats")
def class_stats():
    return viz_service.class_statistics()
