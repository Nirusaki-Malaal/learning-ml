import io
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from plugins.softmax_regression import get_model, predict_image, get_analytics_data

app = FastAPI(title="Neural Digit - Softmax AI Predictor")

app.mount("/statics", StaticFiles(directory="statics"), name="statics")

# Initialize model
get_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents)
    return result

@app.get("/api/analytics")
async def analytics():
    data = get_analytics_data()
    return JSONResponse(content=data)
