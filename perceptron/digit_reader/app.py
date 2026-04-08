import io
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from plugins.perceptron import get_model, predict_image, get_analytics_data

app = FastAPI(title="Neural Digit - AI Predictor")

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
    prediction = predict_image(contents)
    return {"prediction": int(prediction)}

@app.get("/api/analytics")
async def analytics():
    data = get_analytics_data()
    return JSONResponse(content=data)
