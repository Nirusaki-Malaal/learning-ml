# Delhi Weather Predictor 🌤️

A temperature prediction system using **Locally Weighted Linear Regression (LWLR)** trained on Delhi weather data (2013-2024). Features interactive visualizations with live bandwidth adjustment to explore how LWLR locality affects predictions.

## Features ✨

- **LWLR Algorithm**: Locally Weighted Linear Regression gives more weight to nearby data points
- **Live Bandwidth Control**: Adjust τ (tau) in real-time to see how it affects predictions
  - Lower τ → More local fit (captures fine patterns)
  - Higher τ → Smoother global fit
- **Interactive Analytics**: Visualize LWLR curves for each feature with adjustable bandwidth
- **Delhi Weather Dataset**: 3,557 days of weather data from 2013-2024

## Project Structure 📁

```
.
├── main.py                 # FastAPI backend with LWLR implementation
├── delhi_2013_2024.csv     # Delhi weather dataset
├── requirements.txt        # Python dependencies
├── plugins/
│   └── LWLR.py             # Standalone LWLR implementation
├── templates/
│   ├── index.html          # Predictor page with bandwidth slider
│   └── analytics.html      # Interactive LWLR visualizations
└── statics/
    ├── style.css           # Glassmorphism styles
    └── app.js              # Frontend JavaScript
```

## Input Features 🌡️

| Feature | Description |
|---------|-------------|
| Month | Month of year (1-12) |
| Day of Year | Day number (1-365) |
| Humidity | Relative humidity (%) |
| Wind Speed | Wind speed (km/h) |
| Pressure | Sea level pressure (hPa) |

**Target**: Average daily temperature (°C)

## Setup & Execution 🚀

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Open in browser**
   - Predictor: [http://localhost:8000](http://localhost:8000)
   - Analytics: [http://localhost:8000/analytics](http://localhost:8000/analytics)

## How LWLR Works 🧠

Unlike standard linear regression which uses all data equally, LWLR assigns **Gaussian weights** based on distance from the query point:

```
w(i) = exp( -||x(i) - x_query||² / (2τ²) )
```

The bandwidth parameter **τ** controls how quickly weights decay:
- Small τ: Only very close points matter → wiggly fit
- Large τ: Most points contribute → smooth fit (approaches standard regression)

## Tech Stack 🛠️

- **Backend**: FastAPI, NumPy, Pandas
- **Frontend**: Vanilla JS, Chart.js, TailwindCSS
- **Algorithm**: Custom LWLR with closed-form weighted least squares

## API Endpoints 📡

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Predictor page |
| `/analytics` | GET | Analytics dashboard |
| `/predict` | POST | Predict temperature (accepts bandwidth) |
| `/random` | GET | Get random day from dataset |
| `/api/analytics?bandwidth=X` | GET | Get LWLR curves at bandwidth X |

## Screenshots 📸

The app features a warm sun-themed dark UI with:
- Live bandwidth slider affecting predictions in real-time
- Scatter plots with LWLR fit curves
- Actual vs predicted temperature comparison
