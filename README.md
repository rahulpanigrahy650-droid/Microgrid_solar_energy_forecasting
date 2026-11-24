# Heliogrid — AI-Powered Solar Microgrid Forecasting

Solar Energy Forecasting is a simple Flask-based dashboard that forecasts solar generation and simulates microgrid battery dispatch. It preserves the original interactive dashboard, frontend parameters, and workflow while using a ten-year hourly dataset and a single Jupyter notebook for the complete machine-learning process.

## Project workflow

```text
10-year hourly CSV dataset
        ↓
Notebook: clean data + engineer features
        ↓
Notebook: chronological train/test split + scaling
        ↓
Notebook: HistGradientBoostingRegressor training
        ↓
Notebook: metrics, plots, and saved model artifacts
        ↓
Flask app loads the saved model
        ↓
Live weather or local fallback + 24-hour forecast
        ↓
Microgrid controller compares solar with load
        ↓
Battery charge / discharge / grid supply
        ↓
Existing interactive dashboard
```

## Technologies actually used

| Technology | Purpose |
|---|---|
| Python | Application and data-processing language |
| Jupyter Notebook | Contains the complete ML training and evaluation workflow |
| Pandas | Reads and prepares hourly CSV data |
| NumPy | Numerical feature engineering and solar/load calculations |
| Scikit-learn | HistGradientBoostingRegressor and StandardScaler |
| Joblib | Saves and loads the trained model and scaler |
| Matplotlib | Creates training evaluation plots |
| Flask | Serves the dashboard and REST API |
| JavaScript, HTML, CSS | Existing interactive dashboard and controls |
| Chart.js | Frontend forecast and battery charts |
| Open-Meteo | Live weather and irradiance data |

XGBoost, TensorFlow, LSTM, Transformer, PyTorch, OpenCV, FastAPI, authentication, confidence intervals, and hardware control are not part of this simplified implementation.

## Folder structure

```text
solar-microgrid-forecast/
├── app.py
├── microgrid_controller.py
├── train_model.ipynb
├── requirements.txt
├── data/solar_weather_dataset.csv
├── models/
│   ├── solar_forecast_model.joblib
│   ├── feature_scaler.joblib
│   └── feature_columns.joblib
├── results/
│   ├── metrics.json
│   ├── actual_vs_predicted.png
│   ├── prediction_scatter.png
│   └── feature_importance.png
└── frontend/
    ├── index.html
    └── Dashboard_Parameters_Guide.pdf
```

## Run the dashboard

From the project root:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`.

## Retrain the model

Start Jupyter from the project root:

```bash
jupyter notebook
```

Open `train_model.ipynb` and run all cells. The notebook reads the ten-year CSV, trains the active model, evaluates it, and refreshes the files in `models/` and `results/`.

## Dataset

The included dataset contains **87,672 hourly rows** covering **2015–2024**, including leap-year hours. It contains the same twelve columns used by the original project: timestamp, time fields, weather variables, GHI, previous solar output, solar generation, and load demand.

The weather and solar columns come from the Open-Meteo historical archive for the default Delhi location. Solar generation is a transparent derived 10 kW PV conversion from GHI. Load demand remains the project’s deterministic household demand profile so that the existing microgrid controller and dashboard remain compatible.

## Important scope note

This is a software simulation and forecasting dashboard. It does not connect to physical inverters, batteries, or real microgrid hardware.
