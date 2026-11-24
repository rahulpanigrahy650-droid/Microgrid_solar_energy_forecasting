# Solar Energy Forecastingd — Simplified Handoff

## Run the dashboard

From the project root, install the active dependencies and start the Flask app:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`. The existing dashboard is served by Flask, so do not open `frontend/index.html` directly when using the live API workflow.

## Retrain the model

The complete machine-learning workflow is in `train_model.ipynb`. Start Jupyter from the project root:

```bash
jupyter notebook
```

Open `train_model.ipynb` and run all cells. The notebook loads the ten-year CSV, engineers the same features as the original model, trains `HistGradientBoostingRegressor`, calculates MAE/RMSE/R², saves the Joblib artifacts in `models/`, and regenerates plots in `results/`.

## Preserved dashboard behavior

The original webpage, parameters, API workflow, sun-arc scrubber, weather sliders, location search, appliance toggles, forecast charts, BESS visualization, microgrid flow map, CSV export, and PDF guide are preserved.

## Dataset update

The dataset now contains 87,672 hourly rows covering 2015–2024. The weather and solar-radiation fields come from Open-Meteo’s historical archive for the default Delhi location. Solar generation is a transparent derived 10 kW PV conversion from GHI, and load demand remains the existing deterministic household profile so the dashboard workflow remains compatible.

## Simplified structure

The project now keeps only the active application files: `app.py`, `microgrid_controller.py`, `train_model.ipynb`, `data/`, `models/`, `results/`, `frontend/`, `requirements.txt`, and documentation. Optional TensorFlow/LSTM and XGBoost scripts were removed from the active project structure because they were not used by the dashboard.
