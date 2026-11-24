import os
import json
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import cv2
from skimage.feature import hog as skimage_hog
from flask import Flask, jsonify, request, send_file

from microgrid_controller import BatteryState, decide

FEATURE_COLS = [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "temperature_c",
    "humidity_pct", "cloud_cover_pct", "wind_speed_kmh", "ghi_wm2",
    "prev_solar_kw", "solar_lag_24h", "solar_roll_3h",
]
TARGET_COL = "solar_generation_kw"

def engineer_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["solar_lag_24h"] = df["solar_generation_kw"].shift(24).fillna(0)
    df["solar_roll_3h"] = df["solar_generation_kw"].shift(1).rolling(3, min_periods=1).mean().fillna(0)
    return df

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Weather cache to prevent hitting external rate limits
_WEATHER_CACHE = {}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR
DATA_PATH = PROJECT_DIR / "data" / "solar_weather_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FRONTEND_DIR = PROJECT_DIR / "frontend"

print("Loading dataset & trained AI models...")
_df_raw = pd.read_csv(DATA_PATH)
_df = engineer_features(_df_raw)
_model = joblib.load(MODEL_DIR / "solar_forecast_model.joblib")
_scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")
_weather_bundle = joblib.load(MODEL_DIR / "weather_classifier.joblib")
_weather_model = _weather_bundle["model"]
_weather_classes = _weather_bundle["class_names"]

with open(RESULTS_DIR / "metrics.json") as f:
    _metrics = json.load(f)

print(f"Solar Energy Forecasting Backend Ready! Loaded {len(_df)} training records.")

WEATHER_PROFILES = {
    "evening": {"cloud_cover_pct": 45, "temperature_c": 23, "humidity_pct": 65, "wind_speed_kmh": 10, "ghi_wm2": 250},
    "foggy": {"cloud_cover_pct": 90, "temperature_c": 18, "humidity_pct": 94, "wind_speed_kmh": 5, "ghi_wm2": 80},
    "hail": {"cloud_cover_pct": 85, "temperature_c": 17, "humidity_pct": 82, "wind_speed_kmh": 28, "ghi_wm2": 120},
    "lightning": {"cloud_cover_pct": 95, "temperature_c": 22, "humidity_pct": 88, "wind_speed_kmh": 32, "ghi_wm2": 60},
    "night": {"cloud_cover_pct": 35, "temperature_c": 20, "humidity_pct": 70, "wind_speed_kmh": 8, "ghi_wm2": 0},
    "overcast": {"cloud_cover_pct": 88, "temperature_c": 21, "humidity_pct": 78, "wind_speed_kmh": 12, "ghi_wm2": 140},
    "partly_cloudy": {"cloud_cover_pct": 45, "temperature_c": 26, "humidity_pct": 58, "wind_speed_kmh": 14, "ghi_wm2": 620},
    "rain": {"cloud_cover_pct": 92, "temperature_c": 20, "humidity_pct": 90, "wind_speed_kmh": 18, "ghi_wm2": 110},
    "storm": {"cloud_cover_pct": 98, "temperature_c": 21, "humidity_pct": 86, "wind_speed_kmh": 40, "ghi_wm2": 45},
    "sunny": {"cloud_cover_pct": 8, "temperature_c": 29, "humidity_pct": 42, "wind_speed_kmh": 12, "ghi_wm2": 900},
}


def predict_weather_image(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    feature = skimage_hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).reshape(1, -1)

    prediction_index = int(_weather_model.predict(feature)[0])
    predicted_class = _weather_classes[prediction_index]
    decision = _weather_model.decision_function(feature)
    decision_values = np.asarray(decision).reshape(-1)
    if len(decision_values) > 1:
        ordered = np.sort(decision_values)
        margin = float(ordered[-1] - ordered[-2])
        confidence = round(float(np.clip(50 + margin * 25, 50, 99.9)), 1)
    else:
        confidence = 100.0

    return predicted_class, confidence


def build_feature_row(hour_val, day_of_year, temperature, humidity, cloud_cover, wind_speed, ghi, history=None, prev_solar_kw=None):
    """Build a single model input row using realistic recent solar history."""
    history = [] if history is None else [float(v) for v in history if v is not None]

    if history:
        prev_solar_kw = float(history[-1]) if prev_solar_kw is None else float(prev_solar_kw)
        solar_roll_3h = float(np.mean(history[-3:])) if len(history) >= 3 else prev_solar_kw
        solar_lag_24h = float(history[-24]) if len(history) >= 24 else 0.0
    else:
        prev_solar_kw = float(prev_solar_kw or 0.0)
        solar_roll_3h = prev_solar_kw
        solar_lag_24h = 0.0

    return {
        "hour_sin": np.sin(2 * np.pi * hour_val / 24),
        "hour_cos": np.cos(2 * np.pi * hour_val / 24),
        "day_sin": np.sin(2 * np.pi * day_of_year / 365),
        "day_cos": np.cos(2 * np.pi * day_of_year / 365),
        "temperature_c": float(temperature),
        "humidity_pct": float(humidity),
        "cloud_cover_pct": float(cloud_cover),
        "wind_speed_kmh": float(wind_speed),
        "ghi_wm2": float(ghi),
        "prev_solar_kw": prev_solar_kw,
        "solar_lag_24h": solar_lag_24h,
        "solar_roll_3h": solar_roll_3h,
    }


def predict_row(row: dict) -> float:
    """Evaluates ML model for a single feature dictionary."""
    x = pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    x_scaled = _scaler.transform(x)
    pred = float(_model.predict(x_scaled)[0])
    return max(pred, 0.0)


@app.route("/")
def index():
    dashboard = FRONTEND_DIR / "index.html"
    if dashboard.exists():
        return send_file(dashboard)
    return jsonify(
        system="Solar Energy Forecasting",
        status="ONLINE",
        endpoints=[
            "/api/health",
            "/api/weather/live?lat=28.6139&lon=77.2090",
            "/api/weather/geocode?query=Dubai",
            "/api/metrics",
            "/api/history?hours=48",
            "/api/forecast?hours=24",
            "/api/microgrid?hours=24",
            "/api/download/pdf",
        ],
        documentation="https://github.com/sibasishsahu441/solar-microgrid-forecast"
    )


@app.route("/api/health")
def health():
    return jsonify(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_rows=len(_df),
        model_type=_metrics.get("model", "HistGradientBoostingRegressor"),
        metrics={
            "mae_kw": _metrics.get("mae_kw", 0.0355),
            "rmse_kw": _metrics.get("rmse_kw", 0.0461),
            "r2": _metrics.get("r2", 0.9994)
        }
    )


@app.route("/api/metrics")
def metrics():
    return jsonify(_metrics)


@app.route("/api/weather/geocode")
def geocode():
    """Resolves city names or search queries to geographical coordinates."""
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify(error="Query parameter required"), 400

    # Quick presets for major solar hubs and Indian states
    presets = {
        # International Solar Hubs
        "dubai": {"name": "Dubai, UAE", "lat": 25.2048, "lon": 55.2708, "country": "UAE", "is_solar_hub": True},
        "california": {"name": "Mojave Desert Solar Farm, CA", "lat": 35.0110, "lon": -115.4734, "country": "USA", "is_solar_hub": True},
        "munich": {"name": "Munich, Germany", "lat": 48.1351, "lon": 11.5820, "country": "Germany", "is_solar_hub": False},
        "sydney": {"name": "Sydney, Australia", "lat": -33.8688, "lon": 151.2093, "country": "Australia", "is_solar_hub": True},
        "tokyo": {"name": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503, "country": "Japan", "is_solar_hub": False},
        "london": {"name": "London, UK", "lat": 51.5074, "lon": -0.1278, "country": "UK", "is_solar_hub": False},
        
        # Indian States and Solar Hubs
        "rajasthan": {"name": "Jaipur, Rajasthan", "lat": 26.9124, "lon": 75.7873, "country": "India", "is_solar_hub": True},
        "delhi": {"name": "New Delhi, Delhi", "lat": 28.6139, "lon": 77.2090, "country": "India", "is_solar_hub": False},
        "new delhi": {"name": "New Delhi, Delhi", "lat": 28.6139, "lon": 77.2090, "country": "India", "is_solar_hub": False},
        "maharashtra": {"name": "Mumbai, Maharashtra", "lat": 19.0760, "lon": 72.8777, "country": "India", "is_solar_hub": True},
        "mumbai": {"name": "Mumbai, Maharashtra", "lat": 19.0760, "lon": 72.8777, "country": "India", "is_solar_hub": True},
        "karnataka": {"name": "Bangalore, Karnataka", "lat": 12.9716, "lon": 77.5946, "country": "India", "is_solar_hub": True},
        "bangalore": {"name": "Bangalore, Karnataka", "lat": 12.9716, "lon": 77.5946, "country": "India", "is_solar_hub": True},
        "tamil nadu": {"name": "Chennai, Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "country": "India", "is_solar_hub": True},
        "tamil": {"name": "Chennai, Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "country": "India", "is_solar_hub": True},
        "chennai": {"name": "Chennai, Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "country": "India", "is_solar_hub": True},
        "telangana": {"name": "Hyderabad, Telangana", "lat": 17.3850, "lon": 78.4867, "country": "India", "is_solar_hub": True},
        "hyderabad": {"name": "Hyderabad, Telangana", "lat": 17.3850, "lon": 78.4867, "country": "India", "is_solar_hub": True},
        "andhra pradesh": {"name": "Visakhapatnam, Andhra Pradesh", "lat": 17.6869, "lon": 83.2185, "country": "India", "is_solar_hub": True},
        "visakhapatnam": {"name": "Visakhapatnam, Andhra Pradesh", "lat": 17.6869, "lon": 83.2185, "country": "India", "is_solar_hub": True},
        "uttar pradesh": {"name": "Lucknow, Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "country": "India", "is_solar_hub": False},
        "lucknow": {"name": "Lucknow, Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "country": "India", "is_solar_hub": False},
        "punjab": {"name": "Chandigarh, Punjab", "lat": 30.7333, "lon": 76.8667, "country": "India", "is_solar_hub": False},
        "chandigarh": {"name": "Chandigarh, Punjab", "lat": 30.7333, "lon": 76.8667, "country": "India", "is_solar_hub": False},
        "kerala": {"name": "Kochi, Kerala", "lat": 9.9312, "lon": 76.2673, "country": "India", "is_solar_hub": False},
        "kochi": {"name": "Kochi, Kerala", "lat": 9.9312, "lon": 76.2673, "country": "India", "is_solar_hub": False},
        "goa": {"name": "Goa, Goa", "lat": 15.2993, "lon": 73.8243, "country": "India", "is_solar_hub": False},
        "bihar": {"name": "Patna, Bihar", "lat": 25.5941, "lon": 85.1376, "country": "India", "is_solar_hub": False},
        "patna": {"name": "Patna, Bihar", "lat": 25.5941, "lon": 85.1376, "country": "India", "is_solar_hub": False},
        "west bengal": {"name": "Kolkata, West Bengal", "lat": 22.5726, "lon": 88.3639, "country": "India", "is_solar_hub": False},
        "kolkata": {"name": "Kolkata, West Bengal", "lat": 22.5726, "lon": 88.3639, "country": "India", "is_solar_hub": False},
        "jharkhand": {"name": "Ranchi, Jharkhand", "lat": 23.3441, "lon": 85.3096, "country": "India", "is_solar_hub": False},
        "ranchi": {"name": "Ranchi, Jharkhand", "lat": 23.3441, "lon": 85.3096, "country": "India", "is_solar_hub": False},
        "madhya pradesh": {"name": "Bhopal, Madhya Pradesh", "lat": 23.1815, "lon": 79.9864, "country": "India", "is_solar_hub": False},
        "bhopal": {"name": "Bhopal, Madhya Pradesh", "lat": 23.1815, "lon": 79.9864, "country": "India", "is_solar_hub": False},
        "gujarati": {"name": "Ahmedabad, Gujarat", "lat": 23.0225, "lon": 72.5714, "country": "India", "is_solar_hub": True},
        "ahmedabad": {"name": "Ahmedabad, Gujarat", "lat": 23.0225, "lon": 72.5714, "country": "India", "is_solar_hub": True},
    }

    q_lower = query.lower()
    for k, v in presets.items():
        if k in q_lower or q_lower in k:
            return jsonify(results=[v])

    # Query Open-Meteo Geocoding API
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(query)}&count=5&language=en&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Solar Energy Forecasting-EMS/2.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            results = []
            for item in data.get("results", []):
                results.append({
                    "name": f"{item.get('name')}, {item.get('admin1', item.get('country', ''))}",
                    "lat": item.get("latitude"),
                    "lon": item.get("longitude"),
                    "country": item.get("country", ""),
                    "timezone": item.get("timezone", "UTC")
                })
            return jsonify(results=results)
    except Exception as e:
        # Fallback to default
        return jsonify(results=[{"name": query.title(), "lat": 28.6139, "lon": 77.2090, "country": "Global", "error": str(e)}])


@app.route("/api/weather/live")
def live_weather():
    """
    Fetches real-time solar irradiance (GHI, DNI, DHI), cloud cover,
    temperature, humidity, and wind from satellite meteorology APIs.
    """
    try:
        lat = float(request.args.get("lat", 28.6139))
        lon = float(request.args.get("lon", 77.2090))
    except ValueError:
        return jsonify(error="Invalid lat/lon parameters"), 400

    cache_key = f"{round(lat, 2)}_{round(lon, 2)}"
    now_ts = datetime.now().timestamp()

    if cache_key in _WEATHER_CACHE:
        cached_data, cached_time = _WEATHER_CACHE[cache_key]
        if now_ts - cached_time < 300:  # 5 min cache
            return jsonify(cached_data)

    try:
        # Open-Meteo Solar & Weather API (high resolution solar irradiance)
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,surface_pressure,direct_radiation,diffuse_radiation,shortwave_radiation,is_day"
            f"&hourly=temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,shortwave_radiation,direct_radiation,diffuse_radiation"
            f"&forecast_days=2&timezone=auto"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SolarEnergyForecasting-EMS/2.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw = json.loads(resp.read().decode())

        current = raw.get("current", {})
        hourly = raw.get("hourly", {})

        ghi = float(current.get("shortwave_radiation", 0.0) or 0.0)
        dni = float(current.get("direct_radiation", 0.0) or 0.0)
        dhi = float(current.get("diffuse_radiation", 0.0) or 0.0)
        temp_c = float(current.get("temperature_2m", 25.0) or 25.0)
        humidity = float(current.get("relative_humidity_2m", 50.0) or 50.0)
        clouds = float(current.get("cloud_cover", 20.0) or 20.0)
        wind_kmh = float(current.get("wind_speed_10m", 12.0) or 12.0)
        is_day = int(current.get("is_day", 1))

        # Format 24-hour ahead forecast array
        hourly_times = hourly.get("time", [])[:24]
        hourly_ghi = hourly.get("shortwave_radiation", [])[:24]
        hourly_temp = hourly.get("temperature_2m", [])[:24]
        hourly_clouds = hourly.get("cloud_cover", [])[:24]
        hourly_humidity = hourly.get("relative_humidity_2m", [])[:24]
        hourly_wind = hourly.get("wind_speed_10m", [])[:24]

        forecast_24h = []
        now_dt = datetime.now()
        day_of_year = now_dt.timetuple().tm_yday
        history_tail = list(_df["solar_generation_kw"].tail(72).tolist())

        for i in range(len(hourly_times)):
            h_dt = datetime.fromisoformat(hourly_times[i]) if "T" in hourly_times[i] else now_dt
            hour_val = h_dt.hour
            h_ghi = float(hourly_ghi[i] or 0.0) if i < len(hourly_ghi) else 0.0
            h_temp = float(hourly_temp[i] or 25.0) if i < len(hourly_temp) else 25.0
            h_cloud = float(hourly_clouds[i] or 20.0) if i < len(hourly_clouds) else 20.0
            h_hum = float(hourly_humidity[i] or 50.0) if i < len(hourly_humidity) else 50.0
            h_w = float(hourly_wind[i] or 12.0) if i < len(hourly_wind) else 12.0

            feature_row = build_feature_row(
                hour_val=hour_val,
                day_of_year=day_of_year,
                temperature=h_temp,
                humidity=h_hum,
                cloud_cover=h_cloud,
                wind_speed=h_w,
                ghi=h_ghi,
                history=history_tail,
            )
            pred_kw = predict_row(feature_row)
            # Microgrid load demand for this hour
            load_kw = max(3.2 + 2.0 * np.exp(-((hour_val - 8) ** 2) / 4) + 3.2 * np.exp(-((hour_val - 19.5) ** 2) / 6) + 0.6 * np.exp(-((hour_val - 13) ** 2) / 8), 0.4)

            forecast_24h.append({
                "time": hourly_times[i],
                "hour": hour_val,
                "ghi_wm2": round(h_ghi, 1),
                "temperature_c": round(h_temp, 1),
                "cloud_cover_pct": round(h_cloud, 1),
                "predicted_solar_kw": round(pred_kw, 3),
                "load_demand_kw": round(float(load_kw), 3)
            })
            history_tail.append(pred_kw)

        response_data = {
            "source": "Open-Meteo Satellite Solar API + NASA POWER Compatible",
            "lat": lat,
            "lon": lon,
            "current": {
                "ghi_wm2": round(ghi, 1),
                "dni_wm2": round(dni, 1),
                "dhi_wm2": round(dhi, 1),
                "temperature_c": round(temp_c, 1),
                "humidity_pct": round(humidity, 1),
                "cloud_cover_pct": round(clouds, 1),
                "wind_speed_kmh": round(wind_kmh, 1),
                "is_day": is_day,
            },
            "forecast_24h": forecast_24h
        }

        _WEATHER_CACHE[cache_key] = (response_data, now_ts)
        return jsonify(response_data)

    except Exception as e:
        # Graceful fallback simulation if external network is unavailable
        now_dt = datetime.now()
        h = now_dt.hour
        elev_est = max(0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0
        est_ghi = elev_est * 850 * (1 - 0.75 * ((25 / 100) ** 1.3))
        return jsonify({
            "source": "Solar Energy Forecasting Local Mathematical Physics Simulation (Network Fallback)",
            "lat": lat,
            "lon": lon,
            "error_note": str(e),
            "current": {
                "ghi_wm2": round(est_ghi, 1),
                "dni_wm2": round(est_ghi * 0.8, 1),
                "dhi_wm2": round(est_ghi * 0.2, 1),
                "temperature_c": 28.0,
                "humidity_pct": 55.0,
                "cloud_cover_pct": 25.0,
                "wind_speed_kmh": 14.0,
                "is_day": 1 if 6 <= h <= 18 else 0
            },
            "forecast_24h": []
        })


@app.route("/api/history")
def history():
    hours = int(request.args.get("hours", 48))
    tail = _df.tail(hours)
    cols = ["timestamp", "temperature_c", "humidity_pct", "cloud_cover_pct",
            "wind_speed_kmh", "ghi_wm2", "solar_generation_kw", "load_demand_kw"]
    return jsonify(tail[cols].to_dict(orient="records"))


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    hour = float(body.get("hour", 12))
    day_of_year = float(body.get("day_of_year", 172))
    history = body.get("history")
    if history is not None:
        history = [float(v) for v in history]
    else:
        history = list(_df["solar_generation_kw"].tail(72).tolist())

    row = build_feature_row(
        hour_val=float(hour),
        day_of_year=float(day_of_year),
        temperature=float(body.get("temperature", 28)),
        humidity=float(body.get("humidity", 60)),
        cloud_cover=float(body.get("cloud_cover", 20)),
        wind_speed=float(body.get("wind_speed", 12)),
        ghi=float(body.get("ghi", 600)),
        history=history,
        prev_solar_kw=float(body.get("prev_solar", history[-1] if history else 3.0)),
    )
    pred = predict_row(row)
    return jsonify(predicted_solar_kw=round(pred, 3))


@app.route("/api/weather/image", methods=["POST"])
def predict_weather_from_image():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Upload an image using the image field."), 400

    try:
        predicted_class, confidence = predict_weather_image(uploaded_file.read())
        profile = WEATHER_PROFILES.get(predicted_class, WEATHER_PROFILES["partly_cloudy"])
        return jsonify(
            predicted_class=predicted_class,
            confidence_pct=confidence,
            profile=profile,
            source="Cloud Weather HOG + SVM image classifier",
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        app.logger.exception("Weather image prediction failed")
        return jsonify(error=f"Could not classify the image: {exc}"), 500


@app.route("/api/microgrid")
def microgrid():
    hours = int(request.args.get("hours", 24))
    lat = float(request.args.get("lat", 28.6139))
    lon = float(request.args.get("lon", 77.2090))

    # Reuse live weather forecast if available
    w_data = live_weather().json
    fc = w_data.get("forecast_24h", [])

    battery = BatteryState()
    out = []
    if fc:
        for step in fc[:hours]:
            d = decide(step["predicted_solar_kw"], step["load_demand_kw"], battery)
            d["hour"] = step["hour"]
            d["time"] = step.get("time")
            out.append(d)
    return jsonify(out)


@app.route("/api/download/pdf")
def download_pdf():
    pdf_path = FRONTEND_DIR / "Dashboard_Parameters_Guide.pdf"
    if pdf_path.exists():
        return send_file(pdf_path, as_attachment=True, download_name="SolarEnergyForecasting_Parameters_Guide.pdf")
    return jsonify(error="PDF not generated yet. Run generate_pdf_report.py first."), 404


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", port=int(os.getenv("PORT", "5000")), host="0.0.0.0")
