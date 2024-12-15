from flask import Flask, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import requests
import numpy as np
import logging
from datetime import datetime
from prometheus_client import Counter, generate_latest
from flask import Response

# Initialize the Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained LSTM model
MODEL_PATH = "E:/Semester 8/MLOps/Project_Task_1/Deployment/lstm_model.h5"
custom_objects = {
    "mse": MeanSquaredError(),
    "mae": MeanAbsoluteError()
}
model = load_model(MODEL_PATH, custom_objects=custom_objects)
logger.info(f"Loaded model from {MODEL_PATH}")

# Define the feature order expected by the model
FEATURE_ORDER = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi']
SEQUENCE_LENGTH = 6  # Match the sequence length used in training

# OpenWeather API configuration
API_KEY = "119e24626ffc881be87270bae2f7ba40"  # Replace with your valid OpenWeather API key
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
CITY_CONFIG = {
    "lat": 33.6844,
    "lon": 73.0479,
    "name": "Islamabad"
}


# Create a counter for predictions
prediction_requests = Counter('prediction_requests_total', 'Total Prediction Requests')
prediction_success = Counter('prediction_success_total', 'Successful Predictions')
prediction_failure = Counter('prediction_failure_total', 'Failed Predictions')

def fetch_live_data():
    """Fetch live weather and pollution data for Islamabad."""
    try:
        # Fetch weather data
        weather_params = {
            "lat": CITY_CONFIG["lat"],
            "lon": CITY_CONFIG["lon"],
            "appid": API_KEY,
            "units": "metric"
        }
        weather_response = requests.get(WEATHER_URL, params=weather_params)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        # Fetch pollution data
        pollution_params = {
            "lat": CITY_CONFIG["lat"],
            "lon": CITY_CONFIG["lon"],
            "appid": API_KEY
        }
        pollution_response = requests.get(POLLUTION_URL, params=pollution_params)
        pollution_response.raise_for_status()
        pollution_data = pollution_response.json()

        return weather_data, pollution_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching live data: {e}")
        return None, None

def process_live_data(weather_data, pollution_data):
    """Process live data into the required format."""
    try:
        # Extract weather features
        processed_data = {
            "temperature": weather_data["main"]["temp"],
            "humidity": weather_data["main"]["humidity"],
            "pressure": weather_data["main"]["pressure"],
            "wind_speed": weather_data["wind"]["speed"],
            "aqi": 0  # Placeholder for AQI
        }

        # Extract AQI from pollution data
        if pollution_data and "list" in pollution_data and pollution_data["list"]:
            processed_data["aqi"] = pollution_data["list"][0]["main"]["aqi"]

        return processed_data

    except KeyError as e:
        logger.error(f"KeyError in processing data: {e}")
        return None

@app.route("/", methods=["GET"])
def home():
    """Default route."""
    return jsonify({"message": "Welcome to the LSTM Prediction API! Available routes: /health, /predict-live"}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), content_type='text/plain')

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route("/predict-live", methods=["GET"])
def predict_live():
    """
    Fetch live data for Islamabad, process it, and return AQI prediction.
    """
    prediction_requests.inc()  # Increment the prediction requests counter
    try:
        # Fetch live data
        weather_data, pollution_data = fetch_live_data()
        if not weather_data or not pollution_data:
            prediction_failure.inc()  # Increment the failure counter
            return jsonify({"error": "Failed to fetch live data"}), 500

        # Process the data
        live_data = process_live_data(weather_data, pollution_data)
        if not live_data:
            prediction_failure.inc()  # Increment the failure counter
            return jsonify({"error": "Failed to process live data"}), 500

        # Prepare input sequence for the LSTM model
        input_sequence = [[live_data[feature] for feature in FEATURE_ORDER]] * SEQUENCE_LENGTH
        input_sequence = np.array(input_sequence).reshape(1, SEQUENCE_LENGTH, len(FEATURE_ORDER))

        # Predict AQI
        prediction = model.predict(input_sequence).flatten()[0]
        prediction = float(prediction)

        # Increment the success counter
        prediction_success.inc()

        # Return the prediction
        return jsonify({
            "city": CITY_CONFIG["name"],
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "predicted_aqi": prediction,
            "live_data": live_data
        }), 200

    except Exception as e:
        prediction_failure.inc()  # Increment the failure counter
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
