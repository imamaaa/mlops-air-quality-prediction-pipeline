import requests
import json
from dotenv import load_dotenv
from datetime import datetime
import time
import os
import logging
from pathlib import Path

# Configure logging
script_dir = Path(__file__).parent.parent
log_file = script_dir / 'logs' / 'weather_collection.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Configuration
# Load environment variables
load_dotenv()
# Fetch API Key from environment variables
API_KEY = os.getenv("API_KEY")
CITY_CONFIG = {
    'name': 'Islamabad',
    'lat': 33.6844,
    'lon': 73.0479
}

# Windows-style paths using pathlib
DATA_DIR = script_dir / 'data'
WEATHER_DIR = DATA_DIR / 'weather'
POLLUTION_DIR = DATA_DIR / 'pollution'

def setup_directories():
    """Create necessary directories if they don't exist."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    POLLUTION_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Directories created successfully")

def fetch_weather_data():
    """Fetch weather data for Islamabad."""
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": CITY_CONFIG['lat'],
        "lon": CITY_CONFIG['lon'],
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching weather data: {e}")
        return None

def fetch_pollution_data():
    """Fetch air pollution data for Islamabad."""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": CITY_CONFIG['lat'],
        "lon": CITY_CONFIG['lon'],
        "appid": API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching pollution data: {e}")
        return None

def process_weather_data(raw_data):
    """Process and extract relevant weather information."""
    if not raw_data:
        return None
    
    return {
        'city': CITY_CONFIG['name'],
        'timestamp': datetime.utcfromtimestamp(raw_data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
        'temperature': raw_data['main']['temp'],
        'humidity': raw_data['main']['humidity'],
        'pressure': raw_data['main']['pressure'],
        'wind_speed': raw_data['wind']['speed'],
        'weather_condition': raw_data['weather'][0]['main']
    }

def process_pollution_data(raw_data):
    """Process and extract relevant pollution information."""
    if not raw_data or 'list' not in raw_data or not raw_data['list']:
        return None
    
    pollution = raw_data['list'][0]
    return {
        'city': CITY_CONFIG['name'],
        'timestamp': datetime.utcfromtimestamp(pollution['dt']).strftime('%Y-%m-%d %H:%M:%S'),
        'aqi': pollution['main']['aqi'],
        'co': pollution['components']['co'],
        'no2': pollution['components']['no2'],
        'o3': pollution['components']['o3'],
        'pm2_5': pollution['components']['pm2_5'],
        'pm10': pollution['components']['pm10']
    }

def save_data(data, data_type, timestamp):
    """Save collected data to JSON file."""
    if not data:
        return None
    
    directory = WEATHER_DIR if data_type == 'weather' else POLLUTION_DIR
    filename = f"{data_type}_data_{timestamp}.json"
    filepath = directory / filename
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved {data_type} data to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error saving {data_type} data: {e}")
        return None

def run_dvc_commands(filepath):
    """Run DVC commands for the new data file."""
    try:
        os.system(f'dvc add "{filepath}"')
        os.system('dvc push')
        logging.info(f"DVC operations completed for {filepath}")
    except Exception as e:
        logging.error(f"Error in DVC operations: {e}")

def collect_data():
    """Main function to collect both weather and pollution data."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.info(f"Starting data collection for {CITY_CONFIG['name']}")
    
    # Fetch and process weather data
    raw_weather = fetch_weather_data()
    if raw_weather:
        weather_data = process_weather_data(raw_weather)
        weather_file = save_data([weather_data], 'weather', timestamp)
        if weather_file:
            run_dvc_commands(weather_file)
    
    # Add small delay between API calls
    time.sleep(1)
    
    # Fetch and process pollution data
    raw_pollution = fetch_pollution_data()
    if raw_pollution:
        pollution_data = process_pollution_data(raw_pollution)
        pollution_file = save_data([pollution_data], 'pollution', timestamp)
        if pollution_file:
            run_dvc_commands(pollution_file)

if __name__ == "__main__":
    setup_directories()
    collect_data()
