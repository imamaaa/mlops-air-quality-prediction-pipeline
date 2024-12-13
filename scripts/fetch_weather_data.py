import requests
import json
from datetime import datetime

# Your API Key and endpoint
API_KEY = "119e24626ffc881be87270bae2f7ba40"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
CITY = "Islamabad" 

def fetch_weather_data():
    # Parameters for API call
    params = {
        "q": CITY,
        "appid": API_KEY,
        "units": "metric"  # Fetch data in Celsius
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            # Save data to a file with timestamp
            filename = f"data/weather_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Weather data saved to {filename}")
        else:
            print(f"Failed to fetch data: {response.status_code} - {response.reason}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_weather_data()
