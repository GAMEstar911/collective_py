import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# OpenWeatherMap API Key (Replace with your actual key)
API_KEY = "04eed84a2ab68c4bccb3b0c3d1909f19"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Function to fetch weather data
def get_weather(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",  # Get temperature in Celsius
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        weather_info = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
        }
        return weather_info
    else:
        return {"error": "City not found"}

# API route to get weather details
@app.route("/weather", methods=["GET"])
def weather():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "City parameter is required"}), 400

    weather_data = get_weather(city)
    return jsonify(weather_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
