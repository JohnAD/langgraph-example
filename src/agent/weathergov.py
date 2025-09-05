# national weather service
import requests
from typing import TypedDict, Any, Optional, List, Dict


class Period(TypedDict):
    number: int
    name: str
    startTime: str
    endTime: str
    isDaytime: bool
    temperature: int
    temperatureUnit: str
    temperatureTrend: Optional[str]
    windSpeed: str
    windDirection: str
    icon: str
    shortForecast: str
    detailedForecast: str

class Properties(TypedDict):
    updated: str
    units: str
    forecastGenerator: str
    generatedAt: str
    updateTime: str
    validTimes: str
    elevation: Dict[str, Any]
    periods: List[Period]

class Forecasts(TypedDict):
    type: str
    geometry: Dict[str, Any]
    properties: Properties
    

def get_weather_forecast(gridId: str, x: str, y: str) -> Optional[Forecasts]:
    """
    Get weather forecast from National Weather Service API.
    
    Args:
        gridId: The grid office identifier (e.g., 'TOP')
        x: Grid X coordinate
        y: Grid Y coordinate
        
    Returns:
        Forecasts object containing the weather forecast data, or None if error
    """
    url = f"https://api.weather.gov/gridpoints/{gridId}/{x},{y}/forecast"
    
    headers = {
        'User-Agent': 'WeatherApp/1.0 (contact@example.com)'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        return None

def summarize_forecasts(forecasts: Optional[Forecasts]) -> str:
    result = ""
    if forecasts is None:
        result = "Unfortunately, the National Weather Service is unable to provide a weather forecast for your location."
    else:
        for period in forecasts['properties']['periods']:
            result += f"{period['name']}: {period['detailedForecast']}\n"
    return result