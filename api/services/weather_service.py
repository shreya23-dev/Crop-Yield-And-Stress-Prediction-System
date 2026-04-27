import httpx
import numpy as np
import datetime
import calendar

async def fetch_weather_data(lat: float, lon: float, year: int) -> np.ndarray:
    """
    Fetch weather data from Open-Meteo API for the Kharif season (June 1 to Nov 30).
    Returns a numpy array of shape (22, 3) representing 22 weeks of:
    [temp_mean, temp_max, rainfall].
    """
    start_date = f"{year}-06-01"
    end_date = f"{year}-11-30"
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "temperature_2m_max", "precipitation_sum"],
        "timezone": "auto"
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            t_mean = daily.get("temperature_2m_mean", [])
            t_max = daily.get("temperature_2m_max", [])
            rain = daily.get("precipitation_sum", [])
            
            if not t_mean or len(t_mean) < 183:
                return _fallback_weather(year)
                
            # Aggregate daily to 22 weeks
            weekly_data = []
            for w in range(22):
                start_i = w * 7
                end_i = min((w + 1) * 7, len(t_mean))
                if start_i >= len(t_mean):
                    break
                    
                w_tmean = [t for t in t_mean[start_i:end_i] if t is not None]
                w_tmax = [t for t in t_max[start_i:end_i] if t is not None]
                w_rain = [r for r in rain[start_i:end_i] if r is not None]
                
                week_tmean = np.mean(w_tmean) if w_tmean else 28.0
                week_tmax = np.mean(w_tmax) if w_tmax else 33.0
                week_rain = np.sum(w_rain) if w_rain else 0.0
                weekly_data.append([week_tmean, week_tmax, week_rain])
                
            # Pad if less than 22 weeks (e.g. current year not done)
            while len(weekly_data) < 22:
                weekly_data.append([28.0, 33.0, 0.0])
                
            return np.array(weekly_data, dtype=np.float32)
            
    except Exception as e:
        print(f"Weather API failed: {e}")
        return _fallback_weather(year)

def _fallback_weather(year: int) -> np.ndarray:
    """Returns typical climatology if API fails."""
    temp_mean = np.full(22, 28.0, dtype=np.float32)
    temp_max = np.full(22, 33.0, dtype=np.float32)
    rain = np.array([10, 25, 40, 60, 80, 90, 85, 70, 55, 40,
                    30, 20, 15, 12, 10, 8, 6, 5, 5, 4, 4, 3], dtype=np.float32)
    return np.column_stack([temp_mean, temp_max, rain])

def compute_thermal_stress(weather_data: np.ndarray, crop: str) -> float:
    t_mean = weather_data[:, 0].mean()
    t_max = weather_data[:, 1].mean()
    
    # Simple thermal stress heuristic based on avg and max
    # if T_mean > 32 or T_max > 40 it's stressful
    stress = 0.0
    if t_mean > 32: stress += (t_mean - 32) * 0.1
    if t_max > 40: stress += (t_max - 40) * 0.1
    return min(1.0, max(0.0, stress))

def compute_water_stress(weather_data: np.ndarray, lat: float, crop: str) -> float:
    total_rain = weather_data[:, 2].sum()
    
    # Simple water stress heuristic
    # if rain < 400mm, high stress for Rice, less for Bajra
    req = {"Rice": 800, "Jowar": 400, "Bajra": 300, "Soyabean": 500, "Cotton(lint)": 600}
    crop_req = req.get(crop, 500)
    
    if total_rain >= crop_req:
        return 0.0
    
    stress = (crop_req - total_rain) / crop_req
    return min(1.0, max(0.0, float(stress)))

def summarize_weather(weather_data: np.ndarray) -> dict:
    t_mean = float(weather_data[:, 0].mean())
    t_max = float(weather_data[:, 1].max())
    total_rain = float(weather_data[:, 2].sum())
    
    dry_weeks = int(np.sum(weather_data[:, 2] < 5.0))
    desc = "Normal conditions."
    if dry_weeks > 5:
        desc = f"Below-average rainfall with {dry_weeks} dry weeks detected."
    elif total_rain > 1000:
        desc = "Heavy monsoon rainfall recorded over the season."
        
    return {
        "avg_temperature": round(t_mean, 1),
        "max_temperature": round(t_max, 1),
        "total_rainfall": round(total_rain, 1),
        "rainfall_unit": "mm",
        "dry_weeks": dry_weeks,
        "description": desc
    }
