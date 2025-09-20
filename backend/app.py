from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import math
import os
import requests
from datetime import datetime, timedelta
import threading
import time
from typing import List, Dict, Optional, Tuple
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from geopy.distance import geodesic

# Import our enhanced ML model components
try:
    from ship_routing_model import (
        ShipRoutingModel, WeatherDataProvider, 
        OptimalRouteGenerator, RoutePoint, WeatherData, LandMaskService
    )
    print("✅ Enhanced ML model components loaded successfully")
except ImportError as e:
    print(f"❌ ML model components not found: {e}")
    print("Using fallback implementation...")
    ShipRoutingModel = None
    LandMaskService = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedWeatherService:
    """Enhanced weather service with multiple data sources"""
    
    def __init__(self):
        self.openweather_api_key = os.environ.get('OPENWEATHER_API_KEY')
        self.marine_api_key = os.environ.get('MARINE_API_KEY')
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour cache
        
    def get_weather_forecast(self, lat: float, lon: float, hours_ahead: int = 0) -> Dict:
        """Get weather forecast for specified location and time"""
        cache_key = f"{lat:.2f},{lon:.2f},{hours_ahead}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return cached_data
        
        try:
            # Try real API first
            if self.openweather_api_key:
                weather_data = self._get_openweather_data(lat, lon, hours_ahead)
            else:
                weather_data = self._generate_synthetic_weather(lat, lon, hours_ahead)
                
            # Cache the result
            self.cache[cache_key] = (weather_data, time.time())
            return weather_data
            
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return self._generate_synthetic_weather(lat, lon, hours_ahead)
    
    def _get_openweather_data(self, lat: float, lon: float, hours_ahead: int = 0) -> Dict:
        """Get weather data from OpenWeatherMap API"""
        base_url = "http://api.openweathermap.org/data/2.5"
        
        if hours_ahead == 0:
            url = f"{base_url}/weather"
        else:
            url = f"{base_url}/forecast"
        
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.openweather_api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}")
            
        data = response.json()
        
        if hours_ahead == 0:
            return self._format_current_weather(data, lat, lon)
        else:
            return self._format_forecast_weather(data, lat, lon, hours_ahead)
    
    def _format_current_weather(self, data: Dict, lat: float, lon: float) -> Dict:
        """Format current weather data"""
        wind = data.get('wind', {})
        main = data.get('main', {})
        
        return {
            'lat': lat,
            'lon': lon,
            'wind_speed': wind.get('speed', 0) * 1.94384,  # m/s to knots
            'wind_direction': wind.get('deg', 0),
            'wave_height': self._estimate_wave_height_from_wind(wind.get('speed', 0)),
            'wave_period': np.random.uniform(4, 12),
            'wave_direction': wind.get('deg', 0) + np.random.uniform(-30, 30),
            'temperature': main.get('temp', 20),
            'pressure': main.get('pressure', 1013),
            'humidity': main.get('humidity', 50),
            'visibility': data.get('visibility', 10000) / 1000,
            'weather_condition': data.get('weather', [{}])[0].get('main', 'Clear'),
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_forecast_weather(self, data: Dict, lat: float, lon: float, hours_ahead: int) -> Dict:
        """Format forecast weather data"""
        forecast_list = data.get('list', [])
        target_index = min(hours_ahead // 3, len(forecast_list) - 1)
        forecast = forecast_list[target_index] if forecast_list else {}
        
        wind = forecast.get('wind', {})
        main = forecast.get('main', {})
        
        return {
            'lat': lat,
            'lon': lon,
            'wind_speed': wind.get('speed', 0) * 1.94384,
            'wind_direction': wind.get('deg', 0),
            'wave_height': self._estimate_wave_height_from_wind(wind.get('speed', 0)),
            'wave_period': np.random.uniform(4, 12),
            'wave_direction': wind.get('deg', 0) + np.random.uniform(-30, 30),
            'temperature': main.get('temp', 20),
            'pressure': main.get('pressure', 1013),
            'humidity': main.get('humidity', 50),
            'visibility': 10,
            'weather_condition': forecast.get('weather', [{}])[0].get('main', 'Clear'),
            'timestamp': (datetime.now() + timedelta(hours=hours_ahead)).isoformat()
        }
    
    def _estimate_wave_height_from_wind(self, wind_speed_ms: float) -> float:
        """Estimate wave height from wind speed"""
        wind_speed_knots = wind_speed_ms * 1.94384
        
        if wind_speed_knots < 4:
            return np.random.uniform(0.1, 0.3)
        elif wind_speed_knots < 11:
            return np.random.uniform(0.3, 0.8)
        elif wind_speed_knots < 16:
            return np.random.uniform(0.8, 1.5)
        elif wind_speed_knots < 22:
            return np.random.uniform(1.5, 2.5)
        elif wind_speed_knots < 28:
            return np.random.uniform(2.5, 4.0)
        elif wind_speed_knots < 34:
            return np.random.uniform(4.0, 6.0)
        else:
            return np.random.uniform(6.0, 12.0)
    
    def _generate_synthetic_weather(self, lat: float, lon: float, hours_ahead: int = 0) -> Dict:
        """Generate realistic synthetic weather data"""
        lat_factor = abs(lat) / 90.0
        
        current_time = datetime.now() + timedelta(hours=hours_ahead)
        day_of_year = current_time.timetuple().tm_yday
        season_factor = math.sin(2 * math.pi * day_of_year / 365)
        
        # Enhanced monsoon effects for Indian Ocean
        monsoon_factor = 0
        if 5 <= lat <= 25 and 65 <= lon <= 95:
            if 150 <= day_of_year <= 270:  # SW Monsoon
                monsoon_factor = 1.5
            elif day_of_year >= 275 or day_of_year <= 31:  # NE Monsoon
                monsoon_factor = 1.2
        
        base_wind = 10 + lat_factor * 15 + monsoon_factor * 8
        base_wave = 1.5 + lat_factor * 2.5 + monsoon_factor * 1.5
        
        weather_system = np.random.choice(['calm', 'normal', 'rough', 'storm'], 
                                        p=[0.25, 0.50, 0.20, 0.05])
        
        multipliers = {'calm': 0.3, 'normal': 1.0, 'rough': 1.8, 'storm': 3.0}
        system_mult = multipliers[weather_system]
        
        wind_speed = max(0, base_wind * system_mult + season_factor * 8)
        wave_height = max(0.1, base_wave * system_mult + season_factor * 1.5)
        
        return {
            'lat': lat,
            'lon': lon,
            'wind_speed': wind_speed,
            'wind_direction': np.random.uniform(0, 360),
            'wave_height': wave_height,
            'wave_period': np.random.uniform(3, 15),
            'wave_direction': np.random.uniform(0, 360),
            'temperature': 15 + season_factor * 20 - lat_factor * 15,
            'pressure': 1013 + np.random.normal(0, 20),
            'humidity': max(30, min(100, 70 + np.random.normal(0, 15))),
            'visibility': max(1, min(20, 12 - system_mult * 3 + np.random.normal(0, 2))),
            'weather_condition': weather_system.capitalize(),
            'timestamp': current_time.isoformat()
        }

class MLRouteOptimizer:
    """Enhanced ML-based route optimizer with land avoidance"""
    
    def __init__(self):
        self.model = None
        self.weather_provider = None
        self.route_generator = None
        self.is_initialized = False
        self.land_mask = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the enhanced ML model in a separate thread"""
        def init():
            try:
                if ShipRoutingModel is not None:
                    logger.info("🤖 Initializing enhanced ML routing model...")
                    self.weather_provider = WeatherDataProvider()
                    self.model = ShipRoutingModel()
                    self.land_mask = LandMaskService()
                    
                    model_path = "enhanced_ship_routing_model"
                    if os.path.exists(f"{model_path}_model.h5"):
                        logger.info("📂 Loading pre-trained enhanced model...")
                        self.model.load_model(model_path)
                    else:
                        logger.info("🎓 Training new enhanced model with 20,000 samples...")
                        logger.info("⏱️  This will take 5-10 minutes. Please be patient...")
                        self.model.train(self.weather_provider, num_samples=20000)
                        self.model.save_model(model_path)
                    
                    self.route_generator = OptimalRouteGenerator(self.model, self.weather_provider)
                    self.is_initialized = True
                    logger.info("✅ Enhanced ML routing model with land avoidance initialized successfully!")
                else:
                    logger.warning("⚠️  Enhanced ML model components not available, using fallback routing")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize enhanced ML model: {e}")
                
        # Initialize in background thread
        threading.Thread(target=init, daemon=True).start()
    
    def generate_ml_route(self, start_lat: float, start_lon: float, 
                         end_lat: float, end_lon: float, 
                         ship_type: str, departure_time: datetime = None) -> List[List[float]]:
        """Generate route using enhanced ML model with land avoidance"""
        if not self.is_initialized or self.route_generator is None:
            raise Exception("Enhanced ML model not initialized yet, please wait or use fallback routing")
        
        try:
            logger.info(f"🚢 Generating AI route with land avoidance from ({start_lat:.3f}, {start_lon:.3f}) to ({end_lat:.3f}, {end_lon:.3f})")
            
            route_points = self.route_generator.generate_route(
                start_lat, start_lon, end_lat, end_lon, ship_type, departure_time
            )
            
            # Convert RoutePoint objects to coordinate lists
            route_coords = [[point.lon, point.lat] for point in route_points]
            
            logger.info(f"✅ Generated optimized route with {len(route_coords)} waypoints, avoiding land masses")
            return route_coords
            
        except Exception as e:
            logger.error(f"❌ Enhanced ML route generation failed: {e}")
            raise

class FallbackRouteOptimizer:
    """Enhanced physics-based fallback route optimizer with land avoidance"""
    
    def __init__(self, weather_service: EnhancedWeatherService):
        self.weather_service = weather_service
        self.land_mask = LandMaskService() if LandMaskService else None
    
    def generate_route(self, start_lat: float, start_lon: float,
                      end_lat: float, end_lon: float,
                      ship_type: str, departure_time: datetime = None) -> Tuple[List[List[float]], float, float]:
        """Generate route using enhanced physics-based approach with land avoidance"""
        if departure_time is None:
            departure_time = datetime.now()
        
        # Ensure start and end points are in water if land mask is available
        if self.land_mask:
            if self.land_mask.is_on_land(start_lat, start_lon):
                start_lat, start_lon = self.land_mask.find_nearest_water(start_lat, start_lon)
                logger.info(f"Moved start point to water: {start_lat:.4f}, {start_lon:.4f}")
            
            if self.land_mask.is_on_land(end_lat, end_lon):
                end_lat, end_lon = self.land_mask.find_nearest_water(end_lat, end_lon)
                logger.info(f"Moved end point to water: {end_lat:.4f}, {end_lon:.4f}")
        
        # Ship characteristics
        ship_speeds = {"passenger ship": 22, "cargo ship": 16, "tanker": 12}
        base_speed = ship_speeds.get(ship_type.lower(), 16)
        
        route = []
        total_distance = 0
        total_time = 0
        current_lat, current_lon = start_lat, start_lon
        current_time = departure_time
        
        max_iterations = 100  # Reduced for more direct routes
        min_step_km = 100     # Minimum step size in km
        max_step_km = 300     # Maximum step size in km
        
        logger.info(f"Generating physics-based route with land avoidance...")
        
        for iteration in range(max_iterations):
            # Calculate distance to destination
            distance_to_dest = geodesic((current_lat, current_lon), (end_lat, end_lon)).kilometers
            
            if distance_to_dest < 75:  # Within 75km of destination
                route.append([end_lon, end_lat])
                total_distance += distance_to_dest
                total_time += distance_to_dest / (base_speed * 1.852)
                break
            
            # Get weather conditions
            hours_ahead = int((current_time - departure_time).total_seconds() / 3600)
            weather = self.weather_service.get_weather_forecast(current_lat, current_lon, hours_ahead)
            
            # Calculate optimal direction with land avoidance
            bearing_to_dest = self._calculate_bearing(current_lat, current_lon, end_lat, end_lon)
            
            # Check if direct path to destination is blocked by land
            if self.land_mask:
                # Sample points along direct path
                num_samples = max(3, int(distance_to_dest / 100))  # Sample every ~100km
                direct_path_clear = True
                
                for i in range(1, num_samples):
                    t = i / num_samples
                    sample_lat = current_lat + (end_lat - current_lat) * t * 0.3  # Check 30% of the way
                    sample_lon = current_lon + (end_lon - current_lon) * t * 0.3
                    
                    if self.land_mask.is_on_land(sample_lat, sample_lon):
                        direct_path_clear = False
                        break
                
                if not direct_path_clear:
                    # Try alternative bearings to avoid land
                    alternative_bearings = [
                        bearing_to_dest + math.pi/6,   # +30 degrees
                        bearing_to_dest - math.pi/6,   # -30 degrees
                        bearing_to_dest + math.pi/3,   # +60 degrees
                        bearing_to_dest - math.pi/3,   # -60 degrees
                    ]
                    
                    for alt_bearing in alternative_bearings:
                        # Test this bearing
                        test_step_km = min(max_step_km, distance_to_dest * 0.4)
                        test_lat = current_lat + (test_step_km / 111.32) * math.cos(alt_bearing)
                        test_lon = current_lon + (test_step_km / (111.32 * math.cos(math.radians(current_lat)))) * math.sin(alt_bearing)
                        
                        if not self.land_mask.is_on_land(test_lat, test_lon):
                            bearing_to_dest = alt_bearing
                            break
            
            # Weather influence on routing (reduced to avoid excessive deviation)
            wind_direction = math.radians(weather['wind_direction'])
            wind_speed = weather['wind_speed']
            wave_height = weather['wave_height']
            
            wind_factor = math.cos(bearing_to_dest - wind_direction) * 0.1  # Reduced influence
            weather_adjustment = math.radians(wind_factor * 10)  # Max 10 degree adjustment
            
            optimal_bearing = bearing_to_dest + weather_adjustment
            
            # Calculate step distance based on conditions
            weather_resistance = 1 + (wind_speed / 60) ** 2 + (wave_height / 10) ** 2  # Reduced penalty
            effective_speed = base_speed / weather_resistance
            
            # Adaptive step size
            if distance_to_dest > 1000:
                step_km = max_step_km
            elif distance_to_dest > 500:
                step_km = (max_step_km + min_step_km) / 2
            else:
                step_km = max(min_step_km, distance_to_dest * 0.3)
            
            step_km = step_km / weather_resistance  # Adjust for weather
            
            # Calculate next position
            next_lat = current_lat + (step_km / 111.32) * math.cos(optimal_bearing)
            next_lon = current_lon + (step_km / (111.32 * math.cos(math.radians(current_lat)))) * math.sin(optimal_bearing)
            
            # Land avoidance check
            if self.land_mask and self.land_mask.is_on_land(next_lat, next_lon):
                next_lat, next_lon = self.land_mask.find_nearest_water(next_lat, next_lon, max_search_radius=1.0)
            
            route.append([next_lon, next_lat])
            
            # Update counters
            step_distance_actual = geodesic((current_lat, current_lon), (next_lat, next_lon)).kilometers
            total_distance += step_distance_actual
            total_time += step_distance_actual / (effective_speed * 1.852)
            
            current_lat, current_lon = next_lat, next_lon
            current_time += timedelta(hours=step_distance_actual / (effective_speed * 1.852))
        
        logger.info(f"Generated fallback route with {len(route)} waypoints, {total_distance:.1f}km total distance")
        return route, total_distance, total_time
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        return math.atan2(y, x)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points"""
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers

# Initialize services
weather_service = EnhancedWeatherService()
ml_optimizer = MLRouteOptimizer()
fallback_optimizer = FallbackRouteOptimizer(weather_service)

@app.route('/optimize_route', methods=['POST'])
def optimize_route():
    """Enhanced route optimization endpoint with land avoidance"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        logger.info(f"🚢 Route optimization request: {data.get('shipType', 'Unknown')} from {data.get('startPort', 'Unknown')} to {data.get('endPort', 'Unknown')}")
        
        # Validate input
        required_fields = ['shipType', 'startPort', 'endPort', 'departureDate']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        ship_type = data['shipType']
        start_port = data['startPort']  # [lon, lat]
        end_port = data['endPort']      # [lon, lat]
        departure_date = data['departureDate']
        use_ml = data.get('useMLRouting', True)
        
        # Validate coordinates
        if not (-90 <= start_port[1] <= 90 and -180 <= start_port[0] <= 180):
            return jsonify({"error": "Invalid start port coordinates"}), 400
        if not (-90 <= end_port[1] <= 90 and -180 <= end_port[0] <= 180):
            return jsonify({"error": "Invalid end port coordinates"}), 400
        
        # Parse departure time
        try:
            departure_time = datetime.fromisoformat(departure_date.replace('Z', '+00:00'))
        except:
            departure_time = datetime.now()
        
        route_method = "Physics-based"
        
        # Try ML-based routing first if requested and available
        if use_ml and ml_optimizer.is_initialized:
            try:
                logger.info("🤖 Using enhanced AI-based route optimization with land avoidance")
                optimized_route = ml_optimizer.generate_ml_route(
                    start_port[1], start_port[0],  # lat, lon
                    end_port[1], end_port[0],      # lat, lon
                    ship_type, departure_time
                )
                
                # Calculate total distance
                total_distance = 0
                for i in range(len(optimized_route) - 1):
                    lon1, lat1 = optimized_route[i]
                    lon2, lat2 = optimized_route[i+1]
                    total_distance += geodesic((lat1, lon1), (lat2, lon2)).kilometers
                
                # Estimate time
                ship_speeds = {"passenger ship": 22, "cargo ship": 16, "tanker": 12}
                avg_speed = ship_speeds.get(ship_type.lower(), 16)
                travel_time_hours = total_distance / (avg_speed * 1.852)
                
                route_method = "ML"
                
            except Exception as ml_error:
                logger.warning(f"⚠️  Enhanced AI routing failed: {ml_error}, using fallback method")
                use_ml = False
        
        # Use enhanced physics-based fallback if ML failed or not requested
        if not use_ml or not ml_optimizer.is_initialized:
            logger.info("⚙️  Using enhanced physics-based route optimization with land avoidance")
            
            optimized_route, total_distance, travel_time_hours = fallback_optimizer.generate_route(
                start_port[1], start_port[0],  # lat, lon
                end_port[1], end_port[0],      # lat, lon
                ship_type, departure_time
            )
        
        # Get enhanced weather information for the route
        weather_info = []
        try:
            # Sample weather at key points along the route
            sample_points = min(8, len(optimized_route))
            if sample_points > 0:
                step = max(1, len(optimized_route) // sample_points)
                
                for i in range(0, len(optimized_route), step):
                    lon, lat = optimized_route[i]
                    time_offset = i * (travel_time_hours / len(optimized_route))
                    weather = weather_service.get_weather_forecast(lat, lon, int(time_offset))
                    weather_info.append({
                        'position': [lon, lat],
                        'weather': weather
                    })
        except Exception as e:
            logger.warning(f"Failed to get enhanced weather info: {e}")
        
        # Calculate estimated savings
        direct_distance = geodesic((start_port[1], start_port[0]), (end_port[1], end_port[0])).kilometers
        estimated_savings = max(0, (direct_distance * 1.15) - total_distance)  # Assume direct route is 15% longer due to no optimization
        
        response = {
            "optimized_route": optimized_route,
            "total_distance_km": float(total_distance),
            "travel_time_hours": float(travel_time_hours),
            "weather_forecast": weather_info,
            "route_method": route_method,
            "waypoints_count": len(optimized_route),
            "estimated_savings_km": float(estimated_savings),
            "direct_distance_km": float(direct_distance),
            "route_efficiency": float((direct_distance / total_distance) * 100) if total_distance > 0 else 100,
            "land_avoidance_enabled": ml_optimizer.land_mask is not None or fallback_optimizer.land_mask is not None
        }
        
        logger.info(f"✅ Route generated successfully: {len(optimized_route)} waypoints, {total_distance:.1f}km, {route_method} method")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Route optimization error: {e}", exc_info=True)
        return jsonify({"error": f"Route optimization failed: {str(e)}"}), 500

@app.route('/weather', methods=['GET'])
def get_weather():
    """Get enhanced weather information for a specific location"""
    try:
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))
        hours_ahead = int(request.args.get('hours', 0))
        
        # Validate coordinates
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({"error": "Invalid coordinates"}), 400
        
        weather = weather_service.get_weather_forecast(lat, lon, hours_ahead)
        return jsonify(weather), 200
        
    except ValueError:
        return jsonify({"error": "Invalid coordinate format"}), 400
    except Exception as e:
        logger.error(f"Weather request error: {e}")
        return jsonify({"error": "Failed to get weather data"}), 500

@app.route('/model_status', methods=['GET'])
def model_status():
    """Get enhanced ML model initialization status"""
    return jsonify({
        "ml_model_ready": ml_optimizer.is_initialized,
        "weather_service_ready": True,
        "fallback_available": True,
        "land_avoidance_enabled": ml_optimizer.land_mask is not None,
        "training_samples": 20000 if ml_optimizer.is_initialized else 0,
        "model_version": "enhanced_v2"
    })

@app.route('/validate_coordinates', methods=['POST'])
def validate_coordinates():
    """Validate if coordinates are in navigable water"""
    try:
        data = request.json
        lat = float(data.get('lat', 0))
        lon = float(data.get('lon', 0))
        
        is_navigable = True
        message = "Coordinates are in navigable water"
        
        if ml_optimizer.land_mask:
            if ml_optimizer.land_mask.is_on_land(lat, lon):
                is_navigable = False
                message = "Coordinates are on land or in shallow water"
                # Find nearest water
                nearest_lat, nearest_lon = ml_optimizer.land_mask.find_nearest_water(lat, lon)
                return jsonify({
                    "is_navigable": is_navigable,
                    "message": message,
                    "suggested_coordinates": [nearest_lon, nearest_lat],
                    "distance_to_water_km": geodesic((lat, lon), (nearest_lat, nearest_lon)).kilometers
                })
        
        return jsonify({
            "is_navigable": is_navigable,
            "message": message
        })
        
    except Exception as e:
        logger.error(f"Coordinate validation error: {e}")
        return jsonify({"error": "Failed to validate coordinates"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "enhanced_v2.0",
        "features": {
            "ml_model": ml_optimizer.is_initialized,
            "land_avoidance": ml_optimizer.land_mask is not None,
            "weather_service": True,
            "enhanced_routing": True,
            "training_samples": 20000 if ml_optimizer.is_initialized else 0
        },
        "services": {
            "weather": True,
            "routing": True,
            "validation": True
        }
    })

if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced Ship Routing Server v2.0...")
    logger.info(f"🤖 Enhanced ML Model Available: {ShipRoutingModel is not None}")
    logger.info(f"🗺️  Land Avoidance Available: {LandMaskService is not None}")
    logger.info(f"🌦️  OpenWeather API: {'Configured' if weather_service.openweather_api_key else 'Using enhanced synthetic data'}")
    logger.info(f"📊 Training Dataset: 20,000 samples with land avoidance")
    logger.info("⏱️  Note: First-time model training will take 5-10 minutes")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)