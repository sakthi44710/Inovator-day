import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten, Input, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import requests
import json
from datetime import datetime, timedelta
import pickle
import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings
from geopy.distance import geodesic
import random
warnings.filterwarnings('ignore')

@dataclass
class WeatherData:
    """Weather data structure"""
    lat: float
    lon: float
    wind_speed: float
    wind_direction: float
    wave_height: float
    wave_period: float
    wave_direction: float
    temperature: float
    visibility: float
    timestamp: datetime

@dataclass
class RoutePoint:
    """Route point with weather conditions"""
    lat: float
    lon: float
    weather: WeatherData
    fuel_cost: float
    time_cost: float
    risk_factor: float

class LandMaskService:
    """Service to detect land vs water areas"""
    
    def __init__(self):
        # Create a more comprehensive land mask for India and surrounding regions
        self.land_polygons = self._create_land_polygons()
    
    def _create_land_polygons(self):
        """Create simplified land polygons for major landmasses"""
        # Major landmasses in the Indian Ocean region
        land_areas = {
            'india': {
                'bounds': [(6.5, 68.0), (37.5, 97.5)],  # [lat_min, lon_min], [lat_max, lon_max]
                'coastal_buffer': 0.2  # degrees buffer from coast
            },
            'sri_lanka': {
                'bounds': [(5.9, 79.5), (9.9, 81.9)],
                'coastal_buffer': 0.1
            },
            'myanmar': {
                'bounds': [(10.0, 92.0), (28.5, 101.0)],
                'coastal_buffer': 0.2
            },
            'thailand': {
                'bounds': [(5.6, 97.3), (20.5, 105.6)],
                'coastal_buffer': 0.15
            },
            'malaysia': {
                'bounds': [(1.0, 99.6), (7.4, 119.3)],
                'coastal_buffer': 0.1
            },
            'bangladesh': {
                'bounds': [(20.7, 88.0), (26.6, 92.7)],
                'coastal_buffer': 0.15
            }
        }
        return land_areas
    
    def is_on_land(self, lat: float, lon: float) -> bool:
        """Check if coordinates are on land (including coastal buffer)"""
        for region, data in self.land_polygons.items():
            bounds = data['bounds']
            buffer = data['coastal_buffer']
            
            lat_min, lon_min = bounds[0]
            lat_max, lon_max = bounds[1]
            
            # Check if point is within land bounds (with buffer)
            if (lat_min - buffer <= lat <= lat_max + buffer and 
                lon_min - buffer <= lon <= lon_max + buffer):
                
                # Additional refinement for India's coastline
                if region == 'india':
                    if self._is_in_detailed_india_land(lat, lon):
                        return True
                else:
                    return True
        
        # Check if depth is too shallow (simulated)
        if self._is_shallow_water(lat, lon):
            return True
            
        return False
    
    def _is_in_detailed_india_land(self, lat: float, lon: float) -> bool:
        """More detailed check for Indian landmass"""
        # Simplified polygon check for major Indian regions
        
        # Western coast exclusions (Arabian Sea should be navigable)
        if lon < 72.0 and lat > 8.0 and lat < 25.0:
            return False
            
        # Eastern coast exclusions (Bay of Bengal should be navigable)  
        if lon > 82.0 and lat > 8.0 and lat < 22.0:
            return False
            
        # Southern tip exclusions
        if lat < 10.0 and lon > 76.0 and lon < 81.0:
            return False
        
        # If within India's rough boundaries, consider it land
        if (8.0 <= lat <= 35.0 and 68.0 <= lon <= 97.0):
            # Exclude major water bodies
            # Gulf of Kutch
            if (22.0 <= lat <= 23.5 and 68.5 <= lon <= 70.5):
                return False
            # Gulf of Mannar
            if (8.5 <= lat <= 9.5 and 78.0 <= lon <= 79.5):
                return False
            return True
            
        return False
    
    def _is_shallow_water(self, lat: float, lon: float) -> bool:
        """Check for shallow water areas that should be avoided"""
        # Simulate shallow areas near coasts
        for region, data in self.land_polygons.items():
            bounds = data['bounds']
            lat_min, lon_min = bounds[0]
            lat_max, lon_max = bounds[1]
            
            # Check proximity to coastlines
            coastal_distance = min(
                abs(lat - lat_min), abs(lat - lat_max),
                abs(lon - lon_min), abs(lon - lon_max)
            )
            
            # If very close to coast, consider shallow
            if coastal_distance < 0.05:  # ~5km
                return True
                
        return False
    
    def find_nearest_water(self, lat: float, lon: float, max_search_radius: float = 1.0) -> Tuple[float, float]:
        """Find nearest navigable water from a land point"""
        if not self.is_on_land(lat, lon):
            return lat, lon
        
        # Search in expanding circles
        for radius in np.arange(0.1, max_search_radius, 0.1):
            for angle in np.arange(0, 360, 15):  # Check every 15 degrees
                angle_rad = math.radians(angle)
                test_lat = lat + radius * math.cos(angle_rad)
                test_lon = lon + radius * math.sin(angle_rad)
                
                if not self.is_on_land(test_lat, test_lon):
                    return test_lat, test_lon
        
        # If no water found nearby, return original point (will be handled as error)
        return lat, lon

class WeatherDataProvider:
    """Provides weather data from multiple sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.synthetic_data_enabled = True
        self.land_mask = LandMaskService()
        
    def get_weather_data(self, lat: float, lon: float, timestamp: datetime = None) -> WeatherData:
        """Get weather data for a specific location and time"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Try to get real weather data first
        if self.api_key:
            try:
                return self._get_real_weather_data(lat, lon, timestamp)
            except Exception as e:
                print(f"Failed to get real weather data: {e}")
        
        # Fall back to synthetic data
        return self._generate_synthetic_weather_data(lat, lon, timestamp)
    
    def _get_real_weather_data(self, lat: float, lon: float, timestamp: datetime) -> WeatherData:
        """Get real weather data from OpenWeatherMap API"""
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code != 200:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
        
        return WeatherData(
            lat=lat,
            lon=lon,
            wind_speed=data.get('wind', {}).get('speed', 0) * 1.94384,
            wind_direction=data.get('wind', {}).get('deg', 0),
            wave_height=self._estimate_wave_height(data.get('wind', {}).get('speed', 0)),
            wave_period=np.random.uniform(4, 12),
            wave_direction=data.get('wind', {}).get('deg', 0) + np.random.uniform(-30, 30),
            temperature=data.get('main', {}).get('temp', 20),
            visibility=data.get('visibility', 10000) / 1000,
            timestamp=timestamp
        )
    
    def _estimate_wave_height(self, wind_speed_ms: float) -> float:
        """Estimate wave height from wind speed using Beaufort scale approximation"""
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
    
    def _generate_synthetic_weather_data(self, lat: float, lon: float, timestamp: datetime) -> WeatherData:
        """Generate realistic synthetic weather data"""
        # Season factor
        day_of_year = timestamp.timetuple().tm_yday
        season_factor = math.sin(2 * math.pi * day_of_year / 365)
        
        # Latitude factor
        lat_factor = abs(lat) / 90.0
        
        # Monsoon effects for Indian Ocean
        monsoon_factor = 0
        if 5 <= lat <= 25 and 65 <= lon <= 95:  # Indian Ocean region
            # Southwest monsoon (June-September)
            if 150 <= day_of_year <= 270:
                monsoon_factor = 1.5
            # Northeast monsoon (October-December)  
            elif day_of_year >= 275 or day_of_year <= 31:
                monsoon_factor = 1.2
        
        # Base weather conditions
        base_wind = 10 + lat_factor * 15 + monsoon_factor * 8
        base_wave = 1.5 + lat_factor * 2.5 + monsoon_factor * 1.5
        
        # Add realistic variability
        wind_speed = max(0, base_wind + np.random.normal(0, 5) + season_factor * 8)
        wave_height = max(0.1, base_wave + np.random.normal(0, 1) + season_factor * 1.5)
        
        return WeatherData(
            lat=lat,
            lon=lon,
            wind_speed=wind_speed,
            wind_direction=np.random.uniform(0, 360),
            wave_height=wave_height,
            wave_period=np.random.uniform(4, 14),
            wave_direction=np.random.uniform(0, 360),
            temperature=20 + season_factor * 15 - lat_factor * 20 + monsoon_factor * 5,
            visibility=max(1, min(20, 15 - monsoon_factor * 5 + np.random.normal(0, 3))),
            timestamp=timestamp
        )

class ShipRoutingModel:
    """Enhanced neural network model for ship routing optimization"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.land_mask = LandMaskService()
        
    def create_model(self, input_shape: Tuple[int, ...]) -> Model:
        """Create enhanced neural network architecture"""
        # Weather input (enhanced features)
        weather_input = Input(shape=(10,), name='weather_input')
        weather_dense = Dense(64, activation='relu')(weather_input)
        weather_dense = BatchNormalization()(weather_dense)
        weather_dense = Dropout(0.3)(weather_dense)
        weather_dense = Dense(32, activation='relu')(weather_dense)
        
        # Route context input (last 10 waypoints for better context)
        route_input = Input(shape=(10, 2), name='route_input')
        route_lstm = LSTM(64, return_sequences=True)(route_input)
        route_lstm = LSTM(32, return_sequences=True)(route_lstm)
        route_lstm = LSTM(16)(route_lstm)
        
        # Ship parameters input
        ship_input = Input(shape=(5,), name='ship_input')
        ship_dense = Dense(32, activation='relu')(ship_input)
        ship_dense = Dense(16, activation='relu')(ship_dense)
        
        # Environmental context (lat, lon, depth approximation)
        env_input = Input(shape=(3,), name='env_input')
        env_dense = Dense(16, activation='relu')(env_input)
        
        # Combine all inputs
        combined = concatenate([weather_dense, route_lstm, ship_dense, env_dense])
        
        # Enhanced decision network
        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer: [next_lat, next_lon, fuel_cost, time_cost, risk_score]
        output = Dense(5, activation='linear', name='route_output')(x)
        
        model = Model(
            inputs=[weather_input, route_input, ship_input, env_input], 
            outputs=output
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss='mse', 
            metrics=['mae']
        )
        
        return model
    
    def prepare_training_data(self, weather_provider: WeatherDataProvider, 
                            num_samples: int = 20000) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate enhanced training data with 20,000+ samples"""
        print(f"Generating {num_samples} training samples with land avoidance...")
        
        weather_data = []
        route_data = []
        ship_data = []
        env_data = []
        targets = []
        
        # Define major shipping routes in Indian Ocean
        major_routes = [
            # Mumbai to Dubai
            {"start": (19.0760, 72.8777), "end": (25.2048, 55.2708)},
            # Chennai to Singapore
            {"start": (13.0827, 80.2707), "end": (1.3521, 103.8198)},
            # Kolkata to Hong Kong
            {"start": (22.5726, 88.3639), "end": (22.3193, 114.1694)},
            # Kochi to Port Louis
            {"start": (9.9312, 76.2673), "end": (-20.1609, 57.5012)},
            # Visakhapatnam to Yangon
            {"start": (17.6868, 83.2185), "end": (16.8661, 96.1951)},
            # Mumbai to Colombo
            {"start": (19.0760, 72.8777), "end": (6.9271, 79.8612)},
        ]
        
        valid_samples = 0
        attempts = 0
        max_attempts = num_samples * 3  # Allow for failed attempts
        
        while valid_samples < num_samples and attempts < max_attempts:
            attempts += 1
            
            if attempts % 1000 == 0:
                print(f"Generated {valid_samples} valid samples (attempts: {attempts})...")
            
            # Use major routes 70% of the time, random routes 30%
            if np.random.random() < 0.7 and major_routes:
                route = np.random.choice(major_routes)
                start_lat, start_lon = route["start"]
                end_lat, end_lon = route["end"]
                
                # Add some variation to major routes
                start_lat += np.random.normal(0, 0.5)
                start_lon += np.random.normal(0, 0.5)
                end_lat += np.random.normal(0, 0.5)
                end_lon += np.random.normal(0, 0.5)
            else:
                # Generate random oceanic routes
                start_lat = np.random.uniform(-30, 30)
                start_lon = np.random.uniform(50, 120)
                end_lat = np.random.uniform(-30, 30)
                end_lon = np.random.uniform(50, 120)
            
            # Ensure start and end points are in water
            if weather_provider.land_mask.is_on_land(start_lat, start_lon):
                start_lat, start_lon = weather_provider.land_mask.find_nearest_water(start_lat, start_lon)
            
            if weather_provider.land_mask.is_on_land(end_lat, end_lon):
                end_lat, end_lon = weather_provider.land_mask.find_nearest_water(end_lat, end_lon)
            
            # Generate route context (last 10 waypoints)
            route_context = []
            current_lat, current_lon = start_lat, start_lon
            
            # Create a path towards destination
            steps = 10
            for j in range(steps):
                # Move towards destination with some variation
                progress = j / steps
                target_lat = start_lat + (end_lat - start_lat) * progress
                target_lon = start_lon + (end_lon - start_lon) * progress
                
                # Add navigation variation
                dlat = (target_lat - current_lat) * 0.3 + np.random.uniform(-0.2, 0.2)
                dlon = (target_lon - current_lon) * 0.3 + np.random.uniform(-0.2, 0.2)
                
                current_lat += dlat
                current_lon += dlon
                
                # Ensure we stay in water
                if weather_provider.land_mask.is_on_land(current_lat, current_lon):
                    current_lat, current_lon = weather_provider.land_mask.find_nearest_water(
                        current_lat, current_lon, max_search_radius=0.5
                    )
                
                route_context.append([current_lat, current_lon])
            
            # Get weather data for current position
            try:
                weather = weather_provider.get_weather_data(current_lat, current_lon)
            except:
                continue  # Skip invalid weather data
            
            # Enhanced weather features
            weather_features = [
                weather.wind_speed,
                weather.wind_direction / 360.0,  # Normalize
                weather.wave_height,
                weather.wave_period,
                weather.wave_direction / 360.0,  # Normalize
                weather.temperature,
                weather.visibility,
                math.sin(2 * math.pi * weather.timestamp.timetuple().tm_yday / 365),  # Season
                math.cos(2 * math.pi * weather.timestamp.timetuple().tm_hour / 24),  # Time of day
                abs(weather.lat) / 90.0  # Latitude factor
            ]
            
            # Enhanced ship parameters
            ship_type = np.random.randint(0, 3)  # 0: passenger, 1: cargo, 2: tanker
            ship_size = np.random.uniform(0.5, 2.0)
            ship_speed = np.random.uniform(8, 25)
            cargo_weight = np.random.uniform(0, 1)
            ship_age = np.random.uniform(0, 1)  # Normalized age factor
            
            ship_features = [ship_type / 2.0, ship_size, ship_speed / 25.0, cargo_weight, ship_age]
            
            # Environmental features
            env_features = [
                current_lat / 90.0,  # Normalized latitude
                current_lon / 180.0,  # Normalized longitude
                self._estimate_depth(current_lat, current_lon)  # Estimated normalized depth
            ]
            
            # Calculate optimal next waypoint with enhanced physics
            try:
                next_lat, next_lon, fuel_cost, time_cost, risk = self._calculate_optimal_waypoint_enhanced(
                    current_lat, current_lon, end_lat, end_lon, weather, ship_features, weather_provider.land_mask
                )
                
                # Ensure next waypoint is in water
                if weather_provider.land_mask.is_on_land(next_lat, next_lon):
                    next_lat, next_lon = weather_provider.land_mask.find_nearest_water(next_lat, next_lon)
                
                # Validate the waypoint
                if self._is_valid_waypoint(current_lat, current_lon, next_lat, next_lon):
                    weather_data.append(weather_features)
                    route_data.append(route_context)
                    ship_data.append(ship_features)
                    env_data.append(env_features)
                    targets.append([next_lat / 90.0, next_lon / 180.0, fuel_cost, time_cost, risk])
                    valid_samples += 1
                    
            except Exception as e:
                continue  # Skip invalid calculations
        
        print(f"Training data generation complete! Generated {valid_samples} valid samples from {attempts} attempts.")
        
        # Convert to numpy arrays
        X = [
            np.array(weather_data), 
            np.array(route_data), 
            np.array(ship_data),
            np.array(env_data)
        ]
        y = np.array(targets)
        
        return X, y
    
    def _estimate_depth(self, lat: float, lon: float) -> float:
        """Estimate normalized ocean depth (0-1, where 1 is deepest)"""
        # Simplified depth estimation based on distance from coast
        # In real implementation, use bathymetry data
        
        # Deeper water in the middle of ocean basins
        if self.land_mask.is_on_land(lat, lon):
            return 0.0  # On land
        
        # Estimate based on distance from major landmasses
        distances = []
        for region, data in self.land_mask.land_polygons.items():
            bounds = data['bounds']
            lat_center = (bounds[0][0] + bounds[1][0]) / 2
            lon_center = (bounds[0][1] + bounds[1][1]) / 2
            distance = geodesic((lat, lon), (lat_center, lon_center)).kilometers
            distances.append(distance)
        
        min_distance = min(distances)
        
        # Normalize depth (deeper = further from land)
        if min_distance < 50:  # Very close to coast
            return 0.1
        elif min_distance < 200:  # Continental shelf
            return 0.3
        elif min_distance < 500:  # Continental slope
            return 0.6
        else:  # Deep ocean
            return 1.0
    
    def _is_valid_waypoint(self, current_lat: float, current_lon: float, 
                          next_lat: float, next_lon: float) -> bool:
        """Validate that a waypoint is reasonable"""
        # Check distance (shouldn't be too far in one step)
        distance = geodesic((current_lat, current_lon), (next_lat, next_lon)).kilometers
        if distance > 500:  # Max 500km per step
            return False
        
        # Check coordinates are within reasonable bounds
        if not (-90 <= next_lat <= 90 and -180 <= next_lon <= 180):
            return False
            
        return True
    
    def _calculate_optimal_waypoint_enhanced(self, lat: float, lon: float, end_lat: float, end_lon: float,
                                           weather: WeatherData, ship_features: List[float], 
                                           land_mask: LandMaskService) -> Tuple[float, float, float, float, float]:
        """Calculate optimal next waypoint with enhanced physics and land avoidance"""
        ship_type_idx, ship_size, ship_speed, cargo_weight, ship_age = ship_features
        
        # Calculate bearing to destination
        dest_bearing = math.atan2(end_lon - lon, end_lat - lat)
        
        # Weather resistance factors
        wind_resistance = 1 + (weather.wind_speed / 40) ** 2
        wave_resistance = 1 + (weather.wave_height / 8) ** 2
        
        # Ship-specific factors
        ship_efficiency = [0.9, 0.85, 0.8][int(ship_type_idx * 2)]  # passenger, cargo, tanker
        age_factor = 1 + (ship_age * 0.2)  # Older ships less efficient
        
        # Weather optimal direction
        wind_angle = math.radians(weather.wind_direction)
        preferred_direction = wind_angle + math.pi  # Go with the wind when possible
        
        # Balance between destination direction and weather optimal direction
        weather_weight = min(weather.wind_speed / 30, 0.4)  # More weight in high winds
        optimal_bearing = (1 - weather_weight) * dest_bearing + weather_weight * preferred_direction
        
        # Add some controlled randomness for exploration
        bearing_variation = np.random.normal(0, math.pi / 12)  # ±15 degrees standard deviation
        actual_bearing = optimal_bearing + bearing_variation
        
        # Calculate step size based on conditions and distance to destination
        distance_to_dest = geodesic((lat, lon), (end_lat, end_lon)).kilometers
        base_step_km = min(100, distance_to_dest * 0.2)  # Adaptive step size
        
        # Adjust step size for weather
        weather_factor = 1 / (wind_resistance * wave_resistance)
        step_km = base_step_km * weather_factor
        
        # Convert to degrees (approximate)
        step_lat = (step_km / 111.32) * math.cos(actual_bearing)
        step_lon = (step_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(actual_bearing)
        
        # Calculate next position
        next_lat = lat + step_lat
        next_lon = lon + step_lon
        
        # Land avoidance - if next position is on land, try alternative directions
        attempts = 0
        while land_mask.is_on_land(next_lat, next_lon) and attempts < 8:
            # Try different bearings around the optimal bearing
            alternative_bearing = optimal_bearing + (attempts - 4) * math.pi / 6  # ±30° steps
            
            step_lat = (step_km / 111.32) * math.cos(alternative_bearing)
            step_lon = (step_km / (111.32 * math.cos(math.radians(lat)))) * math.sin(alternative_bearing)
            
            next_lat = lat + step_lat
            next_lon = lon + step_lon
            attempts += 1
        
        # If still on land, find nearest water
        if land_mask.is_on_land(next_lat, next_lon):
            next_lat, next_lon = land_mask.find_nearest_water(next_lat, next_lon)
        
        # Calculate costs with enhanced models
        fuel_cost = (wind_resistance * wave_resistance * age_factor * (1 + cargo_weight * 0.3)) / ship_efficiency
        time_cost = fuel_cost / (ship_speed * ship_size)
        
        # Enhanced risk calculation
        weather_risk = min(1.0, (weather.wave_height / 6 + weather.wind_speed / 50) / 2)
        proximity_risk = 0.5 if attempts > 0 else 0.0  # Risk if we had to avoid land
        visibility_risk = max(0, (10 - weather.visibility) / 10)
        
        risk_factor = (weather_risk + proximity_risk + visibility_risk) / 3
        
        return next_lat, next_lon, fuel_cost, time_cost, risk_factor
    
    def train(self, weather_provider: WeatherDataProvider, num_samples: int = 20000):
        """Train the enhanced routing model"""
        print(f"Training model with {num_samples} samples...")
        
        # Generate training data
        X, y = self.prepare_training_data(weather_provider, num_samples)
        
        # Create and train model
        self.model = self.create_model((10,))
        
        # Normalize the data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        # Fit scaler on weather data only
        X[0] = self.scaler.fit_transform(X[0])
        
        print("Starting enhanced model training...")
        history = self.model.fit(
            X, y,
            batch_size=64,
            epochs=100,  # More epochs for better learning
            validation_split=0.2,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
            ]
        )
        
        self.is_trained = True
        print("Enhanced model training complete!")
        
        return history
    
    def predict_next_waypoint(self, weather_data: WeatherData, route_context: List[Tuple[float, float]], 
                            ship_features: List[float], current_pos: Tuple[float, float]) -> RoutePoint:
        """Predict the next optimal waypoint"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare enhanced input data
        weather_features = np.array([[
            weather_data.wind_speed,
            weather_data.wind_direction / 360.0,
            weather_data.wave_height,
            weather_data.wave_period,
            weather_data.wave_direction / 360.0,
            weather_data.temperature,
            weather_data.visibility,
            math.sin(2 * math.pi * weather_data.timestamp.timetuple().tm_yday / 365),
            math.cos(2 * math.pi * weather_data.timestamp.timetuple().tm_hour / 24),
            abs(weather_data.lat) / 90.0
        ]])
        
        # Normalize weather features
        weather_features = self.scaler.transform(weather_features)
        
        # Prepare route context (last 10 waypoints)
        if len(route_context) < 10:
            # Pad with current position if not enough history
            current_pos_list = list(current_pos) if current_pos else [weather_data.lat, weather_data.lon]
            while len(route_context) < 10:
                route_context.insert(0, current_pos_list)
        
        route_features = np.array([route_context[-10:]])
        
        # Prepare ship features (normalized)
        ship_features_norm = np.array([[
            ship_features[0] / 2.0,  # Ship type
            ship_features[1],        # Size
            ship_features[2] / 25.0, # Speed normalized
            ship_features[3],        # Cargo weight
            ship_features[4] if len(ship_features) > 4 else 0.5  # Age factor
        ]])
        
        # Environmental features
        current_lat, current_lon = current_pos if current_pos else (weather_data.lat, weather_data.lon)
        env_features = np.array([[
            current_lat / 90.0,
            current_lon / 180.0,
            self._estimate_depth(current_lat, current_lon)
        ]])
        
        # Make prediction
        prediction = self.model.predict([weather_features, route_features, ship_features_norm, env_features])
        
        # Denormalize coordinates
        next_lat = prediction[0][0] * 90.0
        next_lon = prediction[0][1] * 180.0
        fuel_cost = float(prediction[0][2])
        time_cost = float(prediction[0][3])
        risk_factor = float(prediction[0][4])
        
        # Ensure the predicted point is in water
        if self.land_mask.is_on_land(next_lat, next_lon):
            next_lat, next_lon = self.land_mask.find_nearest_water(next_lat, next_lon)
        
        return RoutePoint(
            lat=float(next_lat),
            lon=float(next_lon),
            weather=weather_data,
            fuel_cost=fuel_cost,
            time_cost=time_cost,
            risk_factor=risk_factor
        )
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model:
            self.model.save(f"{filepath}_model.h5")
            with open(f"{filepath}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Enhanced model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
            with open(f"{filepath}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            print(f"Enhanced model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load model: {e}")

class OptimalRouteGenerator:
    """Generate optimal routes using the enhanced trained model"""
    
    def __init__(self, model: ShipRoutingModel, weather_provider: WeatherDataProvider):
        self.model = model
        self.weather_provider = weather_provider
        self.land_mask = LandMaskService()
    
    def generate_route(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float,
                      ship_type: str, departure_time: datetime = None) -> List[RoutePoint]:
        """Generate optimal route with enhanced land avoidance and waypoint optimization"""
        if departure_time is None:
            departure_time = datetime.now()
        
        # Ensure start and end points are in water
        if self.land_mask.is_on_land(start_lat, start_lon):
            start_lat, start_lon = self.land_mask.find_nearest_water(start_lat, start_lon)
            print(f"Moved start point to water: {start_lat:.4f}, {start_lon:.4f}")
        
        if self.land_mask.is_on_land(end_lat, end_lon):
            end_lat, end_lon = self.land_mask.find_nearest_water(end_lat, end_lon)
            print(f"Moved end point to water: {end_lat:.4f}, {end_lon:.4f}")
        
        # Ship parameters based on type
        ship_params = self._get_ship_parameters(ship_type)
        
        route = []
        current_lat, current_lon = start_lat, start_lon
        current_time = departure_time
        
        # Route context for ML model
        route_context = [(current_lat, current_lon)]
        
        max_iterations = 200  # Reduced for more direct routes
        iteration = 0
        min_step_distance = 50  # Minimum step distance in km
        
        print(f"Generating route from ({start_lat:.3f}, {start_lon:.3f}) to ({end_lat:.3f}, {end_lon:.3f})")
        
        while iteration < max_iterations:
            # Check if we're close to destination
            distance_to_dest = geodesic((current_lat, current_lon), (end_lat, end_lon)).kilometers
            
            if distance_to_dest < 50:  # Within 50 km
                # Add final waypoint
                weather = self.weather_provider.get_weather_data(end_lat, end_lon, current_time)
                final_point = RoutePoint(
                    lat=end_lat,
                    lon=end_lon,
                    weather=weather,
                    fuel_cost=0.1,
                    time_cost=0.1,
                    risk_factor=0.1
                )
                route.append(final_point)
                print(f"Route completed in {iteration} steps, total distance: {self._calculate_total_distance(route):.1f} km")
                break
            
            # Get weather data for current position
            weather = self.weather_provider.get_weather_data(current_lat, current_lon, current_time)
            
            # Predict next waypoint
            try:
                next_point = self.model.predict_next_waypoint(
                    weather, route_context, ship_params, (current_lat, current_lon)
                )
                
                # Ensure minimum step distance and reasonable direction
                step_distance = geodesic((current_lat, current_lon), (next_point.lat, next_point.lon)).kilometers
                
                if step_distance < min_step_distance:
                    # If step is too small, extend it towards destination
                    direction_to_dest = math.atan2(end_lat - current_lat, end_lon - current_lon)
                    step_km = max(min_step_distance, distance_to_dest * 0.3)
                    
                    # Calculate new position
                    next_lat = current_lat + (step_km / 111.32) * math.cos(direction_to_dest)
                    next_lon = current_lon + (step_km / (111.32 * math.cos(math.radians(current_lat)))) * math.sin(direction_to_dest)
                    
                    # Ensure it's in water
                    if self.land_mask.is_on_land(next_lat, next_lon):
                        next_lat, next_lon = self.land_mask.find_nearest_water(next_lat, next_lon)
                    
                    next_point.lat = next_lat
                    next_point.lon = next_lon
                
                # Adjust direction more aggressively towards destination
                next_point = self._adjust_towards_destination(
                    next_point, current_lat, current_lon, end_lat, end_lon, distance_to_dest
                )
                
                route.append(next_point)
                route_context.append((next_point.lat, next_point.lon))
                
                # Update position and time
                current_lat, current_lon = next_point.lat, next_point.lon
                current_time += timedelta(hours=next_point.time_cost)
                
                iteration += 1
                
                if iteration % 10 == 0:
                    remaining_distance = geodesic((current_lat, current_lon), (end_lat, end_lon)).kilometers
                    print(f"Step {iteration}: Current position ({current_lat:.3f}, {current_lon:.3f}), {remaining_distance:.1f} km to destination")
                
            except Exception as e:
                print(f"Error generating waypoint at iteration {iteration}: {e}")
                break
        
        if iteration >= max_iterations:
            print("Warning: Maximum iterations reached. Route may be incomplete.")
        
        # Optimize route by removing unnecessary waypoints
        optimized_route = self._optimize_waypoints(route)
        print(f"Route optimized: {len(route)} -> {len(optimized_route)} waypoints")
        
        return optimized_route
    
    def _optimize_waypoints(self, route: List[RoutePoint]) -> List[RoutePoint]:
        """Remove unnecessary waypoints to create a more direct route"""
        if len(route) <= 2:
            return route
        
        optimized = [route[0]]  # Always keep start point
        
        i = 0
        while i < len(route) - 1:
            current = optimized[-1]
            
            # Look ahead to find the furthest point we can reach directly
            max_direct_index = i + 1
            
            for j in range(i + 2, min(i + 10, len(route))):  # Look up to 10 steps ahead
                target = route[j]
                
                # Check if we can go directly from current to target without hitting land
                if self._is_direct_path_clear(current.lat, current.lon, target.lat, target.lon):
                    max_direct_index = j
                else:
                    break
            
            # Add the furthest reachable point
            if max_direct_index < len(route):
                optimized.append(route[max_direct_index])
                i = max_direct_index
            else:
                break
        
        # Always add the final destination
        if len(optimized) == 0 or optimized[-1] != route[-1]:
            optimized.append(route[-1])
        
        return optimized
    
    def _is_direct_path_clear(self, lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """Check if there's a clear water path between two points"""
        # Sample points along the path
        num_samples = max(3, int(geodesic((lat1, lon1), (lat2, lon2)).kilometers / 50))  # Sample every ~50km
        
        for i in range(1, num_samples):
            t = i / num_samples
            sample_lat = lat1 + (lat2 - lat1) * t
            sample_lon = lon1 + (lon2 - lon1) * t
            
            if self.land_mask.is_on_land(sample_lat, sample_lon):
                return False
        
        return True
    
    def _calculate_total_distance(self, route: List[RoutePoint]) -> float:
        """Calculate total distance of the route"""
        if len(route) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(route) - 1):
            distance = geodesic((route[i].lat, route[i].lon), (route[i+1].lat, route[i+1].lon)).kilometers
            total += distance
        
        return total
    
    def _get_ship_parameters(self, ship_type: str) -> List[float]:
        """Get enhanced ship parameters based on type"""
        ship_params = {
            "passenger ship": [0, 1.0, 20, 0.3, 0.3],  # type, size, speed, cargo, age
            "cargo ship": [1, 1.5, 15, 0.8, 0.5],
            "tanker": [2, 2.0, 12, 1.0, 0.6]
        }
        
        return ship_params.get(ship_type.lower(), [1, 1.0, 15, 0.5, 0.5])
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km"""
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    
    def _adjust_towards_destination(self, predicted_point: RoutePoint, 
                                  current_lat: float, current_lon: float,
                                  dest_lat: float, dest_lon: float, distance_to_dest: float) -> RoutePoint:
        """Adjust predicted point to move more directly towards destination"""
        # Calculate bearing to destination
        dest_bearing = math.atan2(dest_lon - current_lon, dest_lat - current_lat)
        
        # Calculate bearing of predicted point
        pred_bearing = math.atan2(predicted_point.lon - current_lon, predicted_point.lat - current_lat)
        
        # Increase destination bias as we get closer
        if distance_to_dest < 500:  # Within 500km
            dest_weight = 0.8
        elif distance_to_dest < 1000:  # Within 1000km
            dest_weight = 0.6
        else:
            dest_weight = 0.4
        
        # Blend the bearings
        blended_bearing = dest_weight * dest_bearing + (1 - dest_weight) * pred_bearing
        
        # Calculate distance for this step
        step_distance = geodesic((current_lat, current_lon), (predicted_point.lat, predicted_point.lon)).kilometers
        
        # Ensure reasonable step size
        step_distance = min(step_distance, distance_to_dest * 0.4)  # Don't overshoot
        step_distance = max(step_distance, 50)  # Minimum step size
        
        # Calculate new position
        new_lat = current_lat + (step_distance / 111.32) * math.cos(blended_bearing)
        new_lon = current_lon + (step_distance / (111.32 * math.cos(math.radians(current_lat)))) * math.sin(blended_bearing)
        
        # Ensure new position is in water
        if self.land_mask.is_on_land(new_lat, new_lon):
            new_lat, new_lon = self.land_mask.find_nearest_water(new_lat, new_lon)
        
        # Update the predicted point
        predicted_point.lat = new_lat
        predicted_point.lon = new_lon
        
        return predicted_point

# Example usage and training
if __name__ == "__main__":
    print("Initializing Enhanced Ship Routing ML System with Land Avoidance...")
    
    # Initialize components
    weather_provider = WeatherDataProvider()
    model = ShipRoutingModel()
    
    # Train the model with 20,000 samples
    print("Training the enhanced routing model with 20,000 samples...")
    model.train(weather_provider, num_samples=20000)
    
    # Save the model
    model.save_model("enhanced_ship_routing_model")
    
    # Test route generation
    route_generator = OptimalRouteGenerator(model, weather_provider)
    
    # Generate a test route (Mumbai to Dubai) - should avoid land
    print("\nGenerating test route from Mumbai to Dubai (avoiding land)...")
    route = route_generator.generate_route(
        start_lat=19.0760, start_lon=72.8777,  # Mumbai
        end_lat=25.2048, end_lon=55.2708,      # Dubai
        ship_type="cargo ship"
    )
    
    print(f"\nGenerated optimized route with {len(route)} waypoints:")
    for i, point in enumerate(route[:10]):  # Show first 10 waypoints
        print(f"Waypoint {i+1}: ({point.lat:.4f}, {point.lon:.4f}), "
              f"Risk: {point.risk_factor:.3f}, Fuel: {point.fuel_cost:.3f}")
    
    # Calculate and display total distance
    total_distance = route_generator._calculate_total_distance(route)
    print(f"\nTotal route distance: {total_distance:.1f} km")
    print(f"Average waypoint separation: {total_distance/max(1, len(route)-1):.1f} km")
    
    print("\nEnhanced Ship Routing ML System ready with land avoidance!")