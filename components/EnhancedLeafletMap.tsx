import { useEffect, useRef, useState, memo, useCallback } from 'react';
import L from 'leaflet';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, useMapEvents, LayersControl, Circle } from 'react-leaflet';
import { Button } from './ui/button';
import { Play, Pause, Square, Wind, Waves, Thermometer, Eye } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// Weather data interface
interface WeatherInfo {
  position: [number, number];
  weather: {
    wind_speed: number;
    wind_direction: number;
    wave_height: number;
    wave_period: number;
    temperature: number;
    visibility: number;
    weather_condition: string;
    humidity: number;
    pressure: number;
  };
}

// Enhanced props interface
export interface LeafletMapProps {
  route: [number, number][] | null;
  weatherForecast?: WeatherInfo[];
  showWeather: boolean;
  startPort: [number, number] | null;
  endPort: [number, number] | null;
  isSelectingLocation: 'start' | 'end' | null;
  onLocationSelect: (location: [number, number]) => void;
  zoomToLocation: [number, number] | null;
  searchResults: [number, number][];
  defaultCenter: [number, number];
  defaultZoom: number;
}

// Weather overlay component
function WeatherOverlay({ weatherData }: { weatherData: WeatherInfo[] }) {
  return (
    <>
      {weatherData.map((info, index) => {
        const [lon, lat] = info.position;
        const weather = info.weather;
        
        // Color coding based on conditions
        const getConditionColor = (condition: string) => {
          switch (condition.toLowerCase()) {
            case 'storm': return '#ff4444';
            case 'rough': return '#ff8800';
            case 'normal': return '#44ff44';
            case 'calm': return '#4444ff';
            default: return '#888888';
          }
        };
        
        const getWindIntensity = (windSpeed: number) => {
          if (windSpeed < 10) return 0.3;
          if (windSpeed < 20) return 0.6;
          if (windSpeed < 35) return 0.8;
          return 1.0;
        };
        
        return (
          <Circle
            key={index}
            center={[lat, lon]}
            radius={20000} // 20km radius
            color={getConditionColor(weather.weather_condition)}
            fillColor={getConditionColor(weather.weather_condition)}
            fillOpacity={getWindIntensity(weather.wind_speed)}
            weight={2}
          >
            <Popup>
              <div className="weather-popup">
                <h3 className="font-bold text-lg mb-2">Weather Conditions</h3>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <Wind className="h-4 w-4 mr-2" />
                    <span>Wind: {weather.wind_speed.toFixed(1)} knots @ {weather.wind_direction.toFixed(0)}°</span>
                  </div>
                  <div className="flex items-center">
                    <Waves className="h-4 w-4 mr-2" />
                    <span>Waves: {weather.wave_height.toFixed(1)}m</span>
                  </div>
                  <div className="flex items-center">
                    <Thermometer className="h-4 w-4 mr-2" />
                    <span>Temp: {weather.temperature.toFixed(1)}°C</span>
                  </div>
                  <div className="flex items-center">
                    <Eye className="h-4 w-4 mr-2" />
                    <span>Visibility: {weather.visibility.toFixed(1)}km</span>
                  </div>
                  <div className="mt-2 p-2 bg-gray-100 rounded">
                    <span className="font-medium">Condition: {weather.weather_condition}</span>
                  </div>
                </div>
              </div>
            </Popup>
          </Circle>
        );
      })}
    </>
  );
}

// Enhanced ship marker with direction indicator
function EnhancedShipMarker({ position, angle, speed = 0, weatherCondition = 'normal' }: { 
  position: [number, number]; 
  angle: number; 
  speed?: number;
  weatherCondition?: string;
}) {
  // Determine ship color based on conditions
  const getShipColor = () => {
    switch (weatherCondition.toLowerCase()) {
      case 'storm': return '#ff4444';
      case 'rough': return '#ff8800';
      case 'calm': return '#4444ff';
      default: return '#2563eb';
    }
  };
  
  const shipColor = getShipColor();
  
  const shipIcon = L.divIcon({
    className: 'enhanced-ship-icon',
    html: `
      <div style="
        transform: rotate(${(angle * 180 / Math.PI)}deg); /* Corrected: Convert radians to degrees */
        transform-origin: center;
        filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
      ">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none">
          <!-- Ship hull -->
          <path d="M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1 .6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1" 
                stroke="${shipColor}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
          <!-- Ship body -->
          <path d="M19.38 20A11.6 11.6 0 0 0 21 14l-9-4-9 4c0 2.9.94 5.34 2.81 7.76" 
                stroke="${shipColor}" stroke-width="2" fill="${shipColor}" fill-opacity="0.7"/>
          <!-- Ship cabin -->
          <path d="M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6" 
                stroke="${shipColor}" stroke-width="2" fill="${shipColor}" fill-opacity="0.9"/>
          <!-- Mast -->
          <path d="M12 10v4" stroke="${shipColor}" stroke-width="2"/>
          <path d="M12 2v3" stroke="${shipColor}" stroke-width="2"/>
          <!-- Wake effect -->
          <circle cx="12" cy="12" r="18" fill="none" stroke="${shipColor}" stroke-width="1" opacity="0.3"/>
        </svg>
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
  
  return (
    <Marker position={[position[1], position[0]]} icon={shipIcon}>
      <Popup>
        <div>
          <h3 className="font-bold">Ship Status</h3>
          <p>Speed: {speed.toFixed(1)} knots</p>
          <p>Heading: {((angle * 180 / Math.PI) % 360).toFixed(0)}°</p>
          <p>Conditions: {weatherCondition}</p>
        </div>
      </Popup>
    </Marker>
  );
}

// Animation controls component
function AnimationControls({ 
  isAnimating, 
  isPaused, 
  onStart, 
  onPause, 
  onStop, 
  onSpeedChange,
  speed = 1 
}: {
  isAnimating: boolean;
  isPaused: boolean;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
  onSpeedChange: (speed: number) => void;
  speed?: number;
}) {
  return (
    <div className="absolute bottom-4 left-4 z-[1000] bg-white/90 dark:bg-gray-800/90 p-3 rounded-lg shadow-lg backdrop-blur-sm">
      <div className="flex items-center gap-2">
        <Button
          onClick={isAnimating && !isPaused ? onPause : onStart}
          size="sm"
          className="bg-emerald-500 hover:bg-emerald-600"
        >
          {isAnimating && !isPaused ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        
        <Button
          onClick={onStop}
          size="sm"
          variant="outline"
          disabled={!isAnimating}
        >
          <Square className="h-4 w-4" />
        </Button>
        
        <div className="flex items-center gap-1">
          <span className="text-xs">Speed:</span>
          <select 
            value={speed} 
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            className="text-xs p-1 rounded border"
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={5}>5x</option>
          </select>
        </div>
      </div>
    </div>
  );
}

// Map event handler
function MapEventHandler({ 
  isSelectingLocation, 
  onLocationSelect 
}: { 
  isSelectingLocation: 'start' | 'end' | null; 
  onLocationSelect: (location: [number, number]) => void; 
}) {
  useMapEvents({
    click: (e) => {
      if (isSelectingLocation) {
        onLocationSelect([e.latlng.lng, e.latlng.lat]);
      }
    },
  });
  return null;
}

// Zoom handler
function ZoomHandler({ 
  zoomToLocation, 
  defaultCenter, 
  defaultZoom 
}: { 
  zoomToLocation: [number, number] | null; 
  defaultCenter: [number, number]; 
  defaultZoom: number; 
}) {
  const map = useMap();
  
  useEffect(() => {
    if (zoomToLocation) {
      map.flyTo([zoomToLocation[1], zoomToLocation[0]], 10, { duration: 2 });
    } else {
      map.flyTo([defaultCenter[1], defaultCenter[0]], defaultZoom, { duration: 2 });
    }
  }, [map, zoomToLocation, defaultCenter, defaultZoom]);
  
  return null;
}

// Main component
const EnhancedLeafletMap: React.FC<LeafletMapProps> = memo(({
  route,
  weatherForecast = [],
  showWeather,
  startPort,
  endPort,
  isSelectingLocation,
  onLocationSelect,
  zoomToLocation,
  searchResults,
  defaultCenter,
  defaultZoom
}) => {
  const [shipPosition, setShipPosition] = useState<[number, number] | null>(null);
  const [shipAngle, setShipAngle] = useState(0);
  const [shipSpeed, setShipSpeed] = useState(0);
  const [currentWeather, setCurrentWeather] = useState('normal');
  const [isAnimating, setIsAnimating] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [traveledPath, setTraveledPath] = useState<[number, number][]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  
  // Animation logic
  const startAnimation = useCallback(() => {
    if (route && route.length > 0) {
      setIsAnimating(true);
      setIsPaused(false);
      setShipPosition([route[0][0], route[0][1]]);
      setTraveledPath([[route[0][0], route[0][1]]]);
      setCurrentStep(0);
      setProgress(0);
      lastTimeRef.current = performance.now();
      
      const animate = (currentTime: number) => {
        if (!lastTimeRef.current) {
          lastTimeRef.current = currentTime;
        }
        
        const deltaTime = currentTime - lastTimeRef.current;
        const adjustedDelta = deltaTime * animationSpeed;
        
        setProgress(prev => {
          const newProgress = prev + adjustedDelta / 100; // Adjust speed here
          
          if (currentStep < route.length - 1 && newProgress >= 1) {
            setCurrentStep(prev => prev + 1);
            return 0;
          }
          
          return Math.min(newProgress, 1);
        });
        
        lastTimeRef.current = currentTime;
        
        if (currentStep < route.length - 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setIsAnimating(false);
        }
      };
      
      animationRef.current = requestAnimationFrame(animate);
    }
  }, [route, animationSpeed, currentStep]);
  
  const pauseAnimation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    setIsPaused(true);
  }, []);
  
  const stopAnimation = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    setIsAnimating(false);
    setIsPaused(false);
    setShipPosition(null);
    setTraveledPath([]);
    setCurrentStep(0);
    setProgress(0);
  }, []);
  
  // Update ship position based on animation progress
  useEffect(() => {
    if (route && route.length > 1 && currentStep < route.length - 1) {
      const [startLon, startLat] = route[currentStep];
      const [endLon, endLat] = route[currentStep + 1];
      
      const t = progress;
      const newLon = startLon + (endLon - startLon) * t;
      const newLat = startLat + (endLat - startLat) * t;
      
      setShipPosition([newLon, newLat]);
      setTraveledPath(prev => [...prev, [newLon, newLat]]);
      
      // Calculate ship angle
      const dx = endLon - startLon;
      const dy = endLat - startLat;
      setShipAngle(Math.atan2(dy, dx));
      
      // Calculate ship speed (simplified)
      const distance = Math.sqrt(dx * dx + dy * dy) * 111.32; // km
      setShipSpeed(distance * animationSpeed * 10); // Approximate speed in knots
      
      // Get weather conditions for current position
      if (weatherForecast.length > 0) {
        const nearestWeather = weatherForecast.reduce((closest, weather) => {
          const dist1 = Math.sqrt(
            (weather.position[1] - newLat) ** 2 + (weather.position[0] - newLon) ** 2
          );
          const dist2 = Math.sqrt(
            (closest.position[1] - newLat) ** 2 + (closest.position[0] - newLon) ** 2
          );
          return dist1 < dist2 ? weather : closest;
        });
        setCurrentWeather(nearestWeather.weather.weather_condition);
      }
    }
  }, [route, currentStep, progress, animationSpeed, weatherForecast]);
  
  // Resume animation after pause
  useEffect(() => {
    if (isPaused && !isAnimating) {
      // Animation is paused, don't restart
      return;
    }
    
    if (isAnimating && !isPaused && !animationRef.current) {
      // Resume animation
      lastTimeRef.current = performance.now();
      
      const animate = (currentTime: number) => {
        if (!lastTimeRef.current) {
          lastTimeRef.current = currentTime;
        }
        
        const deltaTime = currentTime - lastTimeRef.current;
        const adjustedDelta = deltaTime * animationSpeed;
        
        setProgress(prev => {
          const newProgress = prev + adjustedDelta / 100;
          
          if (currentStep < (route?.length || 0) - 1 && newProgress >= 1) {
            setCurrentStep(prev => prev + 1);
            return 0;
          }
          
          return Math.min(newProgress, 1);
        });
        
        lastTimeRef.current = currentTime;
        
        if (currentStep < (route?.length || 0) - 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setIsAnimating(false);
        }
      };
      
      animationRef.current = requestAnimationFrame(animate);
    }
  }, [isPaused, isAnimating, animationSpeed, currentStep, route?.length]);
  
  // Clean up animation on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  // Create custom icons
  const createCustomIcon = (svgString: string, color: string = 'black') => L.divIcon({
    className: 'custom-icon',
    html: `<div style="color: ${color}; filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.5));">${svgString}</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 28],
  });
  
  const startIcon = createCustomIcon(
    '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="3"/><line x1="12" y1="22" x2="12" y2="8"/><path d="M5 12H2a10 10 0 0 0 20 0h-3"/></svg>',
    '#10b981'
  );
  
  const endIcon = createCustomIcon(
    '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg>',
    '#ef4444'
  );
  
  const searchIcon = createCustomIcon(
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    '#3b82f6'
  );
  
  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={[defaultCenter[1], defaultCenter[0]]}
        zoom={defaultZoom}
        style={{ height: '100%', width: '100%' }}
        className="z-0"
        // Corrected: Removed unused ref that can cause initialization errors
      >
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="OpenStreetMap">
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
          </LayersControl.BaseLayer>
          
          <LayersControl.BaseLayer name="Satellite">
            <TileLayer
              url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://opentopomap.org">OpenTopoMap</a> contributors'
            />
          </LayersControl.BaseLayer>
          
          {showWeather && weatherForecast.length > 0 && (
            <LayersControl.Overlay checked name="Weather Conditions">
              <WeatherOverlay weatherData={weatherForecast} />
            </LayersControl.Overlay>
          )}
          
          {route && (
            <LayersControl.Overlay checked name="Planned Route">
              <Polyline 
                positions={route.map(([lon, lat]) => [lat, lon])} 
                color="#3b82f6" 
                weight={3} 
                opacity={0.7}
                dashArray="10, 5"
              />
            </LayersControl.Overlay>
          )}
          
          {traveledPath.length > 1 && (
            <LayersControl.Overlay checked name="Traveled Path">
              <Polyline 
                positions={traveledPath.map(([lon, lat]) => [lat, lon])} 
                color="#10B981" 
                weight={4}
                opacity={0.9}
              />
            </LayersControl.Overlay>
          )}
        </LayersControl>
        
        {/* Port markers */}
        {startPort && (
          <Marker position={[startPort[1], startPort[0]]} icon={startIcon}>
            <Popup>
              <div className="font-medium">
                <h3 className="font-bold text-green-600">Start Port</h3>
                <p>Coordinates: {startPort[1].toFixed(4)}, {startPort[0].toFixed(4)}</p>
              </div>
            </Popup>
          </Marker>
        )}
        
        {endPort && (
          <Marker position={[endPort[1], endPort[0]]} icon={endIcon}>
            <Popup>
              <div className="font-medium">
                <h3 className="font-bold text-red-600">Destination Port</h3>
                <p>Coordinates: {endPort[1].toFixed(4)}, {endPort[0].toFixed(4)}</p>
              </div>
            </Popup>
          </Marker>
        )}
        
        {/* Ship marker */}
        {shipPosition && (
          <EnhancedShipMarker 
            position={shipPosition} 
            angle={shipAngle}
            speed={shipSpeed}
            weatherCondition={currentWeather}
          />
        )}
        
        {/* Search result markers */}
        {searchResults.map((result, index) => (
          <Marker 
            key={index} 
            position={[result[1], result[0]]} 
            icon={searchIcon}
          >
            <Popup>Search Result {index + 1}</Popup>
          </Marker>
        ))}
        
        <MapEventHandler 
          isSelectingLocation={isSelectingLocation} 
          onLocationSelect={onLocationSelect} 
        />
        
        <ZoomHandler 
          zoomToLocation={zoomToLocation} 
          defaultCenter={defaultCenter} 
          defaultZoom={defaultZoom} 
        />
      </MapContainer>
      
      {/* Animation Controls */}
      {route && route.length > 1 && (
        <AnimationControls
          isAnimating={isAnimating}
          isPaused={isPaused}
          onStart={startAnimation}
          onPause={pauseAnimation}
          onStop={stopAnimation}
          onSpeedChange={setAnimationSpeed}
          speed={animationSpeed}
        />
      )}
      
      {/* Route Progress Indicator */}
      {isAnimating && route && (
        <div className="absolute top-4 right-4 z-[1000] bg-white/90 dark:bg-gray-800/90 p-4 rounded-lg shadow-lg backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="flex flex-col">
              <span className="text-sm font-medium">Route Progress</span>
              <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-emerald-500 transition-all duration-100 ease-out"
                  style={{ 
                    width: `${((currentStep + progress) / Math.max(1, route.length - 1)) * 100}%` 
                  }}
                />
              </div>
              <span className="text-xs text-gray-500 mt-1">
                {currentStep + 1} / {route.length} waypoints
              </span>
            </div>
            
            <div className="flex flex-col items-center">
              <span className="text-xs text-gray-500">Speed</span>
              <span className="text-sm font-medium">{shipSpeed.toFixed(1)} kts</span>
            </div>
            
            <div className="flex flex-col items-center">
              <span className="text-xs text-gray-500">Conditions</span>
              <span className={`text-sm font-medium capitalize ${
                currentWeather === 'storm' ? 'text-red-500' :
                currentWeather === 'rough' ? 'text-orange-500' :
                currentWeather === 'calm' ? 'text-blue-500' :
                'text-green-500'
              }`}>
                {currentWeather}
              </span>
            </div>
          </div>
        </div>
      )}
      
      {/* Weather Legend */}
      {showWeather && weatherForecast.length > 0 && (
        <div className="absolute bottom-4 right-4 z-[1000] bg-white/90 dark:bg-gray-800/90 p-3 rounded-lg shadow-lg backdrop-blur-sm">
          <h4 className="font-medium text-sm mb-2">Weather Legend</h4>
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-blue-500 opacity-60"></div>
              <span>Calm (Low intensity)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-500 opacity-60"></div>
              <span>Normal</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-orange-500 opacity-80"></div>
              <span>Rough</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <span>Storm (High intensity)</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

EnhancedLeafletMap.displayName = 'EnhancedLeafletMap';
export default EnhancedLeafletMap;
