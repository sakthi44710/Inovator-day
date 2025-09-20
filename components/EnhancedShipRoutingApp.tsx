'use client';

import { useState, useCallback, useEffect } from 'react';
import dynamic from 'next/dynamic';
import EnhancedSidebar from './EnhancedSidebar';
import SearchBar from './SearchBar';

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

// Enhanced LeafletMap props interface
interface EnhancedLeafletMapProps {
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

// Dynamically import the enhanced LeafletMap component
const EnhancedLeafletMap = dynamic<EnhancedLeafletMapProps>(() => import('./EnhancedLeafletMap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-gray-100 dark:bg-gray-900">
      <div className="text-center">
        <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p className="text-gray-600 dark:text-gray-400">Loading enhanced map...</p>
      </div>
    </div>
  )
});

// Default map view settings - centered on Arabian Sea for ship routing
const DEFAULT_CENTER: [number, number] = [20.5937, 78.9629]; // Center of India/Arabian Sea
const DEFAULT_ZOOM = 5;

export default function EnhancedShipRoutingApp() {
  const [isNavOpen, setIsNavOpen] = useState(true);
  const [selectedRoute, setSelectedRoute] = useState<[number, number][] | null>(null);
  const [weatherForecast, setWeatherForecast] = useState<WeatherInfo[]>([]);
  const [startPort, setStartPort] = useState<[number, number] | null>(null);
  const [endPort, setEndPort] = useState<[number, number] | null>(null);
  const [isSelectingLocation, setIsSelectingLocation] = useState<'start' | 'end' | null>(null);
  const [showWeather, setShowWeather] = useState(true);
  const [zoomToLocation, setZoomToLocation] = useState<[number, number] | null>(null);
  const [searchResults, setSearchResults] = useState<[number, number][]>([]);
  const [backendStatus, setBackendStatus] = useState<'unknown' | 'connected' | 'error'>('unknown');

  // Check backend connectivity on mount
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/health');
        if (response.ok) {
          setBackendStatus('connected');
          console.log('✅ Backend connected successfully');
        } else {
          setBackendStatus('error');
          console.warn('⚠️ Backend responded with error status');
        }
      } catch (error) {
        setBackendStatus('error');
        console.warn('⚠️ Backend connection failed:', error);
      }
    };

    checkBackendStatus();
    
    // Check every 30 seconds
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handles location selection from the map or search bar
  const handleLocationSelect = useCallback((location: [number, number]) => {
    if (isSelectingLocation === 'start') {
      setStartPort(location);
      console.log('Start port selected:', location);
    } else if (isSelectingLocation === 'end') {
      setEndPort(location);
      console.log('End port selected:', location);
    }
    // Zoom to the newly selected location
    setZoomToLocation(location);
  }, [isSelectingLocation]);

  // Enhanced search with geocoding
  const handleSearch = useCallback(async (query: string) => {
    try {
      console.log('Searching for:', query);
      
      // Use Nominatim API for geocoding
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query + ' port')}&limit=5`
      );
      
      if (response.ok) {
        const data = await response.json();
        const results: [number, number][] = data.map((item: any) => [
          parseFloat(item.lon),
          parseFloat(item.lat)
        ]);
        
        setSearchResults(results);
        
        if (results.length > 0) {
          // Zoom to first result
          setZoomToLocation(results[0]);
          console.log('Found locations:', results);
        }
      } else {
        console.warn('Search failed:', response.statusText);
      }
    } catch (error) {
      console.error('Search error:', error);
      
      // Fallback to some predefined major ports
      const majorPorts: { [key: string]: [number, number] } = {
        'mumbai': [72.8777, 19.0760],
        'dubai': [55.2708, 25.2048],
        'singapore': [103.8198, 1.3521],
        'hong kong': [114.1694, 22.3193],
        'shanghai': [121.4737, 31.2304],
        'rotterdam': [4.4777, 51.9225],
        'hamburg': [9.9937, 53.5511],
        'los angeles': [-118.2437, 34.0522],
        'new york': [-74.0060, 40.7128],
        'london': [-0.1276, 51.5074]
      };
      
      const queryLower = query.toLowerCase();
      const matchedPort = Object.keys(majorPorts).find(port => 
        port.includes(queryLower) || queryLower.includes(port)
      );
      
      if (matchedPort) {
        const coords = majorPorts[matchedPort];
        setSearchResults([coords]);
        setZoomToLocation(coords);
        console.log('Using fallback port:', matchedPort, coords);
      }
    }
  }, []);

  // Finalizes the location selection process
  const handleConfirmLocation = useCallback(() => {
    setIsSelectingLocation(null);
    setZoomToLocation(null);
    setSearchResults([]); // Clear search results
  }, []);

  // Handle weather forecast updates
  const handleWeatherForecastUpdate = useCallback((forecast: WeatherInfo[]) => {
    setWeatherForecast(forecast);
    console.log('Weather forecast updated:', forecast.length, 'points');
  }, []);

  // Clear route and related data
  const clearRoute = useCallback(() => {
    setSelectedRoute(null);
    setWeatherForecast([]);
    setZoomToLocation(null);
  }, []);

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      {/* Enhanced Sidebar */}
      <EnhancedSidebar
        isNavOpen={isNavOpen}
        setIsNavOpen={setIsNavOpen}
        setSelectedRoute={setSelectedRoute}
        setWeatherForecast={handleWeatherForecastUpdate}
        startPort={startPort}
        endPort={endPort}
        setIsSelectingLocation={setIsSelectingLocation}
        showWeather={showWeather}
        setShowWeather={setShowWeather}
        onClearRoute={clearRoute}
        backendStatus={backendStatus}
      />

      {/* Main Map Area */}
      <main className="flex-1 relative">
        {/* Search Bar Overlay */}
        {isSelectingLocation && (
          <div className="absolute top-0 left-0 w-full z-10 p-4 flex justify-center">
            <div className="bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm rounded-lg shadow-lg p-4">
              <div className="text-center mb-3">
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {isSelectingLocation === 'start' ? 
                    '📍 Select Start Port' : 
                    '🎯 Select Destination Port'
                  }
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Click on the map or search for a location
                </p>
              </div>
              
              <SearchBar
                onLocationSelect={handleLocationSelect}
                onSearch={handleSearch}
                onConfirmLocation={handleConfirmLocation}
              />
            </div>
          </div>
        )}

        {/* Backend Status Indicator */}
        {backendStatus === 'error' && (
          <div className="absolute top-4 right-4 z-10 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-red-700 dark:text-red-300">Backend Offline</span>
            </div>
            <p className="text-xs text-red-600 dark:text-red-400 mt-1">
              Using demo mode. Start backend for full features.
            </p>
          </div>
        )}

        {backendStatus === 'connected' && (
          <div className="absolute top-4 right-4 z-10 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-lg p-2 opacity-75">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 bg-green-500 rounded-full"></div>
              <span className="text-xs text-green-700 dark:text-green-300">AI Backend Online</span>
            </div>
          </div>
        )}

        {/* Enhanced Map Component */}
        <EnhancedLeafletMap
          route={selectedRoute}
          weatherForecast={weatherForecast}
          showWeather={showWeather}
          startPort={startPort}
          endPort={endPort}
          isSelectingLocation={isSelectingLocation}
          onLocationSelect={handleLocationSelect}
          zoomToLocation={zoomToLocation}
          searchResults={searchResults}
          defaultCenter={DEFAULT_CENTER}
          defaultZoom={DEFAULT_ZOOM}
        />
      </main>
    </div>
  );
}