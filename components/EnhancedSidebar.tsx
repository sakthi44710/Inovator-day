'use client';

import { motion } from 'framer-motion'
import { ChevronLeft, ChevronRight, Waves, Wifi, WifiOff, AlertCircle } from 'lucide-react'
import { Switch } from './ui/switch'
import { Label } from './ui/label'
import { Button } from './ui/button'
import EnhancedRouteForm from './EnhancedRouteForm'

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

interface EnhancedSidebarProps {
  isNavOpen: boolean
  setIsNavOpen: (isOpen: boolean) => void
  setSelectedRoute: (route: [number, number][]) => void
  setWeatherForecast?: (forecast: WeatherInfo[]) => void
  startPort: [number, number] | null
  endPort: [number, number] | null
  setIsSelectingLocation: (type: 'start' | 'end' | null) => void
  showWeather: boolean
  setShowWeather: (show: boolean) => void
  onClearRoute?: () => void
  backendStatus: 'unknown' | 'connected' | 'error'
}

export default function EnhancedSidebar({ 
  isNavOpen, 
  setIsNavOpen, 
  setSelectedRoute,
  setWeatherForecast,
  startPort, 
  endPort, 
  setIsSelectingLocation,
  showWeather,
  setShowWeather,
  onClearRoute,
  backendStatus
}: EnhancedSidebarProps) {
  
  const getBackendStatusIcon = () => {
    switch (backendStatus) {
      case 'connected':
        return <Wifi className="h-4 w-4 text-green-500" />
      case 'error':
        return <WifiOff className="h-4 w-4 text-red-500" />
      default:
        return <AlertCircle className="h-4 w-4 text-yellow-500 animate-pulse" />
    }
  }

  const getBackendStatusText = () => {
    switch (backendStatus) {
      case 'connected':
        return 'AI Backend Online'
      case 'error':
        return 'Backend Offline'
      default:
        return 'Connecting...'
    }
  }

  return (
    <motion.div 
      className={`bg-white dark:bg-gray-800 shadow-lg overflow-hidden transition-all duration-300 ease-in-out relative flex flex-col`}
      initial={false}
      animate={{ width: isNavOpen ? '28rem' : '5rem' }}
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <motion.div
          animate={{ opacity: isNavOpen ? 1 : 0, transition: { delay: 0.1 } }}
          className="whitespace-nowrap overflow-hidden"
        >
          <h1 className="text-2xl font-bold mb-2 text-emerald-600 dark:text-emerald-400">
            AI Ship Router
          </h1>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Weather-optimized maritime routing
          </p>
        </motion.div>

        {/* Backend Status */}
        {isNavOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1, transition: { delay: 0.2 } }}
            className="mt-3 flex items-center gap-2 text-xs"
          >
            {getBackendStatusIcon()}
            <span className={`font-medium ${
              backendStatus === 'connected' ? 'text-green-600 dark:text-green-400' :
              backendStatus === 'error' ? 'text-red-600 dark:text-red-400' :
              'text-yellow-600 dark:text-yellow-400'
            }`}>
              {getBackendStatusText()}
            </span>
          </motion.div>
        )}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-6">
          {isNavOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, transition: { delay: 0.15 } }}
            >
              {/* Weather Controls */}
              <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Waves className="h-5 w-5 text-blue-500" />
                    <Label htmlFor="weather-toggle" className="font-medium text-gray-700 dark:text-gray-300">
                      Weather Overlay
                    </Label>
                  </div>
                  <Switch
                    id="weather-toggle"
                    checked={showWeather}
                    onCheckedChange={setShowWeather}
                  />
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Display real-time weather conditions and forecasts on the map
                </p>
              </div>

              {/* Route Form */}
              <EnhancedRouteForm
                setSelectedRoute={setSelectedRoute}
                setWeatherForecast={setWeatherForecast}
                isNavOpen={isNavOpen}
                startPort={startPort}
                endPort={endPort}
                setIsSelectingLocation={setIsSelectingLocation}
              />
            </motion.div>
          )}
        </div>
      </div>

      {/* Footer with Clear Route Button and Toggle */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700 space-y-3">
        {isNavOpen && onClearRoute && (startPort || endPort) && (
          <Button
            onClick={onClearRoute}
            variant="outline"
            className="w-full text-sm"
          >
            Clear Route & Markers
          </Button>
        )}
        
        <div className="flex justify-end">
          <button
            onClick={() => setIsNavOpen(!isNavOpen)}
            className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 p-2 rounded-full shadow-md z-10 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            title={isNavOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {isNavOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
          </button>
        </div>
      </div>
    </motion.div>
  )
}