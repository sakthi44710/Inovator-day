import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Ship, 
  Anchor, 
  Navigation, 
  Calendar,
  Brain,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  Route,
  Waves,
  Wind
} from 'lucide-react'
import { Button } from './ui/button'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { DateTimePicker } from "./ui/date-time-picker"
import { Switch } from "./ui/switch"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card"
import { Badge } from "./ui/badge"
import { format } from 'date-fns'

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

interface RouteOptimizationResult {
  optimized_route: [number, number][];
  total_distance_km: number;
  travel_time_hours: number;
  weather_forecast?: WeatherInfo[];
  route_method: 'ML' | 'Physics-based';
  waypoints_count: number;
}

interface ModelStatus {
  ml_model_ready: boolean;
  weather_service_ready: boolean;
  fallback_available: boolean;
}

interface RouteFormProps {
  setSelectedRoute: (route: [number, number][]) => void
  setWeatherForecast?: (forecast: WeatherInfo[]) => void
  isNavOpen: boolean
  startPort: [number, number] | null
  endPort: [number, number] | null
  setIsSelectingLocation: (type: 'start' | 'end' | null) => void
}

export default function EnhancedRouteForm({ 
  setSelectedRoute, 
  setWeatherForecast,
  isNavOpen, 
  startPort, 
  endPort, 
  setIsSelectingLocation 
}: RouteFormProps) {
  const [shipType, setShipType] = useState('')
  const [departureDate, setDepartureDate] = useState<Date | undefined>(new Date(2024, 7, 25))
  const [isLoading, setIsLoading] = useState(false)
  const [currentError, setCurrentError] = useState<string | null>(null)
  const [useMLRouting, setUseMLRouting] = useState(true)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [routeResults, setRouteResults] = useState<RouteOptimizationResult | null>(null)
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false)

  // Check model status on component mount
  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        const response = await fetch('http://localhost:5000/model_status')
        if (response.ok) {
          const status = await response.json()
          setModelStatus(status)
        }
      } catch (error) {
        console.warn('Could not check model status:', error)
      }
    }

    checkModelStatus()
    // Check status every 30 seconds
    const interval = setInterval(checkModelStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  const isFormValid = () => {
    return (
      startPort &&
      endPort &&
      departureDate &&
      shipType !== ''
    );
  };

  const getShipTypeInfo = (type: string) => {
    const shipInfo = {
      'passenger ship': {
        speed: '20-25 knots',
        characteristics: 'High speed, weather sensitive',
        icon: '🚢',
        description: 'Optimized for passenger comfort and schedule adherence'
      },
      'cargo ship': {
        speed: '15-18 knots', 
        characteristics: 'Balanced efficiency',
        icon: '🚛',
        description: 'Balanced between fuel efficiency and delivery time'
      },
      'tanker': {
        speed: '12-15 knots',
        characteristics: 'Fuel efficient, weather resilient', 
        icon: '🛢️',
        description: 'Optimized for fuel efficiency and cargo safety'
      }
    }
    
    return shipInfo[type.toLowerCase() as keyof typeof shipInfo]
  }

  const calculateEstimatedSavings = () => {
    if (!routeResults) return null
    
    // Rough estimates for comparison
    const directDistance = startPort && endPort ? 
      Math.sqrt(
        Math.pow(endPort[1] - startPort[1], 2) + 
        Math.pow(endPort[0] - startPort[0], 2)
      ) * 111.32 : 0
    
    const routeDistance = routeResults.total_distance_km
    const distanceSaved = Math.max(0, directDistance * 1.2 - routeDistance) // Assume 20% longer for direct shipping route
    const timeSaved = distanceSaved / 15 * 1.852 // Rough time calculation
    const fuelSaved = distanceSaved * 0.1 // Rough fuel calculation (tons)
    
    return {
      distance: distanceSaved,
      time: timeSaved,
      fuel: fuelSaved,
      cost: fuelSaved * 500 // Rough cost per ton of fuel
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setCurrentError(null);
    if (!isFormValid()) {
      setCurrentError("Please fill in all fields before calculating the route.");
      return
    }

    setIsLoading(true)
    try {
      console.log('Sending request with data:', {
        shipType,
        startPort,
        endPort,
        departureDate: departureDate ? format(departureDate, 'yyyy-MM-dd') : undefined,
      });

      const response = await fetch('http://localhost:5000/optimize_route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          shipType,
          startPort,
          endPort,
          departureDate: departureDate ? format(departureDate, 'yyyy-MM-dd') : undefined,
          useMLRouting,
        }),
      });

      const data: RouteOptimizationResult = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`);
      }

      console.log('Received data:', data);
      
      setSelectedRoute(data.optimized_route);
      setRouteResults(data);
      
      // Set weather forecast if available and callback provided
      if (data.weather_forecast && setWeatherForecast) {
        setWeatherForecast(data.weather_forecast);
      }
      
    } catch (error: unknown) {
      console.error('Error optimizing route:', error);
      const message =
        typeof error === 'string'
          ? error
          : error instanceof Error
            ? error.message
            : 'Unknown error';
      setCurrentError(`Error optimizing route: ${message}. Please try again.`);
    } finally {
      setIsLoading(false)
    }
  }

  const resetForm = () => {
    setRouteResults(null)
    setSelectedRoute([])
    if (setWeatherForecast) {
      setWeatherForecast([])
    }
  }

  const savings = calculateEstimatedSavings()

  return (
    <div className="space-y-6">
      {/* Model Status Banner */}
      {modelStatus && (
        <Card className={`border-l-4 ${
          modelStatus.ml_model_ready ? 'border-l-green-500 bg-green-50 dark:bg-green-950' : 
          'border-l-yellow-500 bg-yellow-50 dark:bg-yellow-950'
        }`}>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              {modelStatus.ml_model_ready ? (
                <CheckCircle className="h-4 w-4 text-green-600" />
              ) : (
                <AlertTriangle className="h-4 w-4 text-yellow-600" />
              )}
              <span className="text-sm font-medium">
                {modelStatus.ml_model_ready ? 'AI Model Ready' : 'AI Model Loading...'}
              </span>
              {modelStatus.ml_model_ready && (
                <Badge variant="secondary" className="ml-auto">
                  <Brain className="h-3 w-3 mr-1" />
                  ML-Powered
                </Badge>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Route Results Summary */}
      {routeResults && (
        <Card className="bg-gradient-to-r from-blue-50 to-emerald-50 dark:from-blue-950 dark:to-emerald-950">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <Route className="h-5 w-5" />
                Route Generated
              </CardTitle>
              <Badge variant={routeResults.route_method === 'ML' ? 'default' : 'secondary'}>
                {routeResults.route_method === 'ML' ? (
                  <><Brain className="h-3 w-3 mr-1" /> AI Optimized</>
                ) : (
                  <><Zap className="h-3 w-3 mr-1" /> Physics-Based</>
                )}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex items-center gap-2">
                <Navigation className="h-4 w-4 text-blue-500" />
                <span>{routeResults.total_distance_km.toFixed(1)} km</span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-green-500" />
                <span>{(routeResults.travel_time_hours).toFixed(1)} hours</span>
              </div>
              <div className="flex items-center gap-2">
                <Route className="h-4 w-4 text-purple-500" />
                <span>{routeResults.waypoints_count} waypoints</span>
              </div>
              <div className="flex items-center gap-2">
                <Waves className="h-4 w-4 text-teal-500" />
                <span>{routeResults.weather_forecast?.length || 0} weather points</span>
              </div>
            </div>
            
            {savings && savings.distance > 0 && (
              <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">Estimated Savings vs. Direct Route:</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <span>Distance: {savings.distance.toFixed(0)} km</span>
                  <span>Time: {savings.time.toFixed(1)} hours</span>
                  <span>Fuel: {savings.fuel.toFixed(1)} tons</span>
                  <span>Cost: ${savings.cost.toFixed(0)}</span>
                </div>
              </div>
            )}
            
            <Button 
              onClick={resetForm}
              variant="outline" 
              size="sm" 
              className="w-full"
            >
              Plan New Route
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Route Form */}
      {!routeResults && (
        <form onSubmit={handleSubmit} className="space-y-6">
          <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
            <Label htmlFor="shipType" className="text-lg font-semibold text-gray-700 dark:text-gray-300 flex items-center">
              <Ship className="mr-2" /> Ship Type
            </Label>
            <Select onValueChange={setShipType} value={shipType}>
              <SelectTrigger id="shipType" className="w-full mt-1">
                <SelectValue placeholder="Select Ship Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Passenger ship">🚢 Passenger Ship</SelectItem>
                <SelectItem value="Cargo ship">🚛 Cargo Ship</SelectItem>
                <SelectItem value="Tanker">🛢️ Tanker</SelectItem>
              </SelectContent>
            </Select>
            
            {shipType && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="mt-2 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
              >
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {getShipTypeInfo(shipType)?.description}
                </p>
                <div className="flex gap-4 mt-2 text-xs text-gray-500">
                  <span>Speed: {getShipTypeInfo(shipType)?.speed}</span>
                  <span>{getShipTypeInfo(shipType)?.characteristics}</span>
                </div>
              </motion.div>
            )}
          </motion.div>

          <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
            <Label htmlFor="startPort" className="text-lg font-semibold text-gray-700 dark:text-gray-300 flex items-center">
              <Anchor className="mr-2" /> Start Port
            </Label>
            <Button 
              type="button" 
              onClick={() => setIsSelectingLocation('start')} 
              className="mt-2 w-full"
              variant={startPort ? "default" : "outline"}
            >
              {startPort ? 
                `📍 Selected: ${startPort[1].toFixed(2)}, ${startPort[0].toFixed(2)}` : 
                'Select Start Port on Map'
              }
            </Button>
          </motion.div>

          <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
            <Label htmlFor="endPort" className="text-lg font-semibold text-gray-700 dark:text-gray-300 flex items-center">
              <Navigation className="mr-2" /> Destination Port
            </Label>
            <Button 
              type="button" 
              onClick={() => setIsSelectingLocation('end')} 
              className="mt-2 w-full"
              variant={endPort ? "default" : "outline"}
            >
              {endPort ? 
                `🎯 Selected: ${endPort[1].toFixed(2)}, ${endPort[0].toFixed(2)}` : 
                'Select Destination on Map'
              }
            </Button>
          </motion.div>

          <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
            <Label htmlFor="departureDate" className="text-lg font-semibold text-gray-700 dark:text-gray-300 flex items-center">
              <Calendar className="mr-2" /> Departure Date
            </Label>
            <DateTimePicker
              date={departureDate}
              setDate={(newDate) => setDepartureDate(newDate)}
            />
          </motion.div>

          {/* Advanced Options */}
          <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
            <Button
              type="button"
              variant="ghost"
              onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              className="w-full justify-start p-0 h-auto font-normal"
            >
              <span className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                Advanced Options {showAdvancedOptions ? '▼' : '▶'}
              </span>
            </Button>
            
            {showAdvancedOptions && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="mt-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg space-y-4"
              >
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label className="text-sm font-medium">AI-Powered Routing</Label>
                    <p className="text-xs text-gray-500">
                      Use machine learning for weather-optimized routes
                    </p>
                  </div>
                  <Switch
                    checked={useMLRouting}
                    onCheckedChange={setUseMLRouting}
                    disabled={!modelStatus?.ml_model_ready}
                  />
                </div>
                
                {!modelStatus?.ml_model_ready && (
                  <p className="text-xs text-yellow-600 dark:text-yellow-400">
                    AI model is still loading. Physics-based routing will be used.
                  </p>
                )}
              </motion.div>
            )}
          </motion.div>

          <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
            <Button
              type="submit"
              className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-medium py-3"
              disabled={!isFormValid() || isLoading}
            >
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                  <span>
                    {useMLRouting && modelStatus?.ml_model_ready ? 
                      'AI Optimizing Route...' : 
                      'Calculating Route...'
                    }
                  </span>
                </div>
              ) : (
                <div className="flex items-center justify-center gap-2">
                  {useMLRouting && modelStatus?.ml_model_ready ? (
                    <><Brain className="h-4 w-4" /> Generate AI-Optimized Route</>
                  ) : (
                    <><Route className="h-4 w-4" /> Calculate Optimal Route</>
                  )}
                </div>
              )}
            </Button>
          </motion.div>

          {currentError && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg"
            >
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-red-500" />
                <p className="text-red-700 dark:text-red-300 text-sm">{currentError}</p>
              </div>
            </motion.div>
          )}

          {/* Routing Method Info */}
          <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
            <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950">
              <CardContent className="pt-4">
                <div className="flex items-start gap-3">
                  {useMLRouting && modelStatus?.ml_model_ready ? (
                    <Brain className="h-5 w-5 text-blue-500 mt-0.5" />
                  ) : (
                    <Zap className="h-5 w-5 text-purple-500 mt-0.5" />
                  )}
                  <div>
                    <h4 className="font-medium text-sm">
                      {useMLRouting && modelStatus?.ml_model_ready ? 
                        'AI-Powered Routing' : 
                        'Physics-Based Routing'
                      }
                    </h4>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      {useMLRouting && modelStatus?.ml_model_ready ? 
                        'Uses machine learning trained on weather patterns, ship characteristics, and historical data to optimize routes for fuel efficiency, safety, and time.' :
                        'Uses real-time weather data and physics-based calculations to find optimal routes considering wind, waves, and ship characteristics.'
                      }
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </form>
      )}
    </div>
  )
}