import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Ship, Anchor, Navigation, Calendar } from 'lucide-react'
import { Button } from './ui/button'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { DateTimePicker } from "./ui/date-time-picker"
import { format } from 'date-fns'

interface RouteFormProps {
  setSelectedRoute: (route: [number, number][]) => void
  isNavOpen: boolean
  startPort: [number, number] | null
  endPort: [number, number] | null
  setIsSelectingLocation: (type: 'start' | 'end' | null) => void
}

export default function RouteForm({ setSelectedRoute, isNavOpen, startPort, endPort, setIsSelectingLocation }: RouteFormProps) {
  const [shipType, setShipType] = useState('')
  const [departureDate, setDepartureDate] = useState<Date | undefined>(new Date(2024, 7, 25))
  const [isLoading, setIsLoading] = useState(false)
  const [currentError, setCurrentError] = useState<string | null>(null)

  const isFormValid = () => {
    return (
      startPort &&
      endPort &&
      departureDate &&
      shipType !== ''
    );
  };

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
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`);
      }

      console.log('Received data:', data);
      setSelectedRoute(data.optimized_route);
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

  return (
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
            <SelectItem value="Passenger ship">Passenger Ship</SelectItem>
            <SelectItem value="Cargo ship">Cargo Ship</SelectItem>
            <SelectItem value="Tanker">Tanker</SelectItem>
          </SelectContent>
        </Select>
      </motion.div>

      <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
        <Label htmlFor="startPort" className="text-lg font-semibold text-gray-700 dark:text-gray-300 flex items-center">
          <Anchor className="mr-2" /> Start Port
        </Label>
        <Button type="button" onClick={() => setIsSelectingLocation('start')} className="mt-2 w-full">
          {startPort ? `Selected: ${startPort[1].toFixed(2)}, ${startPort[0].toFixed(2)}` : 'Select on Map'}
        </Button>
      </motion.div>

      <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
        <Label htmlFor="endPort" className="text-lg font-semibold text-gray-700 dark:text-gray-300 flex items-center">
          <Navigation className="mr-2" /> End Port
        </Label>
        <Button type="button" onClick={() => setIsSelectingLocation('end')} className="mt-2 w-full">
          {endPort ? `Selected: ${endPort[1].toFixed(2)}, ${endPort[0].toFixed(2)}` : 'Select on Map'}
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

      <motion.div animate={{ opacity: isNavOpen ? 1 : 0 }}>
        <Button
          type="submit"
          className="w-full bg-emerald-500 hover:bg-emerald-600 text-white"
          disabled={!isFormValid() || isLoading}
        >
          {isLoading ? 'Optimizing...' : 'Calculate Optimal Route'}
        </Button>
      </motion.div>

      {currentError && (
        <p className="text-red-500 text-sm mt-2">{currentError}</p>
      )}
    </form>
  )
}
