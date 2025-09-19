import { useState } from 'react'
import { motion } from 'framer-motion'
import { Search } from 'lucide-react'
import { Input } from './ui/input'
import { Button } from './ui/button'

interface SearchBarProps {
  onLocationSelect: (location: [number, number]) => void
  onSearch: (query: string) => void
  onConfirmLocation: () => void
}

interface NominatimResult {
  lat: string;
  lon: string;
  display_name: string;
}

export default function SearchBar({ onLocationSelect, onSearch, onConfirmLocation }: SearchBarProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [suggestions, setSuggestions] = useState<NominatimResult[]>([])
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsLoading(true);
    try {
        // Using Nominatim API for geocoding
        const response = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}`);
        const data: NominatimResult[] = await response.json();
        setSuggestions(data);
        // notify parent of the search query so the onSearch prop is used
        onSearch(searchQuery);
    } catch (error) {
      console.error('Error searching for location:', error);
      setSuggestions([]);
    } finally {
        setIsLoading(false);
    }
  }

  const handleSelect = (lon: number, lat: number) => {
    onLocationSelect([lon, lat]);
    setSearchQuery('');
    setSuggestions([]);
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  }

  return (
    <div className="w-full max-w-md">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative"
      >
        <div className="flex items-center bg-white dark:bg-gray-800 rounded-md shadow-lg">
          <Input
            type="text"
            placeholder="Search for a port or city..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            className="w-full border-0 focus:ring-0"
          />
          <Button onClick={handleSearch} className="m-1" disabled={isLoading}>
            {isLoading ? '...' : <Search className="h-4 w-4" />}
          </Button>
        </div>
        {suggestions.length > 0 && (
          <ul className="absolute w-full bg-white dark:bg-gray-800 mt-1 rounded-md shadow-lg max-h-60 overflow-auto z-20">
            {suggestions.map((suggestion, index) => (
              <li
                key={index}
                className="px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer text-sm"
                onClick={() => handleSelect(parseFloat(suggestion.lon), parseFloat(suggestion.lat))}
              >
                {suggestion.display_name}
              </li>
            ))}
          </ul>
        )}
      </motion.div>
      <div className="flex justify-center mt-2">
          <Button 
            onClick={onConfirmLocation} 
            className="bg-emerald-500 hover:bg-emerald-600 text-white"
          >
            Confirm Location
          </Button>
      </div>
    </div>
  )
}
    