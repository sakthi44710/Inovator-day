'use client';

import { useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import Sidebar from './Sidebar';
import SearchBar from './SearchBar';
import type { LeafletMapProps } from './LeafletMap';

// Dynamically import the LeafletMap component to prevent SSR issues
const LeafletMap = dynamic<LeafletMapProps>(() => import('./LeafletMap'), {
  ssr: false,
  loading: () => <p className="flex h-full w-full items-center justify-center">Loading map...</p>
});

// Default map view settings
const DEFAULT_CENTER: [number, number] = [20.5937, 78.9629]; // Center of India
const DEFAULT_ZOOM = 5;

export default function ShipRoutingApp() {
  const [isNavOpen, setIsNavOpen] = useState(true);
  const [selectedRoute, setSelectedRoute] = useState<[number, number][] | null>(null);
  const [startPort, setStartPort] = useState<[number, number] | null>(null);
  const [endPort, setEndPort] = useState<[number, number] | null>(null);
  const [isSelectingLocation, setIsSelectingLocation] = useState<'start' | 'end' | null>(null);
  const [showWeather] = useState(false); // State for weather overlay
  const [zoomToLocation, setZoomToLocation] = useState<[number, number] | null>(null);

  // Handles location selection from the map or search bar
  const handleLocationSelect = useCallback((location: [number, number]) => {
    if (isSelectingLocation === 'start') {
      setStartPort(location);
    } else if (isSelectingLocation === 'end') {
      setEndPort(location);
    }
    // Zoom to the newly selected location
    setZoomToLocation(location);
  }, [isSelectingLocation]);

  // Handles simulated search results
  const handleSearch = useCallback(async (query: string) => {
    // In a real app, you would fetch from an API here.
    // This is a simulated result for demonstration.
    const results: { name: string; coords: [number, number] }[] = [
      { name: 'India (approx)', coords: [78.9629, 20.5937] },
      { name: 'Delhi', coords: [77.2090, 28.6139] },
      { name: 'Mumbai', coords: [72.8777, 19.0760] },
    ];

    // Use the query to attempt a simple name match; fall back to the first result.
    const q = (query || '').toLowerCase().trim();
    const match = results.find(r => q && r.name.toLowerCase().includes(q));
    const selected = match ? match.coords : results[0].coords;

    // For debugging (and to avoid unused variable lint errors)
    console.debug('Search query:', query, '-> selected:', selected);

    setZoomToLocation(selected);
  }, []);

  // Finalizes the location selection process
  const handleConfirmLocation = useCallback(() => {
    setIsSelectingLocation(null);
    setZoomToLocation(null); // Reset zoom to default view
  }, []);

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      <Sidebar
        isNavOpen={isNavOpen}
        setIsNavOpen={setIsNavOpen}
        setSelectedRoute={setSelectedRoute}
        startPort={startPort}
        endPort={endPort}
        setIsSelectingLocation={setIsSelectingLocation}
      />
      <main className="flex-1 relative">
        {isSelectingLocation && (
          <div className="absolute top-0 left-0 w-full z-10 p-4 flex justify-center">
            <SearchBar
              onLocationSelect={handleLocationSelect}
              onSearch={handleSearch}
              onConfirmLocation={handleConfirmLocation}
            />
          </div>
        )}
        <LeafletMap
          route={selectedRoute}
          showWeather={showWeather}
          startPort={startPort}
          endPort={endPort}
          isSelectingLocation={isSelectingLocation}
          onLocationSelect={handleLocationSelect}
          zoomToLocation={zoomToLocation}
          defaultCenter={DEFAULT_CENTER}
          defaultZoom={DEFAULT_ZOOM}
        />
      </main>
    </div>
  );
}
