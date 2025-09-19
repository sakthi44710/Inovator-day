import { useEffect, useRef, useState, memo } from 'react';
import L from 'leaflet';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, useMapEvents } from 'react-leaflet';
import { Button } from './ui/button';
import 'leaflet/dist/leaflet.css';

// Define the props interface
export interface LeafletMapProps {
  route: [number, number][] | null;
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

// MapEventHandler to handle map clicks
function MapEventHandler({ isSelectingLocation, onLocationSelect }: { isSelectingLocation: 'start' | 'end' | null, onLocationSelect: (location: [number, number]) => void }) {
  useMapEvents({
    click: (e) => {
      if (isSelectingLocation) {
        onLocationSelect([e.latlng.lng, e.latlng.lat]);
      }
    },
  });
  return null;
}

// ZoomHandler to manage map view changes
function ZoomHandler({ zoomToLocation, defaultCenter, defaultZoom }: { zoomToLocation: [number, number] | null, defaultCenter: [number, number], defaultZoom: number }) {
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

// ShipMarker for displaying the animated ship
function ShipMarker({ position, angle }: { position: [number, number], angle: number }) {
  const shipIcon = L.divIcon({
    className: 'custom-icon',
    html: `<div style="transform: rotate(${angle}deg); transform-origin: center;"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1 .6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M19.38 20A11.6 11.6 0 0 0 21 14l--9-4-9 4c0 2.9.94 5.34 2.81 7.76"/><path d="M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6"/><path d="M12 10v4"/><path d="M12 2v3"/></svg></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
  return <Marker position={[position[1], position[0]]} icon={shipIcon} />;
}

// Main LeafletMap component, wrapped in React.memo
const LeafletMap: React.FC<LeafletMapProps> = memo(({
  route,
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
  const mapRef = useRef<L.Map | null>(null);
  const [shipPosition, setShipPosition] = useState<[number, number] | null>(null);
  const [shipAngle, setShipAngle] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [traveledPath, setTraveledPath] = useState<[number, number][]>([]);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    // This effect ensures that we don't try to re-render the map unnecessarily
    // The map instance is stable and should not be re-created
  }, []);

  const startAnimation = () => {
    if (route && route.length > 0) {
      setIsAnimating(true);
      setShipPosition(route[0]);
      setTraveledPath([route[0]]);
      let step = 0;
      let progress = 0;
      const animateShip = (timestamp: number) => {
        if (!animationRef.current) {
          animationRef.current = timestamp;
        }
        const elapsed = timestamp - animationRef.current;
        progress += elapsed / 200; // Adjust speed here

        if (step < route.length - 1) {
          const [startLon, startLat] = route[step];
          const [endLon, endLat] = route[step + 1];
          const t = Math.min(progress, 1);

          const newLon = startLon + (endLon - startLon) * t;
          const newLat = startLat + (endLat - startLat) * t;
          setShipPosition([newLon, newLat]);
          setTraveledPath(prevPath => [...prevPath, [newLon, newLat]]);
          const dx = endLon - startLon;
          const dy = endLat - startLat;
          setShipAngle(Math.atan2(dy, dx) * 180 / Math.PI);
          if (t === 1) {
            step++;
            progress = 0;
          }
          animationRef.current = timestamp;
          requestAnimationFrame(animateShip);
        } else {
          setIsAnimating(false);
          animationRef.current = null;
        }
      };
      requestAnimationFrame(animateShip);
    }
  };

  const createCustomIcon = (svgString: string) => L.divIcon({
    className: 'custom-icon',
    html: `<div style="color: black;">${svgString}</div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 24],
  });

  const startIcon = createCustomIcon('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="3"/><line x1="12" y1="22" x2="12" y2="8"/><path d="M5 12H2a10 10 0 0 0 20 0h-3"/></svg>');
  const endIcon = createCustomIcon('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg>');
  const searchIcon = createCustomIcon('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>');

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={[defaultCenter[1], defaultCenter[0]]}
        zoom={defaultZoom}
        style={{ height: '100%', width: '100%' }}
        className="z-0"
        whenCreated={mapInstance => { mapRef.current = mapInstance; }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {route && <Polyline positions={route.map(([lon, lat]) => [lat, lon])} color="blue" weight={3} opacity={0.7} />}
        {traveledPath.length > 1 && <Polyline positions={traveledPath.map(([lon, lat]) => [lat, lon])} color="#10B981" weight={4} />}
        {startPort && <Marker position={[startPort[1], startPort[0]]} icon={startIcon}><Popup>Start Port</Popup></Marker>}
        {endPort && <Marker position={[endPort[1], endPort[0]]} icon={endIcon}><Popup>End Port</Popup></Marker>}
        {shipPosition && <ShipMarker position={shipPosition} angle={shipAngle} />}
        {showWeather && <TileLayer url={`https://tile.openweathermap.org/map/temp_new/{z}/{x}/{y}.png?appid=YOUR_API_KEY`} attribution='Weather data © OpenWeatherMap' />}
        {(searchResults || []).map((result, index) => <Marker key={index} position={[result[1], result[0]]} icon={searchIcon}><Popup>Search Result {index + 1}</Popup></Marker>)}
        <MapEventHandler isSelectingLocation={isSelectingLocation} onLocationSelect={onLocationSelect} />
        <ZoomHandler zoomToLocation={zoomToLocation} defaultCenter={defaultCenter} defaultZoom={defaultZoom} />
      </MapContainer>
      {route && !isAnimating && (
        <Button
          onClick={startAnimation}
          className="absolute bottom-4 left-4 z-[1000] bg-emerald-500 hover:bg-emerald-600 text-white"
        >
          Start Animation
        </Button>
      )}
    </div>
  );
});

LeafletMap.displayName = 'LeafletMap';
export default LeafletMap;

