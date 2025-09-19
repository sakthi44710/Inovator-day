from flask import Flask, request, jsonify
from flask_cors import CORS
import netCDF4 as nc
import numpy as np
import math
import heapq
import os
from numba import njit
from scipy.interpolate import NearestNDInterpolator

app = Flask(__name__)
# Allow requests from your frontend development server
CORS(app, resources={r"/optimize_route": {"origins": "http://localhost:3000"}})

# --- Constants ---
SPEED_SCALING = {
    "passenger ship": 20,  # Knots
    "cargo ship": 15,
    "tanker": 10
}

# --- Utility Functions ---
@njit
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points on the earth."""
    R = 6371  # Earth radius in kilometers
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_data(file_path):
    """Safely load a NetCDF file, handling FileNotFoundError."""
    if not os.path.exists(file_path):
        # Fallback to a path relative to the script if the direct path fails
        script_dir = os.path.dirname(__file__)
        fallback_path = os.path.join(script_dir, file_path)
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(f"Data file not found at: {file_path} or {fallback_path}. Please check the path and environment variables.")
        return nc.Dataset(fallback_path)
    return nc.Dataset(file_path)


def interpolate_to_grid(src_data, src_lon, src_lat, dst_lon, dst_lat):
    """Interpolate source data to a destination grid using NearestNDInterpolator."""
    src_lon_grid, src_lat_grid = np.meshgrid(src_lon, src_lat)
    src_points = np.column_stack((src_lon_grid.ravel(), src_lat_grid.ravel()))

    # Ensure data is not masked before raveling
    if np.ma.is_masked(src_data):
        src_data = src_data.filled(np.nan)

    interpolator = NearestNDInterpolator(src_points, src_data.ravel())

    dst_lon_grid, dst_lat_grid = np.meshgrid(dst_lon, dst_lat)
    dst_points = np.column_stack((dst_lon_grid.ravel(), dst_lat_grid.ravel()))

    return interpolator(dst_points).reshape(len(dst_lat), len(dst_lon))

def get_interpolated_data():
    """
    Load and interpolate environmental data.
    Paths are loaded from environment variables for flexibility.
    """
    # Set default paths if environment variables are not found
    wave_path = os.environ.get('WAVE_DATA_PATH', 'Wavewatch_III_25_28_2024_to_03_09_2024.nc')
    roms_path = os.environ.get('ROMS_DATA_PATH', 'ROMS_25_08_2024_to_03_09_2024.nc')

    wave_data = load_data(wave_path)
    roms_data = load_data(roms_path)

    lon = wave_data.variables['LON'][:].data
    lat = wave_data.variables['LAT'][:].data
    swh = wave_data.variables['SWH'][0].data
    ws = wave_data.variables['WS'][0].data

    roms_lon = roms_data.variables['LON'][:].data
    roms_lat = roms_data.variables['LAT'][:].data
    sst = interpolate_to_grid(roms_data.variables['SST'][0, 0].data, roms_lon, roms_lat, lon, lat)

    wave_data.close()
    roms_data.close()

    return lon, lat, swh, ws, sst

def create_land_mask(data):
    """Create a land mask from data with NaNs or fill values."""
    return np.isnan(data) | (data < -1e5)

def find_nearest_node(lon_array, lat_array, lon_val, lat_val):
    """Find the nearest grid indices for a given lon/lat point."""
    lon_idx = np.abs(lon_array - lon_val).argmin()
    lat_idx = np.abs(lat_array - lat_val).argmin()
    return lon_idx, lat_idx

# --- A* Routing Algorithm ---
@njit
def heuristic(node, end_node, num_lat, lon, lat):
    """Heuristic for A* algorithm: Haversine distance to the destination."""
    i1, j1 = divmod(node, num_lat)
    i2, j2 = divmod(end_node, num_lat)
    return haversine(lon[i1], lat[j1], lon[i2], lat[j2])

def a_star(start, end, lon, lat, speed, swh, ws, land_mask):
    """A* algorithm to find the optimal path."""
    num_lon, num_lat = len(lon), len(lat)
    num_nodes = num_lon * num_lat

    g_score = np.full(num_nodes, np.inf)
    g_score[start] = 0
    f_score = np.full(num_nodes, np.inf)
    f_score[start] = heuristic(start, end, num_lat, lon, lat)

    predecessors = np.full(num_nodes, -1, dtype=np.int32)
    queue = [(f_score[start], start)]

    while queue:
        _, current_node = heapq.heappop(queue)

        if current_node == end:
            path = []
            while current_node != -1:
                path.append(current_node)
                current_node = predecessors[current_node]
            path.reverse()
            return g_score[end], path

        i, j = divmod(current_node, num_lat)

        # Explore neighbors (up, down, left, right, and diagonals)
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < num_lon and 0 <= nj < num_lat and not land_mask[nj, ni]:
                neighbor = ni * num_lat + nj

                dist = haversine(lon[i], lat[j], lon[ni], lat[nj])
                weather_factor = 1 + 0.1 * swh[nj, ni] + 0.05 * ws[nj, ni]
                time_cost = (dist / speed) * weather_factor

                tentative_g_score = g_score[current_node] + time_cost

                if tentative_g_score < g_score[neighbor]:
                    predecessors[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end, num_lat, lon, lat)
                    heapq.heappush(queue, (f_score[neighbor], neighbor))

    return np.inf, [] # Return empty path if no route is found

# --- Flask Route ---
@app.route('/optimize_route', methods=['POST'])
def optimize_route_endpoint():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    app.logger.info(f"Received data: {data}")

    # --- Input Validation ---
    required_fields = ['shipType', 'startPort', 'endPort', 'departureDate']
    missing_fields = [field for field in required_fields if field not in data or not data[field]]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    try:
        ship_type = data['shipType'].lower()
        if ship_type not in SPEED_SCALING:
            return jsonify({"error": f"Invalid ship type: {data['shipType']}"}), 400
        
        start_port = data['startPort']
        end_port = data['endPort']
        speed = SPEED_SCALING[ship_type]

        # --- Data Loading and Processing ---
        lon, lat, swh, ws, sst = get_interpolated_data()
        land_mask = create_land_mask(sst)
        
        start_i, start_j = find_nearest_node(lon, lat, start_port[0], start_port[1])
        end_i, end_j = find_nearest_node(lon, lat, end_port[0], end_port[1])

        if land_mask[start_j, start_i]:
            return jsonify({"error": "Start point is on land. Please select a point in the water."}), 400
        if land_mask[end_j, end_i]:
            return jsonify({"error": "End point is on land. Please select a point in the water."}), 400

        start_idx = start_i * len(lat) + start_j
        end_idx = end_i * len(lat) + end_j

        # --- Pathfinding ---
        travel_time, path_indices = a_star(start_idx, end_idx, lon, lat, speed, swh, ws, land_mask)

        if not path_indices:
            return jsonify({"error": "No valid route found between the selected points. They may be separated by land."}), 404
        
        # Convert path indices to coordinates
        optimized_route = [[float(lon[i]), float(lat[j])] for node in path_indices for i, j in [divmod(node, len(lat))]]

        # Calculate total distance from the path
        total_distance = 0
        for k in range(len(optimized_route) - 1):
            lon1, lat1 = optimized_route[k]
            lon2, lat2 = optimized_route[k+1]
            total_distance += haversine(lon1, lat1, lon2, lat2)


        return jsonify({
            "travel_time_hours": float(travel_time),
            "total_distance_km": float(total_distance),
            "optimized_route": optimized_route,
        }), 200

    except FileNotFoundError as e:
        app.logger.error(f"Data file error: {e}")
        return jsonify({"error": f"Server configuration error: A required data file was not found."}), 500
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please check the server logs."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
