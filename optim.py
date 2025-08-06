import pandas as pd
import numpy as np
from geopy.distance import geodesic
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# === Step 1: Simulate Sensor Data ===
def generate_sensor_data(n=5, seed=42):
    np.random.seed(seed)
    data = {
        "id": range(n),
        "lat": np.random.uniform(40.60, 40.75, size=n),
        "lon": np.random.uniform(-74.02, -73.93, size=n),
        "temperature_c": np.random.normal(6.0, 2.0, size=n),
        "vibration_level": np.random.uniform(0.1, 0.4, size=n)
    }
    df = pd.DataFrame(data)
    df.to_csv("locations.csv", index=False)
    print(" Sensor data saved to locations.csv")

# === Step 2: Distance Matrix ===
def create_distance_matrix(locations):
    n = len(locations)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            dist = geodesic((locations[i]['lat'], locations[i]['lon']),
                            (locations[j]['lat'], locations[j]['lon'])).km
            row.append(dist)
        matrix.append(row)
    return matrix

# === Step 3: Route Optimization ===
def optimize_route():
    df = pd.read_csv("locations.csv")
    locations = df.to_dict('records')
    distance_matrix = create_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        return int(distance_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_params)

    if solution:
        print(" Optimized delivery route:")
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        print(" ➤ Route:", " -> ".join(map(str, route)))
    else:
        print(" No solution found.")

# === Step 4: Simulate Edge Cases ===
def simulate_heatwave():
    df = pd.read_csv("locations.csv")
    df["temperature_c"] += 10  # simulate extreme heat
    df.to_csv("locations.csv", index=False)
    print(" Heatwave simulated.")

def simulate_delays():
    df = pd.read_csv("locations.csv")
    df["vibration_level"] += 0.5  # simulate bumpy/delayed routes
    df.to_csv("locations.csv", index=False)
    print(" Route delays simulated.")

# === Step 5: Full Test ===
def test_all():
    print(" Generating normal scenario...")
    generate_sensor_data()
    optimize_route()

    print("\n Testing heatwave scenario...")
    simulate_heatwave()
    optimize_route()

    print("\n Testing delay scenario...")
    simulate_delays()
    optimize_route()

if __name__ == "__main__":
    test_all()