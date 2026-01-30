# simulation_gui_logic.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, RegularPolygon
from collections import deque
import random, copy
import numpy as np

from vehicle import Vehicle
from junction import Junction
from crd import find_path
from cbo import optimize_signals
from sensors import get_mock_congestion
from predictor import TrafficPredictor

# === Configurations ===
JUNCTION_POSITIONS = {
    'A': (1, 5), 'B': (3, 5), 'C': (5, 5),
    'D': (1, 3), 'E': (3, 3), 'F': (5, 3),
    'G': (1, 1), 'H': (3, 1)
}
GRAPH = {
    'A': {'B': 1, 'D': 1},
    'B': {'A': 1, 'C': 1, 'E': 1},
    'C': {'B': 1, 'F': 1},
    'D': {'A': 1, 'E': 1, 'G': 1},
    'E': {'B': 1, 'D': 1, 'F': 1, 'H': 1},
    'F': {'C': 1, 'E': 1},
    'G': {'D': 1, 'H': 1},
    'H': {'G': 1, 'E': 1}
}
PHEROMONE = {node: {nbr: 1.0 for nbr in nbrs} for node, nbrs in GRAPH.items()}

INITIAL_VEHICLES = 12
SPAWN_PROB_PER_FRAME = 0.35
EMERGENCY_PROB = 0.12
SPEED_MEAN = 0.18
SPEED_VAR = 0.06
PRED_HISTORY_LEN = 12
CONGESTION_THRESHOLD = 7.0

# === State ===
junctions = {name: Junction(name) for name in JUNCTION_POSITIONS}
vehicles = []
graph = copy.deepcopy(GRAPH)
pheromone = copy.deepcopy(PHEROMONE)

# Traffic Predictor
predictor = TrafficPredictor(list(junctions.keys()), history_len=PRED_HISTORY_LEN)

def ensure_vehicle_attrs(v):
    if not hasattr(v, "path"): v.path = []
    if not hasattr(v, "position_index"): v.position_index = 0
    if not hasattr(v, "progress"): v.progress = 0.0
    if not hasattr(v, "speed"):
        v.speed = max(0.05, random.gauss(SPEED_MEAN, SPEED_VAR))
    if not hasattr(v, "is_emergency"):
        v.is_emergency = False

def spawn_vehicle(next_id=None):
    names = list(junctions.keys())
    start, end = random.sample(names, 2)
    vid = f"V{next_id}" if next_id is not None else f"V{len(vehicles)}"
    v = Vehicle(vid, start, end)
    ensure_vehicle_attrs(v)
    try:
        congestion = get_mock_congestion()
    except Exception:
        congestion = 1.0
    try:
        v.path = find_path(start, end, graph, pheromone, congestion)
    except TypeError:
        v.path = find_path(start, end, graph, pheromone)
    v.is_emergency = (random.random() < EMERGENCY_PROB)
    junctions[start].queues['N'].append(v)
    vehicles.append(v)

# Initial spawn
for i in range(INITIAL_VEHICLES):
    spawn_vehicle(i)

def get_next_frame_plot(frame_index):
    global vehicles

    # Spawn new vehicle
    if random.random() < SPAWN_PROB_PER_FRAME:
        spawn_vehicle()

    # Optimize signals
    try:
        config = optimize_signals({}, {}, {}, {})
    except:
        config = {name: 30 for name in junctions}

    # Process junction queues
    for j in junctions.values():
        j.update_signals(config)
        for d in ['N', 'S']:
            if j.queues[d] and j.signal.get(d, 'red') == 'green':
                j.queues[d].pop(0)  # Vehicle leaves queue

    # Move vehicles
    survivors = []
    for v in vehicles:
        ensure_vehicle_attrs(v)
        if not v.path or v.position_index >= len(v.path) - 1:
            continue
        speed_factor = 1.35 if v.is_emergency else 1.0
        v.progress += v.speed * speed_factor
        if v.progress >= 1.0:
            v.progress -= 1.0
            v.position_index += 1
            if v.position_index < len(v.path) - 1:
                next_junc = v.path[v.position_index]
                direction = 'S' if JUNCTION_POSITIONS[next_junc][1] < JUNCTION_POSITIONS[v.path[v.position_index - 1]][1] else 'N'
                if v not in junctions[next_junc].queues[direction]:
                    junctions[next_junc].queues[direction].append(v)
            else:
                continue  # Vehicle finished journey
        survivors.append(v)
    vehicles = survivors

    # Queue counts
    current_totals = {name: len(j.queues['N']) + len(j.queues['S']) for name, j in junctions.items()}
    predictor.add_observation(current_totals)
    predictor.train_if_needed()
    predictions = predictor.predict_next()

    # === Bar Chart (Grouped) ===
    labels = list(junctions.keys())
    south_counts = [len(junctions[j].queues['S']) for j in labels]
    north_counts = [len(junctions[j].queues['N']) for j in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig_bar, ax1 = plt.subplots(figsize=(5, 4))
    ax1.bar(x - width/2, south_counts, width, color='green', label='South (S)')
    ax1.bar(x + width/2, north_counts, width, color='red', label='North (N)')

    for i, val in enumerate(south_counts):
        ax1.text(x[i] - width/2, val + 0.1, str(val), ha='center', fontsize=8)
    for i, val in enumerate(north_counts):
        ax1.text(x[i] + width/2, val + 0.1, str(val), ha='center', fontsize=8)

    for i, j in enumerate(labels):
        ax1.scatter(x[i], max(south_counts[i], north_counts[i]) + 0.5,
                    marker='D', color='purple', s=40)
        ax1.text(x[i], max(south_counts[i], north_counts[i]) + 0.8,
                 f"{predictions.get(j,0):.1f}", color='purple', ha='center', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(max(south_counts), max(north_counts)) + 5)
    ax1.set_ylabel("Vehicles")
    ax1.set_title(f"Frame {frame_index} – Queue Sizes (Prediction in Purple)")
    ax1.legend()
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

    # === Vehicle Map ===
    fig_map, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 6)
    ax2.grid(True, linestyle=':', color='lightgray', alpha=0.6)

    for start, conns in graph.items():
        for end in conns:
            s = JUNCTION_POSITIONS[start]
            e = JUNCTION_POSITIONS[end]
            ax2.plot([s[0], e[0]], [s[1], e[1]], color="#666666", linewidth=2)
            mx, my = (s[0] + e[0]) / 2, (s[1] + e[1]) / 2
            dx, dy = (e[0] - s[0]) * 0.28, (e[1] - s[1]) * 0.28
            ax2.add_patch(FancyArrowPatch((mx - dx, my - dy), (mx + dx, my + dy),
                                          arrowstyle='-|>', mutation_scale=10, color="#999999"))

    for name, pos in JUNCTION_POSITIONS.items():
        ax2.plot(pos[0], pos[1], 'ko', markersize=10)
        ax2.text(pos[0], pos[1] + 0.18, name, ha='center', fontsize=11, fontweight='bold')

    for v in vehicles:
        if not v.path or v.position_index >= len(v.path) - 1:
            continue
        s_pos = JUNCTION_POSITIONS[v.path[v.position_index]]
        e_pos = JUNCTION_POSITIONS[v.path[v.position_index + 1]]
        x_pos = s_pos[0] + (e_pos[0] - s_pos[0]) * v.progress
        y_pos = s_pos[1] + (e_pos[1] - s_pos[1]) * v.progress
        if v.is_emergency:
            tri = RegularPolygon((x_pos, y_pos), numVertices=3, radius=0.18, orientation=0.5, color='red')
            ax2.add_patch(tri)
        else:
            ax2.plot(x_pos, y_pos, 'o', markersize=6, color='#1f77b4')

    ax2.set_title("Vehicle Flow Map")
    ax2.axis('off')
    return fig_bar, fig_map
    
# === Global traffic queues (persistent state) ===
traffic_queues = {
    f"Junction {j}": {"north_queue": random.randint(0, 2), "south_queue": random.randint(0, 2)}
    for j in ["A", "B", "C", "D", "E", "F", "G", "H"]
}

def get_current_queue_data():
    """Simulate realistic live queue changes for 8 junctions (A–H)."""
    global traffic_queues

    for j in traffic_queues.keys():
        for direction in ["north_queue", "south_queue"]:
            # Gradual change: -1, 0, or +1
            change = random.choice([-1, 0, 1])

            new_val = traffic_queues[j][direction] + change

            # Keep values between 0 and 10
            traffic_queues[j][direction] = max(0, min(new_val, 10))

    # Return as list of dicts
    queues = [{"junction": j,
               "north_queue": q["north_queue"],
               "south_queue": q["south_queue"]}
              for j, q in traffic_queues.items()]

    return queues



    return fig_bar, fig_map
plt.show()
