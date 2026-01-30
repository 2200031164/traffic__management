from map_arrows import draw_vehicle_map
from vehicle import Vehicle
from junction import Junction
from crd import find_path
from cbo import optimize_signals
from sensors import get_mock_congestion

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import random
import csv

# === Create output folders ===
os.makedirs("frames_bar", exist_ok=True)
os.makedirs("frames_map", exist_ok=True)

# === Define 4 junctions in a grid ===
junctions = {
    'A': Junction('A'),
    'B': Junction('B'),
    'C': Junction('C'),
    'D': Junction('D')
}

# === Define the graph connections ===
graph = {
    'A': {'B': 1, 'C': 1},
    'B': {'A': 1, 'D': 1},
    'C': {'A': 1, 'D': 1},
    'D': {'B': 1, 'C': 1}
}

# === Initialize pheromones for CRD pathfinding ===
pheromone = {node: {nbr: 1.0 for nbr in nbrs} for node, nbrs in graph.items()}

# === Initialize log files ===
with open("congestion_log.csv", "w", newline="") as f:
    csv.writer(f).writerow(["Frame", "Congestion"])
with open("queue_log.csv", "w", newline="") as f:
    csv.writer(f).writerow(["Frame", "Junction", "N", "S"])

# === Create vehicles with random origin and destination ===
junction_names = list(junctions.keys())
vehicles = []

congestion = get_mock_congestion()
print(f"[Frame 0] Congestion Level: {congestion:.2f}")

for i in range(10):
    start, end = random.sample(junction_names, 2)
    v = Vehicle(f'V{i}', start, end)
    v.is_emergency = random.random() < 0.2  # 20% chance it's emergency
    v.path = find_path(start, end, graph, pheromone, congestion)
    junctions[start].queues['N'].append(v)
    vehicles.append(v)


# === Signal control variables ===
stats = {'N': 10, 'S': 20}
pbest = {'N': 30, 'S': 20}
gbest = {'N': 25, 'S': 25}
velocity = {'N': 0, 'S': 0}

# === Simulation Loop ===
for frame in range(10):
    print(f"--- Frame {frame} ---")

    # Update congestion and log it
    congestion = get_mock_congestion()
    print(f"[Frame {frame}] Congestion Level: {congestion:.2f}")
    with open("congestion_log.csv", "a", newline="") as f:
        csv.writer(f).writerow([frame, congestion])

    # === Traffic Peak Injection ===
    peak_junctions = random.sample(junction_names, k=random.randint(0, 2))
    for j_name in peak_junctions:
        for _ in range(random.randint(2, 4)):
            start = j_name
            end = random.choice([j for j in junction_names if j != start])
            v = Vehicle(f'PV{len(vehicles)}', start, end)
            v.is_emergency = random.random() < 0.2  # 20% chance
            v.path = find_path(start, end, graph, pheromone, congestion)
            junctions[start].queues['N'].append(v)
            vehicles.append(v)


    # === Signal Optimization and Junction Processing ===
    config = optimize_signals(stats, pbest, gbest, velocity)
    for j in junctions.values():
        j.update_signals(config)
        j.process()

    # === Collect and log queue lengths ===
    top = []     # North queues
    bottom = []  # South queues
    labels = []

    with open("queue_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for name in junctions:
            j = junctions[name]
            labels.append(name)
            n_len = len(j.queues['N'])
            s_len = len(j.queues['S'])
            top.append(n_len)
            bottom.append(s_len)
            writer.writerow([frame, name, n_len, s_len])

    # === Color bars by congestion ===
    colors_top = ['#d62728' if q > 5 else '#1f77b4' for q in top]
    colors_bottom = ['#ff7f0e' if q > 5 else '#2ca02c' for q in bottom]

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(8, 6))
    is_congested = any(q > 6 for q in top + bottom)

    if is_congested:
        ax.set_title(f"🚨 Congestion Alert! Frame {frame} | Congestion: {congestion:.2f}",
                     color='red', fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"🚦 Traffic Queue Status – Frame {frame} | Congestion: {congestion:.2f}",
                     fontsize=14, fontweight='bold')

    bars_bottom = ax.bar(labels, bottom, color=colors_bottom, label='South (S)')
    bars_top = ax.bar(labels, top, bottom=bottom, color=colors_top, label='North (N)')

    for i, (t, b) in enumerate(zip(top, bottom)):
        if t > 0:
            ax.text(i, b + t / 2, str(t), ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        if b > 0:
            ax.text(i, b / 2, str(b), ha='center', va='center', color='black', fontsize=10, fontweight='bold')
            
            
                # 🚑 Show emergency vehicle count above bars
    emergency_count = sum(
        1 for v in junctions[labels[i]].queues['N'] + junctions[labels[i]].queues['S']
        if getattr(v, 'is_emergency', False)
    )
    if emergency_count > 0:
        ax.text(i, top[i] + bottom[i] + 0.5,
                f"🚑{emergency_count}", ha='center', color='red', fontsize=11, fontweight='bold')


    ax.set_ylabel("🚗 Vehicles in Queue", fontsize=12)
    ax.set_xlabel("Junctions", fontsize=12)
    ax.set_ylim(0, 12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"frames_bar/frame_{frame:03}.png", dpi=150)
    plt.close()

    # === Vehicle Path Map ===
    draw_vehicle_map(junctions, vehicles, frame)
