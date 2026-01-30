# map_arrows.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Fixed positions for each junction (for visualization)
junction_positions = {
    'A': (1, 4),
    'B': (4, 4),
    'C': (1, 1),
    'D': (4, 1)
}

def draw_vehicle_map(junctions, vehicles, frame=0):
    os.makedirs("frames_map", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw junctions
    for jname, pos in junction_positions.items():
        ax.plot(pos[0], pos[1], 'ko', markersize=10)
        ax.text(pos[0], pos[1] + 0.2, jname, ha='center', fontsize=12, fontweight='bold')

    # Draw vehicle arrows
    for v in vehicles:
        if not v.path or len(v.path) < 2:
            continue
        start = junction_positions.get(v.path[0])
        end = junction_positions.get(v.path[1])
        if start and end:
            arrow = patches.FancyArrowPatch(
                start, end,
                arrowstyle='->',
                color='red' if getattr(v, 'is_emergency', False) else 'blue',
                mutation_scale=15,
                linewidth=2
            )
            ax.add_patch(arrow)
            label = f"🚑 {v.name}" if getattr(v, 'is_emergency', False) else v.name


            ax.text((start[0]+end[0])/2, (start[1]+end[1])/2 + 0.1,
        label, fontsize=9, color='red' if getattr(v, 'is_emergency', False) else 'green',
        fontweight='bold')

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title(f"Vehicle Paths – Frame {frame}")
    plt.tight_layout()
    plt.savefig(f"frames_map/map_{frame:03}.png")
    plt.close()
