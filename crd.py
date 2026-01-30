import random

def find_path(origin, destination, graph, pheromone, congestion, alpha=1, beta=2):
    path = [origin]
    current = origin
    while current != destination:
        neighbors = graph[current]
        probs = []
        adjusted_alpha = alpha / congestion  # less reliance on pheromones in high congestion
        for n in neighbors:
            tau = pheromone[current][n] ** adjusted_alpha
            eta = (1 / graph[current][n]) ** beta
            probs.append(tau * eta)
        total = sum(probs)
        probs = [p / total for p in probs]
        current = random.choices(list(neighbors.keys()), weights=probs)[0]
        path.append(current)
    return path
