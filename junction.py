import csv

class Junction:
    def __init__(self, name):
        self.name = name
        self.queues = {'N': [], 'S': []}  # Queues for North and South directions
        self.signal = {'N': 'red', 'S': 'red'}  # Signal lights

    def update_signals(self, config):
        """Update signal status based on external config (e.g., congestion-based)."""
        if self.name in config:
            self.signal['N'] = 'green' if config[self.name] > 20 else 'red'
            self.signal['S'] = 'green' if config[self.name] <= 20 else 'red'

    def process(self):
  
     for direction in ['N', 'S']:
        queue = self.queues[direction]
        if not queue:
            continue

        # Emergency check
        emergency_found = getattr(queue[0], 'is_emergency', False)
        if emergency_found:
            self.signal[direction] = 'green'

        if self.signal[direction] == 'green':
            vehicle = queue.pop(0)  # ✅ Remove from queue
            print(f"{vehicle.name} passed from {direction} at junction {self.name}")

            # Log only after passing
            with open("vehicle_log.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    vehicle.name,
                    self.name,
                    direction,
                    "Emergency" if getattr(vehicle, 'is_emergency', False) else "Normal"
                ])

