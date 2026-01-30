

class Vehicle:
    def __init__(self, name, start, end):
        self.name = name
        self.path = []
        self.position_index = 0
        self.progress = 0.0  # 0 to 1
        self.speed = 0.1  # movement speed per frame
        self.current_coords = None
        self.start = start
        self.end = end
        self.is_emergency = False
