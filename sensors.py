
import random

def get_vehicle_count(junction):
    return {road: len(junction.queues[road]) for road in junction.queues}

def detect_emergency_vehicle():
    return random.choice([True, False, False, False])
import random
import time

def get_mock_congestion():
    # Simulate realistic congestion between 1.0 and 2.0
    time_of_day = time.localtime().tm_hour

    if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19:
        # Morning/Evening rush hours
        congestion = round(random.uniform(1.4, 1.8), 2)
    else:
        # Off-peak hours
        congestion = round(random.uniform(1.0, 1.3), 2)

    return congestion
