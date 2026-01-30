
import random

def optimize_signals(junction_stats, pbest, gbest, velocity):
    new_config = {}
    for road in junction_stats:
        v = velocity[road]
        personal = pbest[road]
        global_ = gbest[road]
        new_time = v + 0.5 * random.random() * (personal - junction_stats[road]) + \
                   0.3 * random.random() * (global_ - junction_stats[road])
        new_config[road] = max(5, min(60, int(new_time)))
    return new_config
