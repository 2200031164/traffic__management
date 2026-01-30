from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from simulation_gui_logic import get_current_queue_data, predictor
import random

app = FastAPI()

# Allow frontend (React) to fetch data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later to ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frame_counter = 0
@app.get("/traffic-data")
def traffic_data():
    global frame_counter
    frame_counter += 1

    # === Get current queue sizes (list of dicts) ===
    queues = get_current_queue_data()  

    # === Predictions ===
    predictions = predictor.predict_next()

    # === Alerts & Emergency Flags ===
    alerts = []
    for q in queues:   # ✅ iterate over list
        j = q["junction"]

        # Check north direction
        if q.get("north", 0) > 5 or q.get("north_queue", 0) > 5:
            alerts.append({
                "junction": j,
                "type": "north_congestion",
                "message": f"Northbound congestion at {j} ({q.get('north', q.get('north_queue', 0))} vehicles)"
            })

        # Check south direction
        if q.get("south", 0) > 5 or q.get("south_queue", 0) > 5:
            alerts.append({
                "junction": j,
                "type": "south_congestion",
                "message": f"Southbound congestion at {j} ({q.get('south', q.get('south_queue', 0))} vehicles)"
            })

        # Random emergency event for demo
        if random.random() < 0.05:
            q["emergency"] = True   # 🔥 flag added directly in queue
            alerts.append({
                "junction": j,
                "type": "emergency",
                "message": f"🚑 Emergency vehicle detected at {j}!"
            })
        else:
            q["emergency"] = False  # ensure field always exists

    return {   # ✅ return after loop completes
        "frame": frame_counter,
        "queues": queues,        # already a list
        "predictions": predictions,
        "alerts": alerts
    }



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
