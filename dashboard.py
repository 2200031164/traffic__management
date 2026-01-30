import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Smart Traffic Dashboard", layout="centered")

st.title("🚦 Smart Traffic Monitoring Dashboard")

# Initialize placeholders
congestion_placeholder = st.empty()
frame_placeholder = st.empty()
queue_chart = st.empty()
alert_placeholder = st.empty()

# Simulate reading from a shared CSV (created during simulation)
while True:
    try:
        df = pd.read_csv("congestion_log.csv")
        latest = df.iloc[-1]
        frame = int(latest["Frame"])
        congestion = float(latest["Congestion"])
        
        congestion_placeholder.metric("📈 Congestion Level", f"{congestion:.2f}")
        frame_placeholder.text(f"📷 Current Frame: {frame}")
        
        queue_data = pd.read_csv("queue_log.csv")
        latest_queues = queue_data[queue_data["Frame"] == frame]

        chart_data = pd.DataFrame({
            "North Queue": latest_queues["N"].values,
            "South Queue": latest_queues["S"].values
        }, index=latest_queues["Junction"])

        queue_chart.bar_chart(chart_data)

        if (chart_data["North Queue"] > 5).any() or (chart_data["South Queue"] > 5).any():
            alert_placeholder.warning("⚠️ Congestion Alert: Queues exceed threshold at one or more junctions!")

        time.sleep(1)

    except Exception as e:
        st.info("⏳ Waiting for simulation data...")
                # === Grid Map of Junctions ===
        grid_placeholder = st.empty()
        grid_status = {}
        for idx, row in latest_queues.iterrows():
            total_queue = int(row["N"]) + int(row["S"])
            if total_queue > 10:
                grid_status[row["Junction"]] = "🔴"
            elif total_queue > 5:
                grid_status[row["Junction"]] = "🟡"
            else:
                grid_status[row["Junction"]] = "🟢"

        grid_layout = f"""
        <style>
            .junction-grid {{
                display: grid;
                grid-template-columns: 100px 100px;
                grid-gap: 10px;
                justify-content: center;
                font-size: 32px;
                font-weight: bold;
            }}
        </style>
        <div class="junction-grid">
            <div>{grid_status.get("A", "⬜")} A</div>
            <div>{grid_status.get("B", "⬜")} B</div>
            <div>{grid_status.get("C", "⬜")} C</div>
            <div>{grid_status.get("D", "⬜")} D</div>
        </div>
        """
        grid_placeholder.markdown(grid_layout, unsafe_allow_html=True)

        
        
        
        
        time.sleep(1)
