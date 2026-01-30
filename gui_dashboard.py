import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from simulation_gui_logic import get_next_frame_plot

# =========================
# Global state
# =========================
frame_count = 0
is_paused = False
speed_ms = 2000  # Refresh rate in milliseconds (2 seconds)
bar_canvas = None
map_canvas = None

# =========================
# Tkinter setup
# =========================
root = tk.Tk()
root.title("🚦 Smart Traffic Management Dashboard")
root.geometry("1200x700")

# Title label
title = ttk.Label(root, text="Real-Time Traffic Monitoring", font=("Arial", 18, "bold"))
title.pack(pady=10)

# Container for plots
frame_container = ttk.Frame(root)
frame_container.pack(fill="both", expand=True)

# Queue chart on the left
bar_frame = ttk.LabelFrame(frame_container, text="Queue Status")
bar_frame.grid(row=0, column=0, padx=10, pady=10)

# Vehicle path on the right
map_frame = ttk.LabelFrame(frame_container, text="Vehicle Paths")
map_frame.grid(row=0, column=1, padx=10, pady=10)

# =========================
# Functions
# =========================

def update_frame():
    global frame_count, bar_canvas, map_canvas

    if not is_paused:
        # Clear old charts
        for widget in bar_frame.winfo_children():
            widget.destroy()
        for widget in map_frame.winfo_children():
            widget.destroy()

        # Get updated plots
        try:
            bar_fig, map_fig = get_next_frame_plot(frame_count)
        except Exception as e:
            print(f"Error generating frame {frame_count}: {e}")
            return

        # Show queue chart
        bar_canvas = FigureCanvasTkAgg(bar_fig, master=bar_frame)
        bar_canvas.draw()
        bar_canvas.get_tk_widget().pack()

        # Show traffic map
        map_canvas = FigureCanvasTkAgg(map_fig, master=map_frame)
        map_canvas.draw()
        map_canvas.get_tk_widget().pack()

        frame_count += 1

    # Schedule next frame regardless of pause (it'll skip if paused)
    root.after(speed_ms, update_frame)


def next_frame_manual():
    global is_paused
    is_paused = True
    update_frame()


def toggle_pause():
    global is_paused
    is_paused = not is_paused
    status = "⏸ Pause" if not is_paused else "▶ Resume"
    pause_btn.config(text=status)

# =========================
# Buttons
# =========================
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

ttk.Button(button_frame, text="⏭ Next Frame", command=next_frame_manual).grid(row=0, column=0, padx=10)
pause_btn = ttk.Button(button_frame, text="⏸ Pause", command=toggle_pause)
pause_btn.grid(row=0, column=1, padx=10)
# Add this after your Buttons Frame
legend_frame = ttk.LabelFrame(root, text="Legend", padding=10)
legend_frame.pack(pady=5, fill="x")

legend_text = (
    "🔹 Purple Diamond: Predicted Traffic Size\n"
    "🔺 Red Triangle: Emergency Vehicle\n"
    "🟥 Crimson Highlight: Most Congested Junction\n"
    "🟦 Blue / 🟩 Green: Normal Traffic Flow"
)
legend_label = ttk.Label(legend_frame, text=legend_text, justify="left")
legend_label.pack(anchor="w")

# =========================
# Start animation loop
# =========================
update_frame()

root.mainloop()
