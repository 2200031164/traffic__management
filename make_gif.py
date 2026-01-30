import imageio
import os

def create_gif_from_frames(folder="frames_bar", output="traffic_simulation.gif", fps=2):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    images = [imageio.imread(os.path.join(folder, f)) for f in files]
    imageio.mimsave(output, images, fps=fps)
    print(f"✅ GIF saved as {output}")

if __name__ == "__main__":
    create_gif_from_frames()
