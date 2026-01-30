from PIL import Image
import os

# Folder where your images are saved
frame_dir = "frames_bar"
output_file = "combined_dashboard.png"

# Load all images
images = [Image.open(os.path.join(frame_dir, f)) for f in sorted(os.listdir(frame_dir)) if f.endswith(".png")]

# Number of columns in the grid
cols = 5
rows = (len(images) + cols - 1) // cols

# Get size of one image
img_width, img_height = images[0].size

# Create new blank image (white background)
combined = Image.new('RGB', (cols * img_width, rows * img_height), (255, 255, 255))

# Paste each image into the grid
for i, img in enumerate(images):
    x = (i % cols) * img_width
    y = (i // cols) * img_height
    combined.paste(img, (x, y))

# Save the final combined image
combined.save(output_file)
print(f"Saved: {output_file}")
