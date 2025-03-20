import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

# Define paths
clean_images_dir = r"C:\Users\souja\Documents\watermark_detection_project\dataset\clean_images"
watermarks_dir = r"C:\Users\souja\Documents\watermark_detection_project\dataset\watermarks"
watermarked_dir = r"C:\Users\souja\Documents\watermark_detection_project\dataset\watermarked"
masks_dir = r"C:\Users\souja\Documents\watermark_detection_project\dataset\masks"

# Ensure output directories exist
os.makedirs(watermarks_dir, exist_ok=True)
os.makedirs(watermarked_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Function to generate a random watermark text
def generate_random_text():
    words = ["CONFIDENTIAL", "WATERMARK", "PRIVATE", "COPYRIGHT", "SECURED"]
    return random.choice(words)

# Function to create watermark image
def create_watermark(text, image_size, save_path=None):
    watermark = Image.new("L", image_size, 0)  # Create blank grayscale image
    draw = ImageDraw.Draw(watermark)
    
    # Load a font (adjust path if needed)
    font = ImageFont.load_default()
    
    # Positioning watermark text randomly
    bbox = draw.textbbox((0, 0), text, font=font)
    text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    position = (random.randint(0, image_size[0] - text_size[0]), random.randint(0, image_size[1] - text_size[1]))
    
    # Draw watermark
    draw.text(position, text, fill=255, font=font)
    
    # Save the watermark image if save_path is provided
    if save_path:
        watermark.save(save_path)
    
    return np.array(watermark)

# Process each clean image
for img_name in os.listdir(clean_images_dir):
    img_path = os.path.join(clean_images_dir, img_name)

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping {img_name}, unable to load image.")
        continue

    # Resize to standard size
    image = cv2.resize(image, (256, 256))

    # Generate watermark
    watermark_text = generate_random_text()
    watermark_path = os.path.join(watermarks_dir, f"watermark_{img_name}.png")
    watermark = create_watermark(watermark_text, (256, 256), save_path=watermark_path)

    # Convert watermark to 3-channel (match image channels)
    watermark = cv2.merge([watermark] * 3)

    # Apply watermark
    alpha = 0.15  # Adjust transparency (0.1 to 0.2 works well)
    watermarked_img = cv2.addWeighted(image, 1, watermark, alpha, 0)

    # Save watermarked image and mask
    watermarked_path = os.path.join(watermarked_dir, img_name)
    mask_path = os.path.join(masks_dir, img_name)

    cv2.imwrite(watermarked_path, watermarked_img)
    cv2.imwrite(mask_path, watermark)

    print(f"Generated: {img_name}")

print("Dataset generation complete!")
