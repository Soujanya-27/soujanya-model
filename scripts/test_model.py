import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths
model_path = r"C:\Users\souja\Documents\watermark_detection_project\models\watermark_extractor.h5"
test_images_dir = r"C:\Users\souja\Documents\watermark_detection_project\dataset\test_images"
output_dir = r"C:\Users\souja\Documents\watermark_detection_project"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load trained model
model = load_model(model_path)

# Preprocess function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None, None
    original_size = img.shape[:2]  # (height, width)
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    return np.expand_dims(img_resized, axis=0), original_size

# Recover watermark
def recover_watermark(image_path):
    image, original_size = preprocess_image(image_path)
    if image is None:
        return
    
    extracted_mask = model.predict(image)[0]
    extracted_mask = (extracted_mask * 255).astype(np.uint8)  # Convert back to 0-255 scale
    extracted_mask_resized = cv2.resize(extracted_mask, (original_size[1], original_size[0]))

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, extracted_mask_resized)
    print(f"Recovered watermark saved: {output_path}")

# Process all test images
for img_name in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_name)
    recover_watermark(img_path)
