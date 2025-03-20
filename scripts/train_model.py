import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
watermarked_dir = r"C:\Users\souja\Documents\watermark_detection_project\dataset\watermarked"
masks_dir = r"C:\Users\souja\Documents\watermark_detection_project\dataset\masks"
model_save_path = r"C:\Users\souja\Documents\watermark_detection_project\models\watermark_extractor.h5"

# Image parameters
img_size = (256, 256)

# Load dataset
def load_data():
    watermarked_images = []
    masks = []
    image_files = os.listdir(watermarked_dir)

    for img_name in image_files:
        watermarked_path = os.path.join(watermarked_dir, img_name)
        mask_path = os.path.join(masks_dir, img_name)

        if not os.path.exists(mask_path):  # Ensure corresponding mask exists
            continue

        # Load images
        watermarked_img = img_to_array(load_img(watermarked_path, target_size=img_size)) / 255.0
        mask_img = img_to_array(load_img(mask_path, target_size=img_size, color_mode="grayscale")) / 255.0

        watermarked_images.append(watermarked_img)
        masks.append(mask_img)

    return np.array(watermarked_images), np.array(masks)

# Define U-Net model for watermark extraction
def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv4)
    merge1 = concatenate([conv3, up1], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge1)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    merge2 = concatenate([conv2, up2], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge2)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    merge3 = concatenate([conv1, up3], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(conv7)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Load data
X_train, y_train = load_data()

# Train model
if len(X_train) > 0:
    model = build_unet((*img_size, 3))
    model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.1)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
else:
    print("No training data found!")
