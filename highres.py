import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from tqdm import tqdm

# Paths to the models
SUPER_RES_MODEL_PATH = 'Super_Resolution_Model.h5'
GENERATOR_MODEL_PATH = 'Gan_Generator.h5'  # If you want to use the generator

# Input images path
INPUT_IMAGES_PATH = 'LowResolution/Training/notumor'  # Replace with your input images path

# Output images path
OUTPUT_IMAGES_PATH = 'HighResolutionInceptionandResidual/Training/notumor'  # Replace with your desired output path

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)

# Load the super-resolution model
super_resolution_model = load_model(SUPER_RES_MODEL_PATH)

# Define image sizes
IMG_SIZE_LOW = 64   # Low-resolution size expected by the super-resolution model
IMG_SIZE_HIGH = 128 # High-resolution output size

# Function to super-resolve images from input path
def super_resolve_images(input_path, output_path):
    # Get list of input images
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_name in tqdm(image_files, desc='Processing images'):
        # Construct full image path
        image_path = os.path.join(input_path, image_name)
        
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image {image_path}")
            continue
        
        # Resize image to low resolution expected by the model
        img_low_res = cv2.resize(img, (IMG_SIZE_LOW, IMG_SIZE_LOW), interpolation=cv2.INTER_CUBIC)
        
        # Normalize the image to [-1, 1]
        img_low_res = img_low_res.astype(np.float32)
        img_low_res = img_low_res / 127.5 - 1.0  # Normalize to [-1, 1]
        
        # Expand dimensions to match model input shape
        img_input = np.expand_dims(img_low_res, axis=0)  # Shape: (1, 64, 64)
        img_input = np.expand_dims(img_input, axis=-1)   # Shape: (1, 64, 64, 1)
        
        # Generate super-resolved image
        img_sr = super_resolution_model.predict(img_input)
        
        # The output is in [-1, 1]; rescale back to [0, 255]
        img_sr = (img_sr + 1.0) * 127.5
        img_sr = np.clip(img_sr, 0, 255).astype(np.uint8)
        
        # Remove batch and channel dimensions
        img_sr = img_sr[0, :, :, 0]  # Shape: (128, 128)
        
        # Save the super-resolved image
        output_image_path = os.path.join(output_path, image_name)
        cv2.imwrite(output_image_path, img_sr)
        
    print(f"Super-resolved images have been saved to {output_path}")

# Call the function to process images
super_resolve_images(INPUT_IMAGES_PATH, OUTPUT_IMAGES_PATH)
