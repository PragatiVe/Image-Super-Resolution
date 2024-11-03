import os
from PIL import Image

# List of directories containing the images
DATASET_DIRS = [
    'archive(4)/Testing/glioma',
    'archive(4)/Testing/meningioma',
    'archive(4)/Testing/notumor',
    'archive(4)/Testing/pituitary'
]

# Directory where the low-resolution images will be saved
OUTPUT_DIR = 'LowResolution'

# Accepted image file extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Process each dataset directory
for dataset_dir in DATASET_DIRS:
    # Get the name of the subdirectory (e.g., 'glioma')
    subdir_name = os.path.basename(dataset_dir)
    # Create a corresponding subdirectory in the output directory
    output_subdir = os.path.join(OUTPUT_DIR, subdir_name)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    # Process each file in the dataset directory
    for filename in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, filename)
        # Check if the file is an image
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(file_path) as img:
                    # Resize the image to 64x64 pixels
                    img_resized = img.resize((64, 64))
                    # Save the resized image in the output subdirectory
                    output_file_path = os.path.join(output_subdir, filename)
                    img_resized.save(output_file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
