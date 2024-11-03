import os
import glob
import numpy as np
from PIL import Image
from sewar.full_ref import mse, psnr, ssim, vifp

# Lists containing the paths to your low-resolution and super-resolution image directories
lr_dirs = [
    'LowResolution/Training/glioma',
    'LowResolution/Training/meningioma',
    'LowResolution/Training/pituitary',
    'LowResolution/Training/notumor'
    # Add more paths if needed
]

sr_dirs = [
    'HighResolutionInceptionandResidual/Training/glioma',
    'HighResolutionInceptionandResidual/Training/meningioma',
    'HighResolutionInceptionandResidual/Training/pituitary',
    'HighResolutionInceptionandResidual/Training/notumor'
    # Add more paths if needed
]

# Ensure that the number of directories matches
assert len(lr_dirs) == len(sr_dirs), "Mismatch in the number of directories."

# Lists to store the metric results
mse_list = []
psnr_list = []
ssim_list = []
vif_list = []

# Iterate over directory pairs
for lr_dir, sr_dir in zip(lr_dirs, sr_dirs):
    # Get sorted lists of image file paths in each directory
    lr_images = sorted(glob.glob(os.path.join(lr_dir, '*')))
    sr_images = sorted(glob.glob(os.path.join(sr_dir, '*')))
    
    # Ensure that the number of images matches
    assert len(lr_images) == len(sr_images), f"Mismatch in number of images between {lr_dir} and {sr_dir}"
    
    # Iterate over image pairs
    for lr_path, sr_path in zip(lr_images, sr_images):
        # Load images and convert to RGB
        lr_image = Image.open(lr_path).convert('RGB')
        sr_image = Image.open(sr_path).convert('RGB')

        # Resize the LR image to match the SR image size
        lr_resized = lr_image.resize(sr_image.size, Image.BICUBIC)

        # Convert images to NumPy arrays with uint8 data type
        lr_array = np.array(lr_resized).astype(np.uint8)
        sr_array = np.array(sr_image).astype(np.uint8)

        # Compute the metrics
        mse_value = mse(sr_array, lr_array)
        psnr_value = psnr(sr_array, lr_array)  # No need to specify MAX if data type is uint8
        ssim_value = ssim(sr_array, lr_array)[0]  # ssim returns a tuple
        vif_value = vifp(sr_array, lr_array)

        # Append results to the lists
        mse_list.append(mse_value)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        vif_list.append(vif_value)

        # Print the results for each image
        print(f"Image: {os.path.basename(sr_path)}")
        print(f"  MSE:  {mse_value:.4f}")
        print(f"  PSNR: {psnr_value:.4f} dB")
        print(f"  SSIM: {ssim_value:.4f}")
        print(f"  VIF:  {vif_value:.4f}")
        print("-" * 30)

# Calculate and print average metrics
print("Average Metrics:")
print(f"Average MSE:  {np.mean(mse_list):.4f}")
print(f"Average PSNR: {np.mean(psnr_list):.4f} dB")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")
print(f"Average VIF:  {np.mean(vif_list):.4f}")
