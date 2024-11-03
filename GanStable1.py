import warnings
warnings.filterwarnings('ignore')

import os
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Reshape, Conv2D, Conv2DTranspose, Dense,
                                     BatchNormalization, LeakyReLU, ReLU, Flatten, UpSampling2D)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

# =====================================
# 1. GPU Configuration
# =====================================
# Enable GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

print("Is GPU available:", tf.config.list_physical_devices('GPU'))

# =====================================
# 2. Parameters
# =====================================
IMG_SIZE_LOW = 64         # Low-resolution Image size (64x64)
IMG_SIZE_HIGH = 128       # High-resolution Image size (128x128) after super-resolution
BATCH_SIZE = 32           # Adjusted batch size
BUFFER_SIZE = 200         # Buffer size for shuffling
Z_DIM = 128               # Noise dimension for Generator
EPOCHS = 100              # Number of training epochs

# =====================================
# 3. Dataset Directories
# =====================================
DATASET_DIRS = [
    'archive(4)/Training/glioma',
    'archive(4)/Training/meningioma',
    'archive(4)/Training/notumor',
    'archive(4)/Training/pituitary'
]

# =====================================
# 4. Collect Image Paths
# =====================================
img_paths = []
for dataset_dir in tqdm(DATASET_DIRS, desc="Dataset Directories"):
    if not os.path.isdir(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        continue
    for filename in tqdm(os.listdir(dataset_dir), desc=f"Loading {dataset_dir}", leave=False):
        img_path = os.path.join(dataset_dir, filename)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_paths.append(img_path)

print(f"Total images found: {len(img_paths)}")

# =====================================
# 5. Define Mapping Functions for Low and High-Resolution Images
# =====================================
def map_fn_low(img_path):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=1)  # Grayscale
        img = tf.image.resize(img, (IMG_SIZE_LOW, IMG_SIZE_LOW))
        img = tf.cast(img, tf.float32) / 127.5 - 1    # Normalize to [-1, 1]
        return img
    except tf.errors.InvalidArgumentError:
        # Handle corrupted images by returning a black image
        return tf.zeros((IMG_SIZE_LOW, IMG_SIZE_LOW, 1), dtype=tf.float32)

def map_fn_high(img_path):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=1)  # Grayscale
        img = tf.image.resize(img, (IMG_SIZE_HIGH, IMG_SIZE_HIGH))
        img = tf.cast(img, tf.float32) / 127.5 - 1    # Normalize to [-1, 1]
        return img
    except tf.errors.InvalidArgumentError:
        # Handle corrupted images by returning a black image
        return tf.zeros((IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1), dtype=tf.float32)

# =====================================
# 6. Create TensorFlow Datasets
# =====================================
# Low-resolution dataset
dataset_low = tf.data.Dataset.from_tensor_slices(img_paths)
dataset_low = dataset_low.map(map_fn_low, num_parallel_calls=tf.data.AUTOTUNE)

# High-resolution dataset
dataset_high = tf.data.Dataset.from_tensor_slices(img_paths)
dataset_high = dataset_high.map(map_fn_high, num_parallel_calls=tf.data.AUTOTUNE)

print("Dataset shapes:")
print(f"Low-resolution dataset: {dataset_low}")
print(f"High-resolution dataset: {dataset_high}")

# =====================================
# 7. Fetch a Single Image Sample
# =====================================
try:
    img_sample_low = next(iter(dataset_low.batch(1)))
    img_sample_high = next(iter(dataset_high.batch(1)))
    print(f"Low-Resolution Image sample shape: {img_sample_low.shape}")    # Expected: (1, 64, 64, 1)
    print(f"High-Resolution Image sample shape: {img_sample_high.shape}")  # Expected: (1, 128, 128, 1)
except tf.errors.ResourceExhaustedError as e:
    print("Out of memory error encountered while fetching a single image sample:", e)
except StopIteration:
    print("The dataset is empty.")

# =====================================
# 8. Visualize Sample Images
# =====================================
if 'img_sample_low' in locals() and 'img_sample_high' in locals():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    img_low = img_sample_low[0].numpy().squeeze(-1)
    img_low = (img_low + 1) / 2.0  # Rescale to [0, 1] for display
    axs[0].imshow(img_low, cmap='gray')
    axs[0].set_title('Low-Resolution')
    axs[0].axis('off')
    
    img_high = img_sample_high[0].numpy().squeeze(-1)
    img_high = (img_high + 1) / 2.0  # Rescale to [0, 1] for display
    axs[1].imshow(img_high, cmap='gray')
    axs[1].set_title('High-Resolution')
    axs[1].axis('off')
    plt.show()

# =====================================
# 9. Define the Discriminator Model
# =====================================
discriminator = Sequential([
    Input(shape=(IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1)),  # High-resolution input (128x128)
    Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False),  # 128x128 -> 64x64
    LeakyReLU(0.2),

    Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False),  # 64x64 -> 32x32
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False),  # 32x32 -> 16x16
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False),  # 16x16 -> 8x8
    BatchNormalization(),
    LeakyReLU(0.2),

    Conv2D(1024, kernel_size=4, strides=2, padding='same', use_bias=False),  # 8x8 -> 4x4
    BatchNormalization(),
    LeakyReLU(0.2),

    Flatten(),
    Dense(1, use_bias=False)  # Output logits
])

discriminator.summary()

# =====================================
# 10. Plot the Discriminator Model
# =====================================
try:
    plot_model(
        model=discriminator,
        to_file='Discriminator_Model.png',
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True
    )
    print("Discriminator model plot saved successfully.")
except Exception as e:
    print(f"Failed to plot Discriminator model: {e}")
    print("Ensure that Graphviz is installed and added to your system PATH.")

# =====================================
# 11. Define the Generator Model
# =====================================
generator = Sequential([
    Dense(4*4*1024, use_bias=False, input_shape=(Z_DIM,)),
    BatchNormalization(),
    ReLU(),
    Reshape((4, 4, 1024)),

    Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False),  # 4x4 -> 8x8
    BatchNormalization(),
    ReLU(),

    Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False),  # 8x8 -> 16x16
    BatchNormalization(),
    ReLU(),

    Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),  # 16x16 -> 32x32
    BatchNormalization(),
    ReLU(),

    Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),  # 32x32 -> 64x64
    BatchNormalization(),
    ReLU(),

    Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh', use_bias=False),  # 64x64 -> 128x128
])

generator.summary()

# =====================================
# 12. Plot the Generator Model
# =====================================
try:
    plot_model(
        model=generator,
        to_file='Generator_Model.png',
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True
    )
    print("Generator model plot saved successfully.")
except Exception as e:
    print(f"Failed to plot Generator model: {e}")
    print("Ensure that Graphviz is installed and added to your system PATH.")

# =====================================
# 13. Define Optimizers and Loss Function
# =====================================
gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
disc_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
criterion = BinaryCrossentropy(from_logits=True)

# =====================================
# 14. Function to Display Generated Images
# =====================================
def show_images(epoch=None):
    noise = tf.random.normal([BATCH_SIZE, Z_DIM])
    preds = generator(noise, training=False)
    preds = (preds + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
    preds = np.clip(preds.numpy(), 0, 1)
    num_images_to_show = 1
    if num_images_to_show == 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        img = preds[0].squeeze(-1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    else:
        fig, ax = plt.subplots(1, num_images_to_show, figsize=(15, 10))
        for i in range(num_images_to_show):
            img = preds[i].squeeze(-1)
            ax[i].imshow(img, cmap='gray')
            ax[i].axis('off')
    if epoch:
        plt.savefig(f'gan_generated_images_epoch_{epoch}.png')
    else:
        plt.savefig('gan_generated_images.png')
    plt.show()

# =====================================
# 15. Define Loss Functions
# =====================================
def discriminator_loss(real_output, fake_output):
    real_labels = tf.ones_like(real_output) * 0.9  # Label smoothing
    fake_labels = tf.zeros_like(fake_output) + 0.1
    real_loss = criterion(real_labels, real_output)
    fake_loss = criterion(fake_labels, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return criterion(tf.ones_like(fake_output), fake_output)

# =====================================
# 16. Define Training Step
# =====================================
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, Z_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        
        # Add noise to real and fake images
        real_images_noisy = images + tf.random.normal(shape=images.shape, mean=0.0, stddev=0.1)
        fake_images_noisy = fake_images + tf.random.normal(shape=fake_images.shape, mean=0.0, stddev=0.1)
        
        real_output = discriminator(real_images_noisy, training=True)
        fake_output = discriminator(fake_images_noisy, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Clip gradients to prevent exploding gradients
    gen_gradients = [tf.clip_by_norm(g, 5.0) for g in gen_gradients]
    disc_gradients = [tf.clip_by_norm(g, 5.0) for g in disc_gradients]
    
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        
    return gen_loss, disc_loss

# =====================================
# 17. Super-Resolution Model Definition
# =====================================
# Adjusted to upscale from 64x64 to 128x128
def build_super_resolution_model():
    sr_model = Sequential([
        Input(shape=(IMG_SIZE_LOW, IMG_SIZE_LOW, 1)),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        UpSampling2D(size=(2,2)),  # 64x64 -> 128x128
        Conv2D(32, (3,3), padding='same', activation='relu'),
        Conv2D(1, (3,3), padding='same', activation='tanh')
    ])
    return sr_model

super_resolution_model = build_super_resolution_model()
super_resolution_model.compile(optimizer='adam', loss='mse')  # Use MSE loss for super-resolution
super_resolution_model.summary()

# =====================================
# 18. Training Super-Resolution Model
# =====================================
# Prepare paired dataset for super-resolution
def create_super_resolution_dataset(dataset_high, batch_size):
    dataset_sr = dataset_high.map(lambda x: (tf.image.resize(x, (IMG_SIZE_LOW, IMG_SIZE_LOW)), x))
    dataset_sr = dataset_sr.batch(batch_size)
    return dataset_sr

dataset_sr = create_super_resolution_dataset(dataset_high, BATCH_SIZE)
dataset_sr = dataset_sr.prefetch(buffer_size=tf.data.AUTOTUNE)

# Train the Super-Resolution Model
sr_epochs = 50  # Adjust based on performance
super_resolution_model.fit(dataset_sr,
                           epochs=sr_epochs,
                           steps_per_epoch=100)  # Adjust steps per epoch based on dataset size

# =====================================
# 19. Training Loop for GAN
# =====================================
disc_losses = [] 
gen_losses = [] 

for epoch in tqdm(range(EPOCHS), desc="Training Epochs"): 
    epoch_disc_loss = [] 
    epoch_gen_loss = []
    for image_low, image_high in tqdm(dataset_sr, desc=f"Epoch {epoch+1}", leave=False): 
        try:
            # Step 1: Train Discriminator
            gen_loss, disc_loss = train_step(image_high) 
            epoch_disc_loss.append(disc_loss.numpy())
            epoch_gen_loss.append(gen_loss.numpy())
        except tf.errors.ResourceExhaustedError as e:
            print(f"ResourceExhaustedError during training at epoch {epoch+1}: {e}")
            break  # Exit the inner loop and proceed to the next epoch
    # Append losses only if there were no errors
    if epoch_disc_loss and epoch_gen_loss:
        disc_losses.append(np.mean(epoch_disc_loss))
        gen_losses.append(np.mean(epoch_gen_loss))
        print(f"Epoch: {epoch+1}, Generator Loss: {gen_losses[-1]:.4f}, Discriminator Loss: {disc_losses[-1]:.4f}")
    else:
        print(f"Epoch: {epoch+1} skipped due to ResourceExhaustedError.")
    
    if (epoch + 1) % 10 == 0:
        show_images(epoch + 1)

# =====================================
# 20. Plot Losses
# =====================================
plt.figure(figsize=(10, 5))
plt.plot(disc_losses, label='Discriminator Loss')
plt.plot(gen_losses, label='Generator Loss')
plt.title('Discriminator and Generator Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# =====================================
# 21. Save Models
# =====================================
generator.save('Gan_Generator.h5')
discriminator.save('Gan_Discriminator.h5')
super_resolution_model.save('Super_Resolution_Model.h5')

print("Generator Input Shape:", generator.input_shape)
print("Super-Resolution Model Input Shape:", super_resolution_model.input_shape)

# =====================================
# 22. Image Generation Post-Training
# =====================================
batch_size_gen = 32               # Adjusted batch size
noise_dim_gen = Z_DIM             # Noise dimension matching Z_DIM
number_of_images_to_generate = 2000  # Number of images to generate
generated_images = []

for _ in tqdm(range(number_of_images_to_generate // batch_size_gen), desc="Generating Images"): 
    noise = tf.random.normal([batch_size_gen, noise_dim_gen])
    try:
        generated_image_low = generator(noise, training=False)  # Generate 128x128 images
        # Downscale to 64x64 for super-resolution
        generated_image_low_resized = tf.image.resize(generated_image_low, (IMG_SIZE_LOW, IMG_SIZE_LOW))
        # Upscale back to 128x128 using Super-Resolution Model
        generated_image_high = super_resolution_model(generated_image_low_resized, training=False)
        generated_images.append(generated_image_high)
    except tf.errors.ResourceExhaustedError as e:
        print(f"ResourceExhaustedError during image generation: {e}")
        break

# Concatenate Generated Images
if generated_images:
    generated_images = tf.concat(generated_images, axis=0)
    generated_images_np = generated_images.numpy()
    print(f"Generated Images Shape: {generated_images_np.shape}")  # Expected: (number_of_images_to_generate, 128, 128, 1)
else:
    print("No images were generated due to ResourceExhaustedError.")

# =====================================
# 23. Define Data Generator for CNN Training
# =====================================
def data_generator_cnn(batch_size, num_classes):
    while True:
        noise = tf.random.normal([batch_size, noise_dim_gen])
        generated_image_low = generator(noise, training=False)  # Generate 128x128 images
        # Downscale to 64x64 for super-resolution
        generated_image_low_resized = tf.image.resize(generated_image_low, (IMG_SIZE_LOW, IMG_SIZE_LOW))
        # Upscale back to 128x128 using Super-Resolution Model
        generated_image_high = super_resolution_model(generated_image_low_resized, training=False)
        generated_image_high_np = generated_image_high.numpy()
        train_labels = np.random.randint(0, num_classes, size=(batch_size,))
        yield generated_image_high_np, train_labels

# =====================================
# 24. Define and Train CNN Model
# =====================================
from tensorflow.keras import layers, models

# Number of classes (update as needed)
num_classes = 10

# Define CNN Architecture
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1)),  # Grayscale input
    layers.MaxPooling2D((2, 2)),  # 128 -> 64
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),  # 64 -> 32
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),  # 32 -> 16
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Adjust the number of classes as needed
])

cnn_model.summary()

# Compile the CNN Model
cnn_model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# Define steps per epoch
steps_per_epoch_cnn = number_of_images_to_generate // BATCH_SIZE

# Create data generator
train_data_gen_cnn = data_generator_cnn(batch_size=BATCH_SIZE, num_classes=num_classes)

# Dummy Validation Data
val_images = np.random.random((100, IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1)) * 2 - 1  # Grayscale, normalized to [-1, 1]
val_labels = np.random.randint(0, num_classes, size=(100,))  # Replace with actual validation labels

# Train the CNN Model
try:
    cnn_model.fit(train_data_gen_cnn,
                  steps_per_epoch=steps_per_epoch_cnn,
                  epochs=20,
                  validation_data=(val_images, val_labels))
except tf.errors.ResourceExhaustedError as e:
    print(f"ResourceExhaustedError during CNN training: {e}")

# Evaluate the CNN Model
test_images = np.random.random((100, IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1)) * 2 - 1  # Grayscale, normalized to [-1, 1]
test_labels = np.random.randint(0, num_classes, size=(100,))  # Replace with actual test labels

try:
    test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc:.4f}')
except tf.errors.ResourceExhaustedError as e:
    print(f"ResourceExhaustedError during CNN evaluation: {e}")

# Save the CNN Model
cnn_model.save('gan_model.h5')

# =====================================
# 25. Clean Up After All Operations
# =====================================
# Clear variables not needed to free up memory
del generated_images
del generator
del discriminator
del super_resolution_model
del cnn_model
tf.keras.backend.clear_session()
