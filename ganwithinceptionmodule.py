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
                                     BatchNormalization, LeakyReLU, ReLU, Flatten, UpSampling2D, MaxPooling2D, Concatenate, Dropout)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

# Import TensorFlow Addons
import tensorflow_addons as tfa

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

# Number of classes (update as needed)
num_classes = 4  # Adjusted based on your labels

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
# 4. Collect Image Paths and Labels
# =====================================
img_paths_labels = []
label_mapping = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

for dataset_dir in tqdm(DATASET_DIRS, desc="Dataset Directories"):
    if not os.path.isdir(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        continue
    label_name = os.path.basename(dataset_dir).lower()
    label = label_mapping.get(label_name)
    if label is None:
        print(f"Label not found for directory: {dataset_dir}")
        continue
    for filename in tqdm(os.listdir(dataset_dir), desc=f"Loading {dataset_dir}", leave=False):
        img_path = os.path.join(dataset_dir, filename)
        if os.path.isfile(img_path) and img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_paths_labels.append((img_path, label))

print(f"Total images found: {len(img_paths_labels)}")

# Split image paths and labels
img_paths, labels = zip(*img_paths_labels)
img_paths = list(img_paths)
labels = list(labels)

# =====================================
# 5. Define Mapping Functions for Low and High-Resolution Images
# =====================================
def map_fn_low(img_path, label):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=1)  # Grayscale
        img = tf.image.resize(img, (IMG_SIZE_LOW, IMG_SIZE_LOW))
        img = tf.cast(img, tf.float32) / 127.5 - 1    # Normalize to [-1, 1]
        label = tf.one_hot(label, depth=num_classes)   # One-hot encode the label
        return img, label
    except tf.errors.InvalidArgumentError:
        # Handle corrupted images by returning a black image
        img = tf.zeros((IMG_SIZE_LOW, IMG_SIZE_LOW, 1), dtype=tf.float32)
        label = tf.one_hot(label, depth=num_classes)
        return img, label

def map_fn_high(img_path, label):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=1)  # Grayscale
        img = tf.image.resize(img, (IMG_SIZE_HIGH, IMG_SIZE_HIGH))
        img = tf.cast(img, tf.float32) / 127.5 - 1    # Normalize to [-1, 1]
        label = tf.one_hot(label, depth=num_classes)   # One-hot encode the label
        return img, label
    except tf.errors.InvalidArgumentError:
        # Handle corrupted images by returning a black image
        img = tf.zeros((IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1), dtype=tf.float32)
        label = tf.one_hot(label, depth=num_classes)
        return img, label

# =====================================
# 6. Create TensorFlow Datasets with Labels
# =====================================
dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

# Shuffle the dataset
dataset = dataset.shuffle(len(img_paths), reshuffle_each_iteration=False)

# Split sizes
train_size = int(0.7 * len(img_paths))
val_size = int(0.15 * len(img_paths))
test_size = len(img_paths) - train_size - val_size

# Split the dataset
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

# Map the datasets
train_dataset_low = train_dataset.map(map_fn_low, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset_high = train_dataset.map(map_fn_high, num_parallel_calls=tf.data.AUTOTUNE)

val_dataset_low = val_dataset.map(map_fn_low, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset_high = val_dataset.map(map_fn_high, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset_low = test_dataset.map(map_fn_low, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset_high = test_dataset.map(map_fn_high, num_parallel_calls=tf.data.AUTOTUNE)

print("Datasets created and split into training, validation, and test sets.")

# =====================================
# 7. Fetch a Single Image Sample
# =====================================
try:
    img_sample_low, label_sample_low = next(iter(train_dataset_low.batch(1)))
    img_sample_high, label_sample_high = next(iter(train_dataset_high.batch(1)))
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
# 14. Function to Display Generated Images with Low-Resolution Inputs
# =====================================
def show_images_with_low_res(epoch=None):
    noise = tf.random.normal([BATCH_SIZE, Z_DIM])
    preds_high_res = generator(noise, training=False)

    # Generate low-resolution images
    preds_low_res = tf.image.resize(preds_high_res, (IMG_SIZE_LOW, IMG_SIZE_LOW))

    # Upscale low-resolution images using Super-Resolution Model
    preds_sr = super_resolution_model(preds_low_res, training=False)

    preds_high_res = (preds_high_res + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
    preds_sr = (preds_sr + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
    preds_low_res = (preds_low_res + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]

    num_images_to_show = 3  # Adjust the number of images to display
    fig, axs = plt.subplots(num_images_to_show, 3, figsize=(15, 5 * num_images_to_show))
    for i in range(num_images_to_show):
        # High-Resolution Generated Image
        img_high = preds_high_res[i].numpy().squeeze(-1)
        axs[i, 0].imshow(img_high, cmap='gray')
        axs[i, 0].set_title('Generated High-Res')
        axs[i, 0].axis('off')

        # Low-Resolution Image
        img_low = preds_low_res[i].numpy().squeeze(-1)
        axs[i, 1].imshow(img_low, cmap='gray')
        axs[i, 1].set_title('Generated Low-Res')
        axs[i, 1].axis('off')

        # Super-Resolved Image
        img_sr = preds_sr[i].numpy().squeeze(-1)
        axs[i, 2].imshow(img_sr, cmap='gray')
        axs[i, 2].set_title('Super-Resolved Image')
        axs[i, 2].axis('off')
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
    dataset_sr = dataset_high.map(lambda x, y: (tf.image.resize(x, (IMG_SIZE_LOW, IMG_SIZE_LOW)), x))
    dataset_sr = dataset_sr.batch(batch_size)
    return dataset_sr

dataset_sr = create_super_resolution_dataset(train_dataset_high, BATCH_SIZE)
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

# Calculate the number of steps per epoch
steps_per_epoch = train_size // BATCH_SIZE

for epoch in tqdm(range(EPOCHS), desc="Training Epochs"): 
    epoch_disc_loss = [] 
    epoch_gen_loss = []
    for step, ((image_high, _), (image_low, _)) in enumerate(zip(train_dataset_high.batch(BATCH_SIZE), train_dataset_low.batch(BATCH_SIZE))):
        if step >= steps_per_epoch:
            break
        try:
            # Step 1: Train Discriminator and Generator
            gen_loss_val, disc_loss_val = train_step(image_high) 
            epoch_disc_loss.append(disc_loss_val.numpy())
            epoch_gen_loss.append(gen_loss_val.numpy())
        except tf.errors.ResourceExhaustedError as e:
            print(f"ResourceExhaustedError during training at epoch {epoch+1}, step {step+1}: {e}")
            break  # Exit the inner loop and proceed to the next epoch
    # Append losses only if there were no errors
    if epoch_disc_loss and epoch_gen_loss:
        disc_losses.append(np.mean(epoch_disc_loss))
        gen_losses.append(np.mean(epoch_gen_loss))
        print(f"Epoch: {epoch+1}, Generator Loss: {gen_losses[-1]:.4f}, Discriminator Loss: {disc_losses[-1]:.4f}")
    else:
        print(f"Epoch: {epoch+1} skipped due to ResourceExhaustedError.")

    if (epoch + 1) % 10 == 0:
        show_images_with_low_res(epoch + 1)

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
        generated_image_high = generator(noise, training=False)  # Generate 128x128 images
        # Downscale to 64x64 for super-resolution
        generated_image_low_resized = tf.image.resize(generated_image_high, (IMG_SIZE_LOW, IMG_SIZE_LOW))
        # Upscale back to 128x128 using Super-Resolution Model
        generated_image_sr = super_resolution_model(generated_image_low_resized, training=False)
        generated_images.append(generated_image_sr)
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
def data_generator_cnn(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Training data generator
train_data_gen_cnn = data_generator_cnn(train_dataset_high, batch_size=BATCH_SIZE)

# Validation data generator
val_data_gen_cnn = data_generator_cnn(val_dataset_high, batch_size=BATCH_SIZE)

# =====================================
# 24. Define and Train CNN Model with Inception Module and Metrics
# =====================================
from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout

# Define Inception Module
def inception_module(x, filters):
    # 1x1 convolution branch
    branch1x1 = Conv2D(filters['branch1x1'], (1,1), padding='same', activation='relu')(x)
    
    # 3x3 convolution branch
    branch3x3 = Conv2D(filters['branch3x3_reduce'], (1,1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(filters['branch3x3'], (3,3), padding='same', activation='relu')(branch3x3)
    
    # 5x5 convolution branch
    branch5x5 = Conv2D(filters['branch5x5_reduce'], (1,1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters['branch5x5'], (5,5), padding='same', activation='relu')(branch5x5)
    
    # 3x3 max pooling branch
    branch_pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    branch_pool = Conv2D(filters['branch_pool'], (1,1), padding='same', activation='relu')(branch_pool)
    
    # Concatenate all branches
    output = Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool])
    return output

# Input layer
input_img = Input(shape=(IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1))

# Initial convolution and max pooling
x = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(input_img)
x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

# First Inception module
filters = {
    'branch1x1': 64,
    'branch3x3_reduce': 96,
    'branch3x3': 128,
    'branch5x5_reduce': 16,
    'branch5x5': 32,
    'branch_pool': 32
}
x = inception_module(x, filters)

# Second Inception module
filters = {
    'branch1x1': 128,
    'branch3x3_reduce': 128,
    'branch3x3': 192,
    'branch5x5_reduce': 32,
    'branch5x5': 96,
    'branch_pool': 64
}
x = inception_module(x, filters)

# Max pooling
x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

# Flatten and dense layers
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

# Create the model
cnn_model = Model(inputs=input_img, outputs=output)

cnn_model.summary()

# Compile the CNN Model
cnn_model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',  # Using categorical_crossentropy for one-hot labels
                  metrics=['accuracy', 
                           tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='f1_score')])

# Define steps per epoch
steps_per_epoch_cnn = train_size // BATCH_SIZE

# Validation steps
validation_steps = val_size // BATCH_SIZE

# Train the CNN Model
history = cnn_model.fit(train_data_gen_cnn,
                        steps_per_epoch=steps_per_epoch_cnn,
                        epochs=20,
                        validation_data=val_data_gen_cnn,
                        validation_steps=validation_steps)

# =====================================
# 25. Plot Metrics Over Epochs
# =====================================
epochs_range = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot F1 Score
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['f1_score'], label='Training F1 Score')
plt.plot(epochs_range, history.history['val_f1_score'], label='Validation F1 Score')
plt.legend(loc='lower right')
plt.title('Training and Validation F1 Score')

plt.tight_layout()
plt.show()

# =====================================
# 26. Evaluate the CNN Model
# =====================================
# Prepare test dataset
test_data_gen_cnn = data_generator_cnn(test_dataset_high, batch_size=BATCH_SIZE)
test_steps = test_size // BATCH_SIZE

# Evaluate the CNN Model
test_loss, test_acc, test_f1 = cnn_model.evaluate(test_data_gen_cnn, steps=test_steps)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')

# Save the CNN Model
cnn_model.save('gan_cnn_model_with_inception.h5')

# =====================================
# 27. Clean Up After All Operations
# =====================================
# Clear variables not needed to free up memory
del generated_images
del generator
del discriminator
del super_resolution_model
del cnn_model
tf.keras.backend.clear_session()
