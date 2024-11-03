# =====================================
# 0. Import Libraries and Enable Mixed Precision Training
# =====================================
import warnings
warnings.filterwarnings('ignore')

import os
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Reshape, Conv2D, Conv2DTranspose, Dense,
    BatchNormalization, LeakyReLU, ReLU, Flatten, Add,
    Activation, MaxPooling2D, Concatenate, GlobalAveragePooling2D,
    UpSampling2D  # Ensure UpSampling2D is imported
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Import TensorFlow Addons
import tensorflow_addons as tfa

# Import Mixed Precision
from tensorflow.keras import mixed_precision

# Enable Mixed Precision Training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# =====================================
# 1. GPU Configuration
# =====================================
# Enable GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for GPU: {gpu.name}")
    except RuntimeError as e:
        print(e)

print("Is GPU available:", tf.config.list_physical_devices('GPU'))

# =====================================
# 2. Parameters
# =====================================
IMG_SIZE_LOW = 64         # Low-resolution Image size (64x64)
IMG_SIZE_HIGH = 128       # High-resolution Image size (128x128)
BATCH_SIZE = 8            # Further reduced batch size to fit system memory
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

# Shuffle the data
np.random.shuffle(img_paths_labels)

# Split image paths and labels
if img_paths_labels:
    img_paths, labels = zip(*img_paths_labels)
    img_paths = list(img_paths)
    labels = list(labels)
else:
    img_paths = []
    labels = []

# =====================================
# 5. Define Mapping Function for Paired Low and High-Resolution Images
# =====================================
def map_fn_paired(img_path, label):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=1)  # Grayscale
        img_high = tf.image.resize(img, (IMG_SIZE_HIGH, IMG_SIZE_HIGH))
        img_low = tf.image.resize(img_high, (IMG_SIZE_LOW, IMG_SIZE_LOW))
        img_high = tf.cast(img_high, tf.float16) / 127.5 - 1    # Normalize to [-1, 1] and cast to float16
        img_low = tf.cast(img_low, tf.float16) / 127.5 - 1      # Normalize to [-1, 1] and cast to float16
        label = tf.one_hot(label, depth=num_classes)             # One-hot encode the label
        return img_low, img_high, label
    except tf.errors.InvalidArgumentError:
        # Handle corrupted images by returning black images
        img_high = tf.zeros((IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1), dtype=tf.float16)
        img_low = tf.zeros((IMG_SIZE_LOW, IMG_SIZE_LOW, 1), dtype=tf.float16)
        label = tf.one_hot(label, depth=num_classes)
        return img_low, img_high, label

# =====================================
# 6. Create TensorFlow Datasets with Paired Low and High-Resolution Images
# =====================================
if img_paths and labels:
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
    train_dataset_paired = train_dataset.map(map_fn_paired, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset_paired = val_dataset.map(map_fn_paired, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset_paired = test_dataset.map(map_fn_paired, num_parallel_calls=tf.data.AUTOTUNE)
    
    print("Datasets created and split into training, validation, and test sets.")
else:
    print("No images found. Please check your dataset directories.")

# =====================================
# 7. Fetch a Single Image Sample
# =====================================
if img_paths and labels:
    try:
        img_sample_low, img_sample_high, label_sample = next(iter(train_dataset_paired.batch(1)))
        print(f"Low-Resolution Image sample shape: {img_sample_low.shape}")    # Expected: (1, 64, 64, 1)
        print(f"High-Resolution Image sample shape: {img_sample_high.shape}")  # Expected: (1, 128, 128, 1)
    except tf.errors.ResourceExhaustedError as e:
        print("Out of memory error encountered while fetching a single image sample:", e)
    except StopIteration:
        print("The dataset is empty.")
else:
    print("No samples available to fetch.")

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
    img_high = (img_high + 1) / 2.0  # Rescale from [-1, 1] to [0, 1] for display
    axs[1].imshow(img_high, cmap='gray')
    axs[1].set_title('High-Resolution')
    axs[1].axis('off')
    plt.show()

# =====================================
# 9. Define Residual Block Function
# =====================================
def residual_block(x, filters, kernel_size=3, strides=1, activation='relu'):
    """Defines a residual block with two convolutional layers and optional projection."""
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # If the number of filters has changed or strides !=1, apply a 1x1 convolution to the shortcut
    if shortcut.shape[-1] != filters or strides !=1:
        shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([shortcut, x])
    x = Activation(activation)(x)
    return x

# =====================================
# 10. Define Inception Module Function
# =====================================
def inception_module(x, filters):
    """
    Defines an Inception module with multiple convolutional paths.
    Args:
        x: Input tensor.
        filters: Number of filters for each convolution.
    Returns:
        Concatenated output of the Inception module.
    """
    # 1x1 convolution branch
    branch1 = Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    
    # 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    branch2 = Conv2D(filters, (3,3), padding='same', activation='relu')(branch2)
    
    # 1x1 convolution followed by 5x5 convolution
    branch3 = Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    branch3 = Conv2D(filters, (5,5), padding='same', activation='relu')(branch3)
    
    # 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    branch4 = Conv2D(filters, (1,1), padding='same', activation='relu')(branch4)
    
    # Concatenate all branches
    output = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    return output

# =====================================
# 11. Define the Generator Model with Residual Blocks
# =====================================
def build_generator_with_residual():
    # Inputs
    inputs_low = Input(shape=(IMG_SIZE_LOW, IMG_SIZE_LOW, 1), name='low_res_input')  # Low-resolution input
    noise = Input(shape=(Z_DIM,), name='noise_input')  # Noise vector
    
    # Process Noise
    x = Dense(4*4*512, use_bias=False)(noise)  # Dense layer to project noise
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((4, 4, 512))(x)  # Reshape to 4x4x512
    
    # Upsampling Steps to Match Spatial Dimensions (4x4 -> 64x64)
    # Each Conv2DTranspose doubles the spatial dimensions
    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)  # 4x4 -> 8x8
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)  # 8x8 -> 16x16
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)  # 16x16 -> 32x32
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False)(x)  # 32x32 -> 64x64
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Now, x has shape (64, 64, 32)
    
    # Concatenate with Low-Resolution Input
    # Ensure both tensors have the same spatial dimensions
    # Here, x: 64x64x32 and inputs_low: 64x64x1
    x = Concatenate(axis=-1)([x, inputs_low])  # Resulting shape: (64, 64, 33)
    
    # Add Residual Blocks to Enhance Feature Learning
    x = residual_block(x, filters=64)  # Residual Block with 64 filters
    x = residual_block(x, filters=64)  # Another Residual Block
    
    # Final Upsampling to Reach 128x128
    x = Conv2DTranspose(16, kernel_size=4, strides=2, padding='same', use_bias=False)(x)  # 64x64 -> 128x128
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Final Output Layer
    # Use 'tanh' activation and cast to 'float32' for numerical stability
    x = Conv2D(1, kernel_size=3, padding='same', activation='tanh', dtype='float32')(x)  # 128x128x1
    
    # Define the Model
    generator_model = Model([inputs_low, noise], x, name='Generator_with_Residual')
    return generator_model

# Instantiate and summarize the Generator
generator = build_generator_with_residual()
generator.summary()

# =====================================
# 12. Define the Discriminator Model with Residual Blocks and Spectral Normalization
# =====================================
def build_discriminator_with_residual():
    inputs_high = Input(shape=(IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1), name='high_res_input')  # High-resolution input
    
    # Apply Spectral Normalization to stabilize training
    x = tfa.layers.SpectralNormalization(Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False))(inputs_high)  # 128x128 -> 64x64
    x = LeakyReLU(0.2)(x)

    # First Residual Block
    x = residual_block(x, 128, strides=2)  # 64x64 -> 32x32
    x = tfa.layers.SpectralNormalization(Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Second Residual Block
    x = residual_block(x, 256, strides=2)  # 32x32 -> 16x16
    x = tfa.layers.SpectralNormalization(Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Third Residual Block
    x = residual_block(x, 512, strides=2)  # 16x16 -> 8x8
    x = tfa.layers.SpectralNormalization(Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Flatten and Dense Layers
    x = Flatten()(x)
    # Ensure the final Dense layer is float32 for numerical stability in mixed precision
    x = Dense(1, use_bias=False, dtype='float32')(x)  # Output logits

    discriminator_model = Model(inputs_high, x, name='Discriminator_with_Residual')
    return discriminator_model

# Instantiate and summarize the Discriminator
discriminator = build_discriminator_with_residual()
discriminator.summary()

# =====================================
# 13. Plot the Discriminator and Generator Models
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
# 14. Define Optimizers and Loss Functions
# =====================================
gen_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
disc_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
criterion = BinaryCrossentropy(from_logits=True)

# =====================================
# 15. Define Perceptual Loss Using Pre-trained VGG19
# =====================================
# Note: Since the images are grayscale, we need to convert them to RGB by duplicating channels
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(IMG_SIZE_HIGH, IMG_SIZE_HIGH, 3))
vgg.trainable = False

def perceptual_loss(y_true, y_pred):
    # Convert grayscale to RGB by duplicating channels
    y_true_rgb = tf.image.grayscale_to_rgb((y_true + 1) / 2.0)
    y_pred_rgb = tf.image.grayscale_to_rgb((y_pred + 1) / 2.0)
    # Preprocess for VGG19
    y_true_pre = tf.keras.applications.vgg19.preprocess_input(y_true_rgb * 255.0)
    y_pred_pre = tf.keras.applications.vgg19.preprocess_input(y_pred_rgb * 255.0)
    # Extract features
    y_true_features = vgg(y_true_pre)
    y_pred_features = vgg(y_pred_pre)
    # Compute MSE between features
    return tf.reduce_mean(tf.square(y_true_features - y_pred_features))

# =====================================
# 16. Define Combined Generator Loss (Adversarial + Content + Perceptual)
# =====================================
def combined_generator_loss(fake_output, generated_images, real_images):
    adversarial = generator_loss(fake_output)
    # Cast real_images to float32 to match generated_images
    content = tf.reduce_mean(tf.abs(tf.cast(real_images, tf.float32) - generated_images))  # L1 loss with type casting
    perceptual = perceptual_loss(real_images, generated_images)
    # Cast all components to float32 to ensure type consistency
    adversarial = tf.cast(adversarial, tf.float32)
    perceptual = tf.cast(perceptual, tf.float32)
    return adversarial + 100 * content + 10 * perceptual  # Weighted sum of losses

# =====================================
# 17. Define Generator and Discriminator Loss Functions
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
# 18. Define Training Step for cGAN with Combined Loss
# =====================================
@tf.function
def train_step_cgan(low_res, high_res):
    batch_size = tf.shape(low_res)[0]
    noise = tf.random.normal([batch_size, Z_DIM], dtype=tf.float16)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_high_res = generator([low_res, noise], training=True)
        
        # Discriminator predictions
        real_output = discriminator(high_res, training=True)
        fake_output = discriminator(generated_high_res, training=True)
        
        # Compute losses
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = combined_generator_loss(fake_output, generated_high_res, high_res)
    
    # Compute gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients with mixed precision optimizers
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# =====================================
# 19. Define Function to Display Generated Images
# =====================================
def show_images_with_low_res(epoch=None):
    noise = tf.random.normal([BATCH_SIZE, Z_DIM], dtype=tf.float16)  # Ensure noise is float16
    # Select random low-res images from validation set
    low_res_batch, high_res_batch, _ = next(iter(val_dataset_paired.batch(BATCH_SIZE)))
    generated_high_res = generator([low_res_batch, noise], training=False)
    
    # Rescale images for display
    generated_high_res = (generated_high_res.numpy().squeeze() + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
    low_res = (low_res_batch.numpy().squeeze(-1) + 1) / 2.0  # Rescale to [0,1]
    high_res_real = (high_res_batch.numpy().squeeze(-1) + 1) / 2.0  # Rescale to [0,1]
    
    num_images_to_show = 3  # Adjust the number of images to display
    fig, axs = plt.subplots(num_images_to_show, 3, figsize=(15, 5 * num_images_to_show))
    for i in range(num_images_to_show):
        # Low-Resolution Input
        axs[i, 0].imshow(low_res[i], cmap='gray')
        axs[i, 0].set_title('Low-Resolution Input')
        axs[i, 0].axis('off')

        # Generated High-Resolution Image
        axs[i, 1].imshow(generated_high_res[i], cmap='gray')
        axs[i, 1].set_title('Generated High-Res')
        axs[i, 1].axis('off')

        # Real High-Resolution Image
        axs[i, 2].imshow(high_res_real[i], cmap='gray')
        axs[i, 2].set_title('Real High-Res')
        axs[i, 2].axis('off')
    if epoch:
        plt.savefig(f'gan_generated_images_epoch_{epoch}.png')
    else:
        plt.savefig('gan_generated_images.png')
    plt.show()

# =====================================
# 20. Define Super-Resolution Model (Optional)
# =====================================
def build_super_resolution_model():
    sr_model = Sequential([
        Input(shape=(IMG_SIZE_LOW, IMG_SIZE_LOW, 1)),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        UpSampling2D(size=(2,2)),  # 64x64 -> 128x128
        Conv2D(16, (3,3), padding='same', activation='relu'),
        Conv2D(1, (3,3), padding='same', activation='tanh', dtype='float32')  # Ensure output is float32
    ])
    return sr_model

super_resolution_model = build_super_resolution_model()
super_resolution_model.compile(optimizer='adam', loss='mse')  # Use MSE loss for super-resolution
super_resolution_model.summary()

# =====================================
# 21. Training Super-Resolution Model (Optional)
# =====================================
if img_paths and labels:
    # Prepare paired dataset for super-resolution
    def create_super_resolution_dataset(dataset_paired, batch_size):
        dataset_sr = dataset_paired.map(lambda low, high, label: (tf.image.resize(low, (IMG_SIZE_LOW, IMG_SIZE_LOW)), high))
        dataset_sr = dataset_sr.batch(batch_size, drop_remainder=True)
        dataset_sr = dataset_sr.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset_sr
    
    dataset_sr = create_super_resolution_dataset(train_dataset_paired, BATCH_SIZE)
    steps_per_epoch_sr = 100  # Adjust steps per epoch based on dataset size
    
    # Train the Super-Resolution Model
    sr_epochs = 50  # Adjust based on performance
    super_resolution_model.fit(dataset_sr,
                               epochs=sr_epochs,
                               steps_per_epoch=steps_per_epoch_sr)
else:
    print("Super-Resolution Model training skipped due to lack of data.")

# =====================================
# 22. Training Loop for Conditional GAN
# =====================================
if img_paths and labels:
    disc_losses = [] 
    gen_losses = [] 
    
    for epoch in tqdm(range(EPOCHS), desc="Training Epochs"): 
        epoch_disc_loss = [] 
        epoch_gen_loss = []
        # Use pre-fetched batches to prevent iterator exhaustion
        train_paired_batch = train_dataset_paired.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
        
        for low_res, high_res, _ in tqdm(train_paired_batch, desc=f"Epoch {epoch+1}", leave=False): 
            try:
                # Step: Train cGAN
                gen_loss, disc_loss = train_step_cgan(low_res, high_res) 
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
            if (epoch + 1) % 10 == 0:
                show_images_with_low_res(epoch + 1)
        else:
            print(f"Epoch: {epoch+1} skipped due to ResourceExhaustedError.")
else:
    print("GAN training skipped due to lack of data.")

# =====================================
# 23. Plot Losses Over Epochs
# =====================================
if disc_losses and gen_losses:
    plt.figure(figsize=(10, 5))
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.plot(gen_losses, label='Generator Loss')
    plt.title('Discriminator and Generator Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
else:
    print("No losses to plot.")

# =====================================
# 24. Save Models
# =====================================
if img_paths and labels:
    generator.save('Gan_Generator.h5')
    discriminator.save('Discriminator.h5')
    super_resolution_model.save('Super_Resolution_Model.h5')
    
    print("Generator Input Shape:", generator.input_shape)
    print("Super-Resolution Model Input Shape:", super_resolution_model.input_shape)
else:
    print("Models not saved due to lack of data.")

# =====================================
# 25. Image Generation Post-Training
# =====================================
if img_paths and labels:
    batch_size_gen = 8               # Further reduced batch size
    noise_dim_gen = Z_DIM             # Noise dimension matching Z_DIM
    number_of_images_to_generate = 2000  # Number of images to generate
    generated_images = []
    
    # Prepare generator for inference
    generator_inference = generator
    generator_inference.trainable = False
    
    # Generate images in batches
    for _ in tqdm(range(number_of_images_to_generate // batch_size_gen), desc="Generating Images"): 
        # Sample random low-res images from validation set
        low_res_batch, high_res_batch, _ = next(iter(val_dataset_paired.batch(batch_size_gen)))
        noise = tf.random.normal([batch_size_gen, noise_dim_gen], dtype=tf.float16)
        try:
            generated_high_res = generator_inference([low_res_batch, noise], training=False)
            generated_images.append(generated_high_res)
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
else:
    print("Image generation skipped due to lack of data.")

# =====================================
# 26. Define Data Generator for CNN Training
# =====================================
def data_generator_cnn(dataset, batch_size):
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

if img_paths and labels:
    # Training data generator
    train_data_gen_cnn = data_generator_cnn(train_dataset_paired, batch_size=BATCH_SIZE)
    
    # Validation data generator
    val_data_gen_cnn = data_generator_cnn(val_dataset_paired, batch_size=BATCH_SIZE)
else:
    print("CNN data generators not created due to lack of data.")

# =====================================
# 27. Define and Train CNN Model with Residual Blocks and Inception Modules
# =====================================
def build_cnn_with_inception():
    inputs = Input(shape=(IMG_SIZE_HIGH, IMG_SIZE_HIGH, 1))
    
    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)  # 128 -> 64
    
    # Inception Module 1
    x = inception_module(x, filters=32)  # Output channels: 128
    x = MaxPooling2D((2, 2))(x)  # 64 -> 32
    
    # Residual Block 1
    x = residual_block(x, filters=128)  # Ensures both paths have 128 channels
    
    # Inception Module 2
    x = inception_module(x, filters=64)  # Output channels: 256
    x = MaxPooling2D((2, 2))(x)  # 32 -> 16
    
    # Residual Block 2
    x = residual_block(x, filters=256)  # Ensures both paths have 256 channels
    
    # Inception Module 3
    x = inception_module(x, filters=128)  # Output channels: 512
    x = MaxPooling2D((2, 2))(x)  # 16 -> 8
    
    # Flatten and Dense Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    cnn_model = Model(inputs, x, name='CNN_with_Inception')
    return cnn_model

if img_paths and labels:
    cnn_model = build_cnn_with_inception()
    cnn_model.summary()
else:
    cnn_model = None
    print("CNN model not built due to lack of data.")

# =====================================
# 28. Compile the CNN Model
# =====================================
if cnn_model:
    cnn_model.compile(optimizer='adam', 
                      loss='categorical_crossentropy',  # Using categorical_crossentropy for one-hot labels
                      metrics=['accuracy', 
                               tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='f1_score')])
else:
    print("CNN model compilation skipped due to lack of data.")

# =====================================
# 29. Add Callbacks for Early Stopping and Checkpoints
# =====================================
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

if cnn_model:
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_cnn_model.h5', monitor='val_loss', save_best_only=True)
else:
    early_stopping = None
    checkpoint = None

# =====================================
# 30. Train the CNN Model with Callbacks
# =====================================
if cnn_model and img_paths and labels:
    steps_per_epoch_cnn = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE
    
    history = cnn_model.fit(
        train_data_gen_cnn,
        steps_per_epoch=steps_per_epoch_cnn,
        epochs=20,
        validation_data=val_data_gen_cnn,
        validation_steps=validation_steps,
        callbacks=[early_stopping, checkpoint] if early_stopping and checkpoint else None
    )
else:
    print("CNN model training skipped due to lack of data.")

# =====================================
# 31. Plot Metrics Over Epochs
# =====================================
if cnn_model and 'history' in locals():
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
else:
    print("No metrics to plot.")

# =====================================
# 32. Evaluate the CNN Model
# =====================================
if cnn_model and img_paths and labels:
    # Prepare test dataset
    test_data_gen_cnn = data_generator_cnn(test_dataset_paired, batch_size=BATCH_SIZE)
    test_steps = test_size // BATCH_SIZE
    
    # Evaluate the CNN Model
    test_loss, test_acc, test_f1 = cnn_model.evaluate(test_data_gen_cnn, steps=test_steps)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    
    # Save the CNN Model (best model already saved via checkpoint)
    cnn_model.save('gan_cnn_model.h5')
    print("CNN model saved as 'gan_cnn_model.h5'.")
else:
    print("CNN model evaluation skipped due to lack of data.")

# =====================================
# 33. Compute Evaluation Metrics (MSE, PSNR, SSIM)
# =====================================
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
# Note: VIF is not directly available in skimage. Consider using other libraries or omit it.

def evaluate_generated_images(generator, dataset_paired, num_batches=100):
    mse_values = []
    psnr_values = []
    ssim_values = []
    
    for low_res, high_res, _ in tqdm(dataset_paired.batch(1).take(num_batches), desc="Evaluating Images"):
        noise = tf.random.normal([1, Z_DIM], dtype=tf.float16)
        generated_high_res = generator([low_res, noise], training=False)
        
        # Rescale images from [-1,1] to [0,1]
        generated_image = (generated_high_res.numpy().squeeze() + 1) / 2.0
        real_image = (high_res.numpy().squeeze() + 1) / 2.0
        
        # Compute metrics
        mse = compare_mse(real_image, generated_image)
        psnr = compare_psnr(real_image, generated_image, data_range=1.0)
        ssim = compare_ssim(real_image, generated_image, data_range=1.0)
        
        mse_values.append(mse)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    
    print(f"Average MSE: {np.mean(mse_values):.4f}")
    print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_values):.4f}")

if img_paths and labels:
    evaluate_generated_images(generator, test_dataset_paired)
else:
    print("Image evaluation skipped due to lack of data.")

# =====================================
# 34. Clean Up After All Operations
# =====================================
# Clear variables not needed to free up memory
if img_paths and labels:
    del generated_images 
    del generator
    del discriminator
    del super_resolution_model
    del cnn_model
    tf.keras.backend.clear_session()
    print("Cleaned up resources and cleared TensorFlow session.")
else:
    print("No resources to clean up.")
