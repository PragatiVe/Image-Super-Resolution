import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress TensorFlow warnings for cleaner output
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Define image properties
IMG_HEIGHT = 512
IMG_WIDTH = 512
CHANNELS = 1  # Grayscale; use 3 for RGB

# Path to your MRI dataset
DATA_DIR = 'E:/coding/Mlproject/archive(4)'  # Ensure all images are in this directory

# Parameters
LATENT_DIM = 128
BATCH_SIZE = 16  # Adjust based on your GPU memory
EPOCHS = 100  # Adjust based on your computational resources

# Create ImageDataGenerator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Create a generator for training
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode=None,  # For GANs, labels are not required
    shuffle=True
)

# Check if images are found
if train_generator.samples == 0:
    raise ValueError("No images found in the specified directory. Please check the DATA_DIR path and ensure images are in the 'images' subdirectory.")

# Convert generator to tf.data.Dataset
def generator_to_dataset(generator):
    while True:
        batch = generator.next()
        yield batch

train_ds = tf.data.Dataset.from_generator(
    lambda: generator_to_dataset(train_generator),
    output_types=tf.float32,
    output_shapes=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
)

# Cache, shuffle, and prefetch (remove .batch to avoid double batching)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(buffer_size=1000)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build Generator
def build_generator():
    model = Sequential()
    model.add(Dense(4 * 4 * 1024, input_dim=LATENT_DIM))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4, 4, 1024)))  # Starting from 4x4

    # Upsampling blocks
    model.add(UpSampling2D())  # 8x8
    model.add(Conv2D(512, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())  # 16x16
    model.add(Conv2D(256, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())  # 32x32
    model.add(Conv2D(128, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())  # 64x64
    model.add(Conv2D(64, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())  # 128x128
    model.add(Conv2D(32, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())  # 256x256
    model.add(Conv2D(16, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())  # 512x512
    model.add(Conv2D(CHANNELS, kernel_size=5, padding='same', activation='sigmoid'))

    return model

# Build Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=5, strides=2, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Instantiate models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers and Losses
g_opt = Adam(learning_rate=0.0002, beta_1=0.5)
d_opt = Adam(learning_rate=0.0002, beta_1=0.5)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

# Define GAN model
class MRI_GAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.metric_logs = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "d_loss": [],
            "g_loss": []
        }
        
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss
        
    def train_step(self, batch):
        real_images = batch
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal((batch_size, LATENT_DIM))

        # Generate fake images
        fake_images = self.generator(random_latent_vectors, training=False)

        # Labels for real and fake images
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # Train Discriminator
        with tf.GradientTape() as d_tape:
            real_predictions = self.discriminator(real_images, training=True)
            fake_predictions = self.discriminator(fake_images, training=True)
            real_loss = self.d_loss(real_labels, real_predictions)
            fake_loss = self.d_loss(fake_labels, fake_predictions)
            total_d_loss = (real_loss + fake_loss) / 2

        d_grad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        # Train Generator
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_predictions = self.discriminator(fake_images, training=False)
            total_g_loss = self.g_loss(misleading_labels, fake_predictions)

        g_grad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.generator.trainable_variables))
        
        # Metrics Calculation (Optional)
        # Convert TensorFlow tensors to numpy arrays for metrics calculation
        labels_np = tf.concat([real_labels, fake_labels], axis=0).numpy()
        pred_labels_np = tf.concat([real_predictions, fake_predictions], axis=0).numpy()
        pred_labels_np = np.round(pred_labels_np)

        # Calculate accuracy, precision, recall, and F1 score
        accuracy = accuracy_score(labels_np, pred_labels_np)
        precision = precision_score(labels_np, pred_labels_np, zero_division=0)
        recall = recall_score(labels_np, pred_labels_np, zero_division=0)
        f1 = f1_score(labels_np, pred_labels_np, zero_division=0)

        # Log the metrics
        self.metric_logs["accuracy"].append(accuracy)
        self.metric_logs["precision"].append(precision)
        self.metric_logs["recall"].append(recall)
        self.metric_logs["f1"].append(f1)
        self.metric_logs["d_loss"].append(total_d_loss.numpy())
        self.metric_logs["g_loss"].append(total_g_loss.numpy())
        
        return {
            "d_loss": total_d_loss,
            "g_loss": total_g_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

# Callback for saving images during training
class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=LATENT_DIM):
        self.num_img = num_img
        self.latent_dim = latent_dim
        os.makedirs('images', exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal((self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = tf.clip_by_value(generated_images, 0.0, 1.0)  # Ensure images are in [0,1]
        generated_images = (generated_images * 255).numpy().astype(np.uint8)
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_epoch{epoch+1}_{i+1}.png'))

# Function to plot the metrics in separate graphs
def plot_metrics(metric_logs):
    epochs = range(1, len(metric_logs["accuracy"]) + 1)
    
    plt.figure(figsize=(20, 15))

    # Plot Accuracy
    plt.subplot(3, 2, 1)
    plt.plot(epochs, metric_logs["accuracy"], label='Accuracy', color='blue')
    plt.title('Discriminator Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Precision
    plt.subplot(3, 2, 2)
    plt.plot(epochs, metric_logs["precision"], label='Precision', color='green')
    plt.title('Discriminator Precision over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(3, 2, 3)
    plt.plot(epochs, metric_logs["recall"], label='Recall', color='red')
    plt.title('Discriminator Recall over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Plot F1 Score
    plt.subplot(3, 2, 4)
    plt.plot(epochs, metric_logs["f1"], label='F1 Score', color='purple')
    plt.title('Discriminator F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # Plot Discriminator Loss
    plt.subplot(3, 2, 5)
    plt.plot(epochs, metric_logs["d_loss"], label='Discriminator Loss', color='orange')
    plt.title('Discriminator Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Generator Loss
    plt.subplot(3, 2, 6)
    plt.plot(epochs, metric_logs["g_loss"], label='Generator Loss', color='cyan')
    plt.title('Generator Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Instantiate and compile the GAN
mri_gan = MRI_GAN(generator, discriminator)
mri_gan.compile(g_opt, d_opt, g_loss, d_loss)

# Instantiate the callback
model_monitor = ModelMonitor(num_img=5, latent_dim=LATENT_DIM)

# Train the GAN
history = mri_gan.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[model_monitor]
)

# Plot the metrics after training
plot_metrics(mri_gan.metric_logs)

# Generate and visualize images after training
def generate_and_save_images(model, epoch, latent_dim=LATENT_DIM, num_images=5):
    random_latent_vectors = tf.random.normal((num_images, latent_dim))
    generated_images = model.generator(random_latent_vectors)
    generated_images = tf.clip_by_value(generated_images, 0.0, 1.0)
    generated_images = (generated_images * 255).numpy().astype(np.uint8)

    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Generated Images at Epoch {epoch}')
    plt.show()

# Example usage after training
generate_and_save_images(mri_gan, EPOCHS)
