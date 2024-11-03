import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras import mixed_precision
import gc
import matplotlib.pyplot as plt

# Set mixed precision policy (optional, remove if causing issues)
mixed_precision.set_global_policy('mixed_float16')

# Import TensorFlow Addons for F1Score metric
import tensorflow_addons as tfa

# Paths to datasets
train_dir = 'E:/coding/Mlproject/HighResolutionResidual/Training'
test_dir = 'E:/coding/Mlproject/HighResolutionResidual/Testing'

# Define parameters
batch_size = 16
img_size = (128, 128)  # Input image size
num_classes = 4      # Adjust according to your dataset

# Create training and validation datasets using validation_split
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.15,
    subset='training',
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'  # One-hot encoded labels
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.15,
    subset='validation',
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# Create test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# **Save class names before transformations**
class_names = test_dataset.class_names

# Data augmentation
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

# Apply data augmentation to training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Normalization layer
normalization_layer = Rescaling(1./255)

# Apply normalization to datasets
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Prefetch datasets
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Load base model (VGG16 expects input size of (224, 224, 3))
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

# Unfreeze last few layers for fine-tuning (optional)
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Build the model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Adjusted input shape
    layers.Resizing(224, 224),  # Resize input images to match VGG16 expected input size
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='sigmoid'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model with metrics
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='f1_score')])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10  # Adjust epochs as needed
)

# Plot metrics over epochs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
f1 = history.history['f1_score']
val_f1 = history.history['val_f1_score']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, f1, label='Training F1 Score')
plt.plot(epochs_range, val_f1, label='Validation F1 Score')
plt.legend(loc='lower right')
plt.title('F1 Score')

plt.tight_layout()
plt.show()

# Evaluate on test dataset
test_loss, test_acc, test_f1 = model.evaluate(test_dataset)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')

# Compute precision and recall using scikit-learn
from sklearn.metrics import classification_report

# Get true labels and predictions from the test dataset
y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# **Use the saved class_names variable**
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Clean up
gc.collect()
