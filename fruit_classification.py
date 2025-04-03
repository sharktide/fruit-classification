import kagglehub

# Download latest version
path = kagglehub.dataset_download("moltean/fruits")

print("Path to dataset files:", path)



import tensorflow as tf

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')


import numpy as np
import os
import pathlib
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.utils import class_weight
from tensorflow.keras.mixed_precision import set_global_policy

# Set the global mixed precision policy
set_global_policy('mixed_float16')


# Set image dimensions
img_height = 224
img_width = 224
batch_size = 256

str_path = '/kaggle/input/fruits/fruits-360_100x100/fruits-360/Training'
data_dir_train = pathlib.Path(str_path)

str_path = '/kaggle/input/fruits/fruits-360_100x100/fruits-360/Test'
data_dir_val = pathlib.Path(str_path)

# Load the datasets without applying cache/prefetch initially
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_val,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Access class_names before any dataset transformation
class_names = train_ds.class_names
print(class_names)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomContrast(0.3),
    layers.RandomTranslation(0.2, 0.2),
])


train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Apply caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 176
# Define the base pre-trained model (MobileNet)

# Define the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model
base_model.trainable = False

# Create the custom model on top of the pre-trained model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),  # Reduce from 512 to 256
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Callbacks
callbacksUsed = [
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
]

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacksUsed
)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:75]:  # Adjust the number of layers to freeze if necessary
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# Save the fine-tuned model
model.save('fruit_detection.keras')
print("Transfer learning with MobileNet completed and model saved.")
