import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight

# Dataset Path
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# Image Size
IMG_SIZE = (224, 224)

# Batch Size
BATCH_SIZE = 32

# Number of Epochs
EPOCHS = 10
FINE_TUNE_EPOCHS = 10

# --- FIX: no manual rescale=1./255 — EfficientNet normalizes internally ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nDataset Loaded Successfully\n")
print("Training Images :", train_generator.samples)
print("Validation Images :", validation_generator.samples)
print("Testing Images :", test_generator.samples)
print("\nClass Mapping")
print(train_generator.class_indices)

# --- Class weights to handle imbalance ---
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights_array))
print("\nClass Weights :", class_weights)

# Load EfficientNetB0 without the top classification layer
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the pretrained layers (phase 1)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --- FIX: drop line_length, it crashes in narrow terminals ---
print("\nModel Created Successfully\n")

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

checkpoint_path = os.path.join("model", "pneumonia_model.keras")

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1

)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# --- Phase 1: train the classification head ---
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# --- Phase 2: fine-tune the top of EfficientNet ---
base_model.trainable = True

# Freeze all but the last ~30 layers so early general features stay intact
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # much lower LR for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# --- Evaluate ---
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Loss : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# Save the final trained model
model.save("model/pneumonia_model.keras")
print("\nModel saved successfully!")

# --- Combine history from both phases for plotting ---
acc = history.history["accuracy"] + fine_tune_history.history["accuracy"]
val_acc = history.history["val_accuracy"] + fine_tune_history.history["val_accuracy"]
loss_hist = history.history["loss"] + fine_tune_history.history["loss"]
val_loss_hist = history.history["val_loss"] + fine_tune_history.history["val_loss"]

plt.figure(figsize=(8, 5))
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(loss_hist, label="Training Loss")
plt.plot(val_loss_hist, label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()