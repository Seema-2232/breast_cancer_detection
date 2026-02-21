import os, time, joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ===========================================================
# CONFIGURATION
# ===========================================================
train_dir = r"C:\Users\Administrator\OneDrive\Desktop\breast_cancer_detection\dataset\train"
val_dir   = r"C:\Users\Administrator\OneDrive\Desktop\breast_cancer_detection\dataset\test"

img_size = (160, 160)        # Smaller size = faster training
batch_size = 8
epochs_stage1 = 10
epochs_stage2 = 5
model_path = "breast_cancer_cnn_model_fast.keras"   # ✅ new format (no .h5)

# ===========================================================
# DATA PIPELINES
# ===========================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)

# ===========================================================
# CLASS WEIGHTS
# ===========================================================
labels = train_gen.classes
classes = np.unique(labels)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
class_weight = dict(zip(classes, weights))
print("⚖️ Class weights computed:", class_weight)

# Save label mapping for app
model_info = {"class_indices": train_gen.class_indices}
joblib.dump(model_info, "model_info.pkl")
print("💾 Saved class mapping → model_info.pkl")

# ===========================================================
# MODEL (MobileNetV3Large)
# ===========================================================
print("🧠 Building MobileNetV3Large model (fast CPU training)...")

base_model = MobileNetV3Large(include_top=False, weights=None, input_shape=(160, 160, 3))

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)
output = layers.Dense(train_gen.num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

print(model.summary())

# ===========================================================
# CALLBACKS
# ===========================================================
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1)
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1)
csv_logger = CSVLogger("training_fast_log.csv", append=True)

callbacks = [checkpoint, early_stop, reduce_lr, csv_logger]

# ===========================================================
# STAGE 1 – Train classification head
# ===========================================================
print("\n🚀 Stage 1 – Training classification head...")
t0 = time.time()

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs_stage1,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

print(f"⏱ Stage 1 complete in {(time.time() - t0)/60:.1f} min")

# ===========================================================
# STAGE 2 – Fine-tune deeper layers
# ===========================================================
print("\n🔧 Stage 2 – Fine-tuning deeper layers...")
base_model.trainable = True
for layer in base_model.layers[:-60]:  # Freeze early layers
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

t1 = time.time()
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs_stage2,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)
print(f"⏱ Stage 2 complete in {(time.time() - t1)/60:.1f} min")

# ===========================================================
# SAVE & PLOTS
# ===========================================================
model.save(model_path)
print(f"\n✅ Final model saved → {model_path}")

def merge_hist(h1, h2):
    merged = {}
    for k in h1.history.keys():
        merged[k] = h1.history[k] + h2.history.get(k, [])
    return merged

history = merge_hist(history1, history2)

plt.figure(figsize=(8,5))
plt.plot(history["accuracy"], label="train acc")
plt.plot(history["val_accuracy"], label="val acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("training_fast_accuracy.png")

plt.figure(figsize=(8,5))
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_fast_loss.png")

print("📊 Saved → training_fast_accuracy.png, training_fast_loss.png")
print("\n🎉 Fast training complete – model ready for Streamlit app.")
