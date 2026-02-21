import joblib
import tensorflow as tf
import numpy as np

# Load model + label map
model = tf.keras.models.load_model("breast_cancer_cnn_model_fast.keras", compile=False)
label_map = joblib.load("model_info.pkl")

print("🧠 Label Map Keys:", label_map.keys())
print("🧠 Number of labels in map:", len(label_map))
print("🧠 Model output classes:", model.output_shape[-1])