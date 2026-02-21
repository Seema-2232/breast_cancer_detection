import joblib

# Load the broken label map
label_map = joblib.load("model_info.pkl")

# Check what’s inside
if "class_indices" in label_map:
    label_map = label_map["class_indices"]

# Invert dictionary (Keras class_indices are like {'adenosis_40X': 0, ...})
fixed_map = {v: k for k, v in label_map.items()}

# Save the fixed version
joblib.dump(fixed_map, "model_info.pkl")

print("✅ Fixed label map saved!")
print("Total classes:", len(fixed_map))
print("Example:", list(fixed_map.items())[:5])
