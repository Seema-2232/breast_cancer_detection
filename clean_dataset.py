import os, re, shutil

# Path to original dataset and cleaned destination
base_dir = r"C:\Users\Administrator\OneDrive\Desktop\breast_cancer_detection\merged_data"
clean_dir = r"C:\Users\Administrator\OneDrive\Desktop\breast_cancer_detection\clean_data"

os.makedirs(clean_dir, exist_ok=True)

print("🚀 Cleaning dataset and merging subtypes correctly...")

for root, dirs, _ in os.walk(base_dir):
    for d in dirs:
        src = os.path.join(root, d)
        # Skip top-level benign/malignant folders themselves
        if src == base_dir:
            continue

        # Remove magnification text (_40X, _100X, etc.)
        clean_name = re.sub(r'_\d+X', '', d)
        dst = os.path.join(clean_dir, clean_name)
        os.makedirs(dst, exist_ok=True)

        for subdir, _, files in os.walk(src):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_file = os.path.join(subdir, f)
                    try:
                        shutil.copy(src_file, dst)
                    except Exception as e:
                        print(f"⚠️ Could not copy {src_file}: {e}")

print(f"✅ Clean dataset ready at: {clean_dir}")
