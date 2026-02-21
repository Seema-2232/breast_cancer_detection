import os, shutil, random
from sklearn.model_selection import train_test_split

source_dir = r"C:\Users\Administrator\OneDrive\Desktop\breast_cancer_detection\clean_data"
base_out = r"C:\Users\Administrator\OneDrive\Desktop\breast_cancer_detection\dataset"
train_out = os.path.join(base_out, "train")
test_out = os.path.join(base_out, "test")

for p in [train_out, test_out]:
    os.makedirs(p, exist_ok=True)

print("🚀 Splitting dataset...")

for class_name in os.listdir(source_dir):
    src_class = os.path.join(source_dir, class_name)
    if not os.path.isdir(src_class): continue

    images = [os.path.join(src_class, f) for f in os.listdir(src_class) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for img_list, out_dir in [(train_imgs, train_out), (test_imgs, test_out)]:
        dst = os.path.join(out_dir, class_name)
        os.makedirs(dst, exist_ok=True)
        for img in img_list:
            shutil.copy(img, dst)

print("✅ Split complete!")
print(f"Train: {train_out}")
print(f"Test: {test_out}")
