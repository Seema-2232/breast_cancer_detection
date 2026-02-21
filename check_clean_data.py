import os

base_dir = r"C:\Users\Administrator\OneDrive\Desktop\breast_cancer_detection\clean_data"

empty_classes = []
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(images) == 0:
        empty_classes.append(folder)

if empty_classes:
    print("⚠️ Empty folders found:")
    for c in empty_classes:
        print("   -", c)
else:
    print("✅ All class folders have images.")
