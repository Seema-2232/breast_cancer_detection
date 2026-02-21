import os

base_dir = r"C:\Users\Administrator\OneDrive\Desktop\breast_cancer_detection\clean_data"
removed = 0

for folder in os.listdir(base_dir):
    path = os.path.join(base_dir, folder)
    if os.path.isdir(path) and len(os.listdir(path)) == 0:
        os.rmdir(path)
        print(f"🧹 Removed empty folder: {folder}")
        removed += 1

if removed == 0:
    print("✅ No empty folders found.")
else:
    print(f"✅ Cleaned up {removed} empty folder(s).")
