import os

# Dataset Path
dataset_path = r"C:\Users\admin\Desktop\DIAT\Radar_Dataset"

folders = ["train", "validation", "test"]

print("=" * 50)
print("RADAR DATASET IMAGE COUNT")
print("=" * 50)

for folder in folders:

    print(f"\n{folder.upper()} DATASET")
    print("-" * 30)

    folder_path = os.path.join(dataset_path, folder)

    total = 0

    for cls in sorted(os.listdir(folder_path)):

        class_path = os.path.join(folder_path, cls)

        if os.path.isdir(class_path):

            count = len([
                file for file in os.listdir(class_path)
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ])

            print(f"{cls:15s} : {count}")

            total += count

    print("-" * 30)
    print(f"Total Images   : {total}")