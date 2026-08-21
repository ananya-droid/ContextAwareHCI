import os
import numpy as np

DATASET_DIR = "dataset"

labels = ["HELLO", "YES", "NO", "PLEASE", "THANK_YOU",
          "HELP", "NEED", "WATER", "FOOD", "BYE"]

print("\n==============================")
print("DATASET CHECK")
print("==============================\n")

total = 0

for label in labels:

    folder = os.path.join(DATASET_DIR, label)

    if not os.path.exists(folder):
        print(f"{label:12} -> NOT CREATED")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    valid = 0

    for file in files:
        path = os.path.join(folder, file)

        data = np.load(path)

        if data.shape == (30, 63):
            valid += 1
        else:
            print(f"  WARNING: {file} has shape {data.shape}")

    print(f"{label:12} -> {valid} samples")

    total += valid

print("\n==============================")
print(f"TOTAL VALID SAMPLES: {total}")
print("==============================")