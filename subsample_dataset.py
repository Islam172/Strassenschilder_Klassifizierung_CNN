# Author: Islam Elmaaroufi



import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# === KONFIGURATION ===
SOURCE_DIR = Path("/Users/mac/Downloads/OgreImagesOnly/OgreImagesOnly/Ogre")   # Ursprungsordner
TARGET_DIR = Path("/Users/mac/Desktop/Ogre_subsample") # Zielordner
MAX_PER_CLASS = 200                  # Max. Bilder pro Klasse

# Zielordner neu anlegen
if TARGET_DIR.exists():
    shutil.rmtree(TARGET_DIR)
TARGET_DIR.mkdir(parents=True)

# Bildformate
IMG_EXT = [".png", ".jpg", ".jpeg", ".bmp"]

# Zähle und kopiere
class_counts = defaultdict(int)
total_copied = 0

for class_folder in SOURCE_DIR.iterdir():
    if class_folder.is_dir():
        images = [img for img in class_folder.iterdir() if img.suffix.lower() in IMG_EXT]
        selected = random.sample(images, min(len(images), MAX_PER_CLASS))

        # Zielordner für Klasse
        target_class_dir = TARGET_DIR / class_folder.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in selected:
            shutil.copy(img_path, target_class_dir)
            class_counts[class_folder.name] += 1
            total_copied += 1

# Ergebnis ausgeben
print(f"\n Subsampling abgeschlossen: {total_copied} Bilder kopiert.\n")
for cls, count in sorted(class_counts.items()):
    print(f"{cls}: {count} Bilder")
