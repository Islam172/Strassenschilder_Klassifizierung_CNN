#Author: Islam Elmaaroufi

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

# Beispielhafte Pfade (bitte ggf. anpassen)
output_dir = "/Users/mac/Desktop/Nachher"

# Geladene Arrays
X_train = np.load(f"{output_dir}/X_train.npy")
y_train = np.load(f"{output_dir}/y_train.npy")

# Label Encoder laden
with open(f"{output_dir}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Bildanzeige vorbereiten
num_samples = 5
fig, axs = plt.subplots(1, num_samples, figsize=(15, 4))

# ImageNet Normalisierungswerte
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Visualisierung
for i in range(num_samples):
    idx = random.randint(0, len(X_train) - 1)
    image = X_train[idx]  # Shape: (C, H, W)
    label = label_encoder.inverse_transform([y_train[idx]])[0]

    # Rücktransformieren
    image = image * std[:, None, None] + mean[:, None, None]  # shape (3, H, W)
    image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 2, 0))  # -> (H, W, C)

    axs[i].imshow(image)
    axs[i].set_title(label)
    axs[i].axis("off")

plt.tight_layout()
plt.show()
