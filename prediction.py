#Author: Gerhard Horst Reglich / Islam Elmaaroufi 

import os
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# === KONFIGURATION ===
MODEL_PATH = "/Users/mac/Desktop/Nachher/models/model_traffic_signs.pth"  # <- Pfad zum Modell
LABEL_ENCODER_PATH = "/Users/mac/Desktop/Nachher/label_encoder.pkl"      # <- Pfad zum LabelEncoder
IMAGE_PATH = "/Users/mac/Desktop/download-2.jpg"                      # <- Bild das du klassifizieren willst
IMAGE_SIZE = (224, 224)

# === Transform (wie beim Training) ===
transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Bild laden und vorbereiten ===
img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

# === LabelEncoder und Modell laden ===
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# === Vorhersage ===
with torch.no_grad():
    output = model(input_tensor)
    predicted_index = torch.argmax(output, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

# === Ergebnis anzeigen ===
print(f"Bildpfad: {IMAGE_PATH}")
print(f"Vorhergesagte Klasse: {predicted_label}")
img.show()
