#Author Islam Elmaaroufi

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === KONFIGURATION ===
data_dir = "/Users/mac/Desktop/Nachher"
model_path = os.path.join(data_dir, "models", "model_traffic_signs2.pth")
label_encoder_path = os.path.join(data_dir, "label_encoder.pkl")
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TESTDATEN LADEN ===
X_test = np.load(os.path.join(data_dir, "X_test.npy"))
y_test = np.load(os.path.join(data_dir, "y_test.npy"))

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

# === LABEL ENCODER LADEN ===
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

# === MODELL KONSTRUIEREN UND LADEN ===
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# === VORHERSAGEN UND METRIKEN ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# === GLOBALE METRIKEN ===
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print("\n📊 TESTDATEN-METRIKEN")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

#Ebgebnisse
"""
📊 TESTDATEN-METRIKEN
==================================================
Accuracy:  0.9658
Precision: 0.9657
Recall:    0.9658
F1-Score:  0.9644

"""




